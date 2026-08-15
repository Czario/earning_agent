"""Tests for the period-detection agent (agent/period.py).

The reporting period is decided ONLY by the agent reading the filing document:
  • parsing of finalize_period JSON,
  • business rules (Q4 == annual, sanity window, quarterly must name a quarter),
  • the agent run through the shared loop (custom terminal tool + parser),
  • failure semantics — no deterministic fallback, the run FAILS.
"""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from earnings_agents.agent.period import (
    PeriodDetectionError,
    _parse_period_result,
    apply_period_business_rules,
    run_period_detection,
    detect_period_node,
)

FAKE_DOC = """\
APPLIED MATERIALS ANNOUNCES THIRD QUARTER 2026 RESULTS
SANTA CLARA, Calif., Aug. 13, 2026 - Applied Materials, Inc. (NASDAQ: AMAT)
today reported results for its third quarter ended July 26, 2026.

Three Months Ended          Three Months Ended
July 26, 2026               July 27, 2025
Revenue  $ 9,115            $ 7,509
"""

GOOD_JSON = (
    '{"period_type": "quarterly", "period_end": "2026-07-26", '
    '"quarter": 3, "period_label": "Three Months Ended July 26, 2026"}'
)


# ── _parse_period_result ─────────────────────────────────────────────────────

def test_parse_period_result_valid():
    parsed = _parse_period_result(GOOD_JSON)
    assert parsed is not None
    assert parsed["period_type"] == "quarterly"
    assert parsed["period_end"] == date(2026, 7, 26)
    assert parsed["quarter"] == 3
    assert parsed["period_label"] == "Three Months Ended July 26, 2026"


def test_parse_period_result_strips_fences_and_preamble():
    raw = 'Here is the answer:\n```json\n' + GOOD_JSON + "\n```"
    parsed = _parse_period_result(raw)
    assert parsed is not None and parsed["quarter"] == 3


def test_parse_period_result_quarter_null_for_annual():
    raw = GOOD_JSON.replace('"quarter": 3', '"quarter": null')
    parsed = _parse_period_result(raw)
    assert parsed is not None and parsed["quarter"] is None


def test_parse_period_result_accepts_string_quarter():
    raw = GOOD_JSON.replace('"quarter": 3', '"quarter": "3"')
    parsed = _parse_period_result(raw)
    assert parsed is not None and parsed["quarter"] == 3


@pytest.mark.parametrize("bad", [
    "not json at all",
    '{"period_type": "quarterly"}',                    # missing period_end
    GOOD_JSON.replace('"quarterly"', '"half-yearly"'),  # bad type
    GOOD_JSON.replace('"2026-07-26"', '"July 26"'),     # bad date
    GOOD_JSON.replace('"quarter": 3', '"quarter": 5'),  # bad quarter
    GOOD_JSON.replace('"quarter": 3', '"quarter": "x"'),
])
def test_parse_period_result_rejects_bad_input(bad):
    assert _parse_period_result(bad) is None


@pytest.mark.parametrize("raw_date, expected", [
    ("2026-06-30", date(2026, 6, 30)),
    ("6/30/2026", date(2026, 6, 30)),
    ("06/30/2026", date(2026, 6, 30)),
    ("June 30, 2026", date(2026, 6, 30)),
    ("Jun 30, 2026", date(2026, 6, 30)),
])
def test_parse_period_result_accepts_common_date_formats(raw_date, expected):
    """Agents write dates in many forms — the parser must not reject a valid
    finalize just because period_end isn't ISO."""
    raw = GOOD_JSON.replace('"2026-07-26"', json.dumps(raw_date))
    parsed = _parse_period_result(raw)
    assert parsed is not None
    assert parsed["period_end"] == expected


def test_business_rules_normalize_unparseable_label():
    """A sloppy label like 'Three months ended 6/30/2026' must be normalized to
    a standard header so upsert_concept_values can parse the period date."""
    result, error = apply_period_business_rules(
        _valid_parsed(
            period_end=date(2026, 6, 30),
            period_label="Three months ended 6/30/2026",
        )
    )
    assert error is None
    assert result["period_label"] == "Three Months Ended June 30, 2026"


def test_business_rules_keep_parseable_label():
    result, error = apply_period_business_rules(_valid_parsed())
    assert error is None
    assert result["period_label"] == "Three Months Ended July 26, 2026"


def test_business_rules_normalize_annual_label():
    result, error = apply_period_business_rules(
        _valid_parsed(
            period_type="annual", quarter=None,
            period_end=date(2026, 7, 25),
            period_label="FY ended 7/25/26",
        )
    )
    assert error is None
    assert result["period_label"] == "Fiscal Year Ended July 25, 2026"


# ── apply_period_business_rules ──────────────────────────────────────────────

def _valid_parsed(**overrides):
    base = {
        "period_type": "quarterly",
        "period_end": date(2026, 7, 26),
        "quarter": 3,
        "period_label": "Three Months Ended July 26, 2026",
    }
    base.update(overrides)
    return base


def test_business_rules_quarter_4_coerced_to_annual():
    result, error = apply_period_business_rules(_valid_parsed(quarter=4))
    assert error is None
    assert result["period_type"] == "annual"
    assert result["quarter"] is None


def test_business_rules_fourth_quarter_label_coerced_to_annual():
    result, error = apply_period_business_rules(
        _valid_parsed(quarter=None, period_label="Fourth Quarter 2026 Results")
    )
    assert error is None
    assert result["period_type"] == "annual"
    assert result["quarter"] is None


def test_business_rules_annual_drops_quarter():
    result, error = apply_period_business_rules(
        _valid_parsed(period_type="annual", quarter=2)
    )
    assert error is None
    assert result["period_type"] == "annual"
    assert result["quarter"] is None


def test_business_rules_quarterly_requires_quarter():
    result, error = apply_period_business_rules(_valid_parsed(quarter=None))
    assert result is None
    assert "quarter is null" in error


def test_business_rules_rejects_future_period_end():
    result, error = apply_period_business_rules(
        _valid_parsed(period_end=date(2026, 8, 5)),
        filing_date=date(2026, 7, 27),
    )
    assert result is None and "after the filing date" in error


def test_business_rules_rejects_ancient_period_end():
    result, error = apply_period_business_rules(
        _valid_parsed(period_end=date(2024, 1, 1)),
        filing_date=date(2026, 7, 27),
    )
    assert result is None and "before the filing date" in error


def test_business_rules_accepts_close_dates():
    result, error = apply_period_business_rules(
        _valid_parsed(period_end=date(2026, 7, 26)),
        filing_date=date(2026, 7, 27),
    )
    assert error is None and result["quarter"] == 3


def test_business_rules_without_filing_date_skips_window():
    result, error = apply_period_business_rules(
        _valid_parsed(period_end=date(2026, 7, 26)),
    )
    assert error is None and result["period_type"] == "quarterly"


# ── run_period_detection (agent loop integration) ────────────────────────────

class _FakeChat:
    """Minimal chat-model stub: scripted messages, records bound tools."""

    def __init__(self, script):
        self.script = list(script)
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        if not self.script:
            return AIMessage(content="")
        return self.script.pop(0)


def _tool_call(name, args):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"id-{name}", "type": "tool_call"},
    ])


@pytest.fixture
def fake_chat():
    fake = _FakeChat([
        _tool_call("search", {"query": "Ended"}),
        _tool_call("finalize_period", {"result_json": GOOD_JSON}),
    ])
    with patch("earnings_agents.agent.loop.build_chat_llm", return_value=fake):
        yield fake


def test_run_period_detection_uses_agent_loop_and_returns_period(fake_chat):
    period = run_period_detection(
        FAKE_DOC, ticker="AMAT", company_name="Applied Materials Inc",
        cik="0000006951", fy_end_month=10, fy_end_code="1026",
        filing_date=date(2026, 7, 27),
    )
    assert period == {
        "period_type": "quarterly",
        "period_end": date(2026, 7, 26),
        "quarter": 3,
        "period_label": "Three Months Ended July 26, 2026",
    }
    # The shared loop must have bound the custom terminal tool + document tools.
    names = {t.name for t in fake_chat.bound_tools}
    assert "finalize_period" in names
    assert "search" in names and "read_lines" in names


def test_run_period_detection_fails_without_agent_result():
    """The loop is open-ended — a failing LLM (not a step cap) is what
    terminates a run that never finalizes."""

    class _FailingChat(_FakeChat):
        def invoke(self, messages):
            raise RuntimeError("provider down")

    with patch("earnings_agents.agent.loop.build_chat_llm", return_value=_FailingChat([])):
        with pytest.raises(PeriodDetectionError, match="no result"):
            run_period_detection(FAKE_DOC, ticker="AMAT")


def test_run_period_detection_fails_on_unverifiable_output():
    bad = GOOD_JSON.replace('"quarter": 3', '"quarter": null')  # quarterly w/o quarter
    fake = _FakeChat([_tool_call("finalize_period", {"result_json": bad})])
    with patch("earnings_agents.agent.loop.build_chat_llm", return_value=fake):
        with pytest.raises(PeriodDetectionError, match="quarter is null"):
            run_period_detection(FAKE_DOC, ticker="AMAT")


def test_run_period_detection_recovers_json_from_last_message():
    """The shared loop's recovery regex finds the JSON even without finalize."""
    fake = _FakeChat([AIMessage(content=(
        'The period is quarterly. {"period_type": "quarterly", '
        '"period_end": "2026-07-26", "quarter": 3, '
        '"period_label": "Three Months Ended July 26, 2026"}'
    ))])
    with patch("earnings_agents.agent.loop.build_chat_llm", return_value=fake):
        period = run_period_detection(FAKE_DOC, ticker="AMAT")
    assert period["quarter"] == 3


# ── detect_period_node ───────────────────────────────────────────────────────

def _company():
    return {
        "cik": "0000006951",
        "name": "APPLIED MATERIALS INC",
        "fiscal_year_end_month": 10,
        "fiscal_year_end_code": "1026",
    }


def test_detect_period_node_sets_state_on_success():
    period = {
        "period_type": "quarterly",
        "period_end": date(2026, 7, 26),
        "quarter": 3,
        "period_label": "Three Months Ended July 26, 2026",
    }
    state = {"ticker": "AMAT", "raw_text": FAKE_DOC, "company_name": "Applied"}
    with (
        patch("earnings_agents.integrations.normalize.get_company_by_ticker", return_value=_company()),
        patch("earnings_agents.agent.period.run_period_detection", return_value=period),
    ):
        out = detect_period_node(state)
    assert out["cik"] == "0000006951"
    assert out["fiscal_year_end_month"] == 10
    assert out["detected_period_type"] == "quarterly"
    assert out["detected_quarter"] == 3
    assert out["period_label"] == "Three Months Ended July 26, 2026"
    assert out["sec_report_date"] == "2026-07-26"
    assert out["detected_period"]["quarter"] == 3


def test_detect_period_node_fails_when_agent_fails():
    state = {"ticker": "AMAT", "raw_text": FAKE_DOC, "company_name": "Applied"}
    with (
        patch("earnings_agents.integrations.normalize.get_company_by_ticker", return_value=_company()),
        patch("earnings_agents.agent.period.run_period_detection",
              side_effect=PeriodDetectionError("period agent produced no result for AMAT")),
    ):
        out = detect_period_node(state)
    assert out["status"] == "failed"
    assert "no result" in out["error"]


def test_detect_period_node_parses_string_filing_date():
    """filing_date arrives from state as an ISO string — the node must parse it
    before the business-rule sanity window (regression: E2E TypeError)."""
    from datetime import date as _date

    period = {
        "period_type": "quarterly",
        "period_end": date(2026, 7, 26),
        "quarter": 3,
        "period_label": "Three Months Ended July 26, 2026",
    }
    state = {"ticker": "AMAT", "raw_text": FAKE_DOC, "company_name": "Applied",
             "filing_date": "2026-07-27"}
    with (
        patch("earnings_agents.integrations.normalize.get_company_by_ticker", return_value=_company()),
        patch("earnings_agents.agent.period.run_period_detection", return_value=period) as mock,
    ):
        out = detect_period_node(state)
    assert out["detected_period_type"] == "quarterly"
    assert mock.call_args.kwargs["filing_date"] == _date(2026, 7, 27)


def test_detect_period_node_skips_when_company_missing():
    state = {"ticker": "UNKN", "raw_text": FAKE_DOC, "company_name": "Unknown"}
    with patch("earnings_agents.integrations.normalize.get_company_by_ticker", return_value=None):
        out = detect_period_node(state)
    assert out["status"] == "skipped"
