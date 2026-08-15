"""Industry context — advisory SIC data for the extraction agent.

Locks the contract:
  • ``normalize_data.companies.industry`` ({sic_code, sic_description})
    reaches the extraction prompt directly — not only through a voluntary
    ``get_company_info()`` call;
  • ``get_company_by_ticker`` returns it;
  • the graph nodes populate ``state["company_industry"]``;
  • missing industry data never fails a run;
  • the context is advisory-only — it can never change the concept target
    or invent values.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from earnings_agents.agent.industry import (
    build_industry_context,
    normalize_company_industry,
)


TESLA_INDUSTRY = {
    "sic_code": "3711",
    "sic_description": "Motor Vehicles & Passenger Car Bodies",
}


# ── normalize_company_industry ───────────────────────────────────────────────

def test_normalize_industry_returns_clean_profile():
    assert normalize_company_industry(TESLA_INDUSTRY) == TESLA_INDUSTRY


def test_normalize_industry_strips_and_rejects_garbage():
    assert normalize_company_industry(
        {"sic_code": " 3711 ", "sic_description": "  Auto  ", "category": "junk"}
    ) == {"sic_code": "3711", "sic_description": "Auto"}
    assert normalize_company_industry(None) is None
    assert normalize_company_industry("3711") is None
    assert normalize_company_industry({}) is None
    assert normalize_company_industry({"sic_code": " ", "sic_description": ""}) is None


# ── build_industry_context ───────────────────────────────────────────────────

def test_context_contains_sic_and_advisory_guardrails():
    block = build_industry_context(TESLA_INDUSTRY)
    assert "SIC: 3711" in block
    assert "Motor Vehicles & Passenger Car Bodies" in block
    assert "advisory" in block.lower()
    assert "not in the concept list" in block
    assert "infer financial values" in block


def test_context_when_industry_missing_is_explicit():
    block = build_industry_context(None)
    assert "unavailable" in block.lower()
    assert "Do not infer" in block


# ── get_company_by_ticker ────────────────────────────────────────────────────

def _mock_companies_db(doc):
    mock_col = MagicMock()
    mock_col.find_one.return_value = doc
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)
    return mock_db, mock_col


def test_get_company_by_ticker_returns_industry():
    from earnings_agents.integrations import normalize as ndc

    mock_db, mock_col = _mock_companies_db({
        "cik": "0001318605",
        "name": "Tesla, Inc.",
        "corporate_info": {"fiscal_year_end": "1231"},
        "industry": TESLA_INDUSTRY,
    })
    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        result = ndc.get_company_by_ticker("tsla")

    assert result["industry"] == TESLA_INDUSTRY
    assert result["fiscal_year_end_month"] == 12
    assert "industry" in mock_col.find_one.call_args.args[1]


def test_get_company_by_ticker_missing_industry_is_empty_dict():
    from earnings_agents.integrations import normalize as ndc

    mock_db, _ = _mock_companies_db({
        "cik": "000123",
        "name": "No Industry Co",
        "corporate_info": {"fiscal_year_end": "0630"},
    })
    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        result = ndc.get_company_by_ticker("NOIND")

    assert result["industry"] == {}


# ── graph nodes populate company_industry ────────────────────────────────────

def test_load_concepts_populates_company_industry():
    from earnings_agents.nodes import concepts as node

    company = {"cik": "000123", "fiscal_year_end_month": 12,
               "fiscal_year_end_code": "1231", "name": "Example Co",
               "industry": TESLA_INDUSTRY}
    state = {"ticker": "EXMP", "sec_report_date": "2026-07-26",
             "detected_period_type": "quarterly"}
    with (
        patch.object(node, "get_company_by_ticker", return_value=company),
        patch.object(node, "get_recently_valued_concept_ids", return_value={"x"}),
        patch.object(node, "get_statement_concepts",
                     return_value=[{"_id": "x", "concept": "us-gaap:C",
                                    "label": "Label", "path": "001",
                                    "statement_type": "income"}]),
    ):
        result = node.load_company_concepts_node(state)

    assert result["company_industry"] == TESLA_INDUSTRY


def test_detect_period_node_populates_company_industry():
    from earnings_agents.agent.period import detect_period_node

    period = {
        "period_type": "quarterly",
        "period_end": date(2026, 7, 26),
        "quarter": 3,
        "period_label": "Three Months Ended July 26, 2026",
    }
    company = {
        "cik": "0000006951",
        "name": "APPLIED MATERIALS INC",
        "fiscal_year_end_month": 10,
        "fiscal_year_end_code": "1026",
        "industry": {"sic_code": "3674",
                     "sic_description": "Semiconductors & Related Devices"},
    }
    state = {"ticker": "AMAT", "raw_text": "doc text", "company_name": "Applied"}
    with (
        patch("earnings_agents.integrations.normalize.get_company_by_ticker",
              return_value=company),
        patch("earnings_agents.agent.period.run_period_detection",
              return_value=period) as mock,
    ):
        out = detect_period_node(state)

    assert out["company_industry"] == company["industry"]
    assert mock.call_args.kwargs["company_industry"] == company["industry"]


# ── extraction prompt injection ──────────────────────────────────────────────

def _pipeline_state(**overrides):
    state = {
        "ticker": "TSLA",
        "company_name": "Tesla, Inc.",
        "cik": "0001318605",
        "company_industry": TESLA_INDUSTRY,
        "target_concepts": [{"_id": "a", "concept": "us-gaap:Revenue",
                             "label": "Revenue", "path": "001",
                             "statement_type": "income"}],
        "recent_concept_ids": ["a"],
        "raw_text": "Revenue\n$1,000\n",
        "extraction_attempts": 0,
        "detected_period_type": "quarterly",
        "sec_report_date": "2026-07-26",
        "period_label": "Three Months Ended July 26, 2026",
        "fiscal_year_end_code": "1231",
    }
    state.update(overrides)
    return state


def _run_pipeline_with_fake_loop(state):
    """Run agent_document_pipeline_node with all heavy deps patched.

    Returns ``(output_state, captured_system_prompt)``.
    """
    from earnings_agents.agent import pipeline

    captured: dict[str, str] = {}

    def _fake_loop(system_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        return {"[us-gaap:Revenue]": 1_000_000, "__scale__": "as-is"}

    with (
        patch.object(pipeline, "prescan_document", return_value=("as-is", None)),
        patch.object(pipeline, "load_prior_values", return_value={}),
        patch.object(pipeline, "build_concept_list",
                     return_value='• [us-gaap:Revenue] — "Revenue"'),
        patch.object(pipeline, "build_pi_tools", return_value=[]),
        patch.object(pipeline, "run_agent_loop", side_effect=_fake_loop),
        patch.object(pipeline, "map_concepts",
                     return_value=({"a": 1_000_000.0}, {"a": "[us-gaap:Revenue]"},
                                   {"[us-gaap:Revenue]"})),
        patch("earnings_agents.agent.derive.derive_missing_concepts",
              return_value=({"a": 1_000_000.0}, set())),
    ):
        out = pipeline.agent_document_pipeline_node(state)

    return out, captured["system_prompt"]


def test_pipeline_injects_industry_context_into_system_prompt():
    out, prompt = _run_pipeline_with_fake_loop(_pipeline_state())

    assert "SIC: 3711" in prompt
    assert "Motor Vehicles & Passenger Car Bodies" in prompt
    assert "advisory" in prompt.lower()
    assert out["status"] == "extracted"
    assert out["concept_metrics"] == {"a": 1_000_000.0}


def test_pipeline_prompt_marks_missing_industry_explicitly():
    state = _pipeline_state()
    del state["company_industry"]
    out, prompt = _run_pipeline_with_fake_loop(state)

    assert "unavailable" in prompt.lower()
    assert out["status"] == "extracted"


def test_pipeline_passes_industry_profile_to_tools():
    from earnings_agents.agent import pipeline

    state = _pipeline_state()
    with (
        patch.object(pipeline, "prescan_document", return_value=("as-is", None)),
        patch.object(pipeline, "load_prior_values", return_value={}),
        patch.object(pipeline, "build_concept_list",
                     return_value='• [us-gaap:Revenue] — "Revenue"'),
        patch.object(pipeline, "build_pi_tools", return_value=[]) as tools_mock,
        patch.object(pipeline, "run_agent_loop",
                     return_value={"[us-gaap:Revenue]": 1_000_000,
                                   "__scale__": "as-is"}),
        patch.object(pipeline, "map_concepts",
                     return_value=({"a": 1_000_000.0}, {"a": "[us-gaap:Revenue]"},
                                   {"[us-gaap:Revenue]"})),
        patch("earnings_agents.agent.derive.derive_missing_concepts",
              return_value=({"a": 1_000_000.0}, set())),
    ):
        pipeline.agent_document_pipeline_node(state)

    assert tools_mock.call_args.kwargs["company_industry"] == TESLA_INDUSTRY


# ── get_company_info tool ────────────────────────────────────────────────────

def test_get_company_info_returns_cached_profile_without_db():
    from earnings_agents.agent.tools import build_pi_tools
    from earnings_agents.integrations import normalize as ndc

    with patch.object(ndc, "_get_client",
                      side_effect=AssertionError("DB must not be queried")):
        tools = build_pi_tools("doc", prior_values={}, cik="0001318605",
                               company_name="Tesla, Inc.",
                               company_industry=TESLA_INDUSTRY)
        info = next(t for t in tools if t.name == "get_company_info")
        out = info.invoke({})

    assert "Motor Vehicles & Passenger Car Bodies" in out
    assert "SIC 3711" in out
    assert "Tesla, Inc." in out


def test_get_company_info_falls_back_to_db_without_cached_profile():
    from earnings_agents.agent.tools import build_pi_tools
    from earnings_agents.integrations import normalize as ndc

    mock_db, _ = _mock_companies_db({
        "cik": "0001318605",
        "name": "Tesla, Inc.",
        "corporate_info": {"fiscal_year_end": "1231", "entity_type": "operating"},
        "market_info": {"exchanges": ["Nasdaq"]},
        "industry": TESLA_INDUSTRY,
    })
    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        tools = build_pi_tools("doc", prior_values={}, cik="0001318605",
                               company_name="Tesla, Inc.")
        info = next(t for t in tools if t.name == "get_company_info")
        out = info.invoke({})

    assert "Motor Vehicles & Passenger Car Bodies" in out
    assert "Fiscal Year End: 1231" in out
    assert "Nasdaq" in out


def test_worker_publishes_industry_lines_with_call_industry_kind():
    """Regression: after .strip() the message starts with [industry] — the
    worker kind check must match the stripped form."""
    from earnings_agents.progress import make_call_callback

    class _Pub:
        def __init__(self):
            self.events = []

        def publish(self, node, msg, ms=None, kind="call"):
            self.events.append((msg.strip(), kind))

    pub = _Pub()
    cb = make_call_callback(pub, "TSLA", [0])
    cb("  [industry]  context injected — SIC 3711 (X)")
    cb("  [tool]  step 1 → search()")
    cb("  [llm]  agent step 1  → calling llm  (deepseek)")

    kinds = [kind for _, kind in pub.events]
    assert kinds == ["call_industry", "call", "call_llm"]

