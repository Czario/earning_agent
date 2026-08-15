"""Period-detection agent — the single source of truth for the reporting period.

Runs BEFORE concept loading: the agent reads the plain-text press release and
reports the current reporting period (quarterly vs annual, period-end date,
quarter 1–3, and the exact column-header label).  The company's fiscal
year-end (MMDD, from ``normalize_data.companies``) is given to the agent as an
anchor.

Design rules:
  • **Agent-only** — no deterministic/regex period inference anywhere in the
    decision path.  If the agent fails or its output is invalid, the run
    FAILS; the worker's job-level retry is the only safety net.
  • **Q4 == annual** — a Fourth Quarter release is the fiscal year-end.
    When a release shows both a Q4 and a fiscal-year column, the fiscal-year
    (annual) column is authoritative — Q4 is never extracted.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from typing import Any

from earnings_agents.agent.loop import run_agent_loop
from earnings_agents.agent.tools import build_pi_tools
from earnings_agents.state import EarningsAgentState

logger = logging.getLogger(__name__)

# Sanity window for the agent-reported period end, relative to the 8-K filing
# date.  A period end must be in the recent past (or at most ~1 week ahead for
# edge drift) — anything else is a wrong column or a hallucination.
_SANITY_MAX_FUTURE_DAYS = 7
_SANITY_MAX_BACK_DAYS = 450

_PERIOD_RECOVERY_RX = re.compile(r'\{[^{}]*"period_type"[^{}]*\}', re.DOTALL)


PERIOD_SYSTEM_PROMPT = """\
You are a period-detection agent.  Your ONLY job is to determine the CURRENT
reporting period of an SEC earnings press release (plain text).

The company's fiscal year-end is {fy_end} (MMDD format — e.g. 1231 = December
31, 0630 = June 30).

STEPS
  1. get_document_info() for an overview — the text may be a BUNDLE of
     exhibits; the FIRST document is the press release carrying the period
     header.  Read the period from there.
  2. search("Ended") or search("quarter") to locate the period header
     (e.g. "Three Months Ended July 26, 2026",
     "results for its third quarter ended July 26, 2026",
     "Fiscal Year Ended July 25, 2026").
  3. read_lines() the header region and read the CURRENT-period column
     header exactly as printed.

RULES — CRITICAL
  • Identify the CURRENT period only — never the prior-year comparison
    column, never guidance/outlook dates, never datelines.
  • Fourth Quarter (Q4) IS the fiscal year-end: report period_type="annual"
    and quarter=null.  We never extract Q4 as a quarter.
  • If the release shows BOTH a fourth-quarter column AND a fiscal-year
    column, report the FISCAL-YEAR (annual) column — never Q4.
  • period_end is the current period's end date — MUST be YYYY-MM-DD
    (e.g. "2026-07-26").
  • period_label is the exact current column header text, and MUST include
    the full month name and year (e.g. "Three Months Ended July 26, 2026",
    "Fiscal Year Ended July 25, 2026").
  • If you cannot find the period, say so in your final JSON as best you can
    — do not guess dates you have not read.

Call finalize_period with a JSON string:
  {{"period_type": "quarterly" | "annual",
    "period_end": "YYYY-MM-DD",
    "quarter": 1|2|3|null,
    "period_label": "exact column header text"}}
"""

FINALIZE_PERIOD_DESCRIPTION = (
    "Call this when you have determined the current reporting period. "
    "Pass a JSON string with:\n"
    '  - period_type: "quarterly" or "annual" (Q4 is ALWAYS annual)\n'
    '  - period_end: "YYYY-MM-DD" (current period end date, e.g. "2026-07-26")\n'
    "  - quarter: 1, 2, 3, or null (null for annual)\n"
    '  - period_label: the exact current column header text with full month '
    'name and year (e.g. "Three Months Ended July 26, 2026")'
)


class PeriodDetectionError(RuntimeError):
    """The period agent failed or produced an unusable answer."""


# ── Result parsing + business rules ─────────────────────────────────────────

def _parse_period_end(raw: Any) -> date | None:
    """Parse an agent-supplied period end, accepting common formats:
    ISO ("2026-06-30"), US slash ("6/30/2026"), and month-name
    ("June 30, 2026" / "Jun 30, 2026")."""
    if isinstance(raw, date):
        return raw
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None
    from earnings_agents.integrations.normalize import parse_period_end_date
    return parse_period_end_date(s)


def _parse_period_result(result_str: str) -> dict[str, Any] | None:
    """Parse the agent's ``finalize_period`` JSON into a typed dict.

    Returns ``None`` when the JSON is malformed or fields are invalid.
    """
    cleaned = (
        result_str.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]
    end_brace = cleaned.rfind("}")
    if end_brace >= 0:
        cleaned = cleaned[: end_brace + 1]
    try:
        parsed: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    period_type = str(parsed.get("period_type", "")).strip().lower()
    if period_type not in ("quarterly", "annual"):
        return None

    period_end = _parse_period_end(parsed.get("period_end"))
    if period_end is None:
        return None

    quarter = parsed.get("quarter")
    if quarter is not None:
        try:
            quarter = int(quarter)
        except (ValueError, TypeError):
            return None
        if quarter not in (1, 2, 3, 4):
            return None

    label = str(parsed.get("period_label", "")).strip() or None

    return {
        "period_type": period_type,
        "period_end": period_end,
        "quarter": quarter,
        "period_label": label,
    }


def apply_period_business_rules(
    parsed: dict[str, Any],
    *,
    filing_date: date | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the agent's period and apply the Q4→annual business rule.

    Returns ``(result, None)`` on success or ``(None, error_message)`` when the
    answer is unusable.  This is a gate, not a fallback — rejection fails the
    run.

    Rules:
      • quarter == 4 or a "fourth quarter" label → coerce to annual, quarter None
      • annual periods never carry a quarter
      • period_end must fall inside the sanity window vs the filing date
      • quarterly periods must name a quarter (1–3)
    """
    period_type = parsed["period_type"]
    period_end: date = parsed["period_end"]
    quarter: int | None = parsed["quarter"]
    label: str | None = parsed["period_label"]

    # Business rule: Q4 == annual.  We never extract a fourth-quarter column.
    label_lower = (label or "").lower()
    if quarter == 4 or "fourth quarter" in label_lower:
        if quarter == 4:
            logger.info("period rules: quarter=4 → coerced to annual")
        else:
            logger.info("period rules: 'Fourth Quarter' label → coerced to annual")
        period_type, quarter = "annual", None

    if period_type == "annual":
        quarter = None

    # Make the label save-safe: upsert_concept_values parses the period date
    # from it.  If the agent's label carries no parseable date (e.g.
    # "Three months ended 6/30/2026"), normalize it to a standard header form
    # built from the validated period_end.
    from earnings_agents.integrations.normalize import parse_period_end_date
    if (
        label
        and parse_period_end_date(label) is None
        and period_end.isoformat() not in label
    ):
        normalized = (
            f"Fiscal Year Ended {period_end.strftime('%B %d, %Y')}"
            if period_type == "annual"
            else f"Three Months Ended {period_end.strftime('%B %d, %Y')}"
        )
        logger.warning(
            "period rules: label %r has no parseable date — normalized to %r",
            label, normalized,
        )
        label = normalized

    if filing_date is not None:
        if period_end > filing_date + timedelta(days=_SANITY_MAX_FUTURE_DAYS):
            return None, (
                f"period_end {period_end.isoformat()} is more than "
                f"{_SANITY_MAX_FUTURE_DAYS} days after the filing date "
                f"{filing_date.isoformat()} — wrong column or hallucination"
            )
        if period_end < filing_date - timedelta(days=_SANITY_MAX_BACK_DAYS):
            return None, (
                f"period_end {period_end.isoformat()} is more than "
                f"{_SANITY_MAX_BACK_DAYS} days before the filing date "
                f"{filing_date.isoformat()} — wrong column or hallucination"
            )

    if period_type == "quarterly" and quarter is None:
        return None, "period_type=quarterly but quarter is null"

    return {
        "period_type": period_type,
        "period_end": period_end,
        "quarter": quarter,
        "period_label": label,
    }, None


# ── Agent run ────────────────────────────────────────────────────────────────

def run_period_detection(
    raw_text: str,
    *,
    ticker: str,
    company_name: str = "",
    company_industry: dict | None = None,
    cik: str | None = None,
    fy_end_month: int | None = None,
    fy_end_code: str | None = None,
    filing_date: date | None = None,
    document_map: list[dict] | None = None,
) -> dict[str, Any]:
    """Run the period-detection agent over *raw_text*.

    *document_map* (when the filing is a multi-exhibit bundle) is surfaced by
    ``get_document_info`` so the agent knows the first document is the press
    release carrying the period header.  *company_industry* feeds the cached
    ``get_company_info`` tool (the period prompt itself does not use it).

    Raises :class:`PeriodDetectionError` when the agent produces nothing usable.
    Returns the validated period dict:
    ``{period_type, period_end, quarter, period_label}``.
    """
    fy_end = fy_end_code or (f"{fy_end_month:02d}00" if fy_end_month else "unknown")
    system_prompt = PERIOD_SYSTEM_PROMPT.format(fy_end=fy_end)

    initial_message = (
        f"This is a {len(raw_text):,}-character SEC earnings press release. "
        "Determine the current reporting period, then call finalize_period."
    )

    tools = build_pi_tools(
        raw_text, prior_values={}, cik=cik, company_name=company_name,
        company_industry=company_industry,
        document_map=document_map,
    )

    result = run_agent_loop(
        system_prompt=system_prompt,
        initial_message=initial_message,
        tools=tools,
        ticker=ticker,
        finalize_name="finalize_period",
        finalize_description=FINALIZE_PERIOD_DESCRIPTION,
        parse_final_result=_parse_period_result,
        recovery_regex=_PERIOD_RECOVERY_RX,
    )
    if result is None:
        raise PeriodDetectionError(
            f"period agent produced no result for {ticker}"
        )

    validated, error = apply_period_business_rules(
        result, filing_date=filing_date,
    )
    if validated is None:
        raise PeriodDetectionError(f"period agent output rejected: {error}")

    logger.info(
        "period detection for %s: type=%s end=%s quarter=%s label=%r",
        ticker, validated["period_type"], validated["period_end"],
        validated["quarter"], validated["period_label"],
    )
    return validated


# ── Graph node ───────────────────────────────────────────────────────────────

def detect_period_node(state: EarningsAgentState) -> EarningsAgentState:
    """Agent-only period detection.  Failure → ``status="failed"`` (no fallback)."""
    from earnings_agents.hooks import report_call

    ticker = state["ticker"]
    raw_text = state.get("raw_text") or ""

    # The period agent needs the company's fiscal year-end as an anchor.
    from earnings_agents.integrations.normalize import get_company_by_ticker
    try:
        company = get_company_by_ticker(ticker)
    except Exception as exc:  # noqa: BLE001
        company = None
        report_call(f"  [period]  ✗ company lookup failed — {str(exc)[:60]}")
    if company is None:
        report_call(f"  [period]  ✗ {ticker} not in normalize_data — cannot anchor period")
        return {
            **state,
            "status": "skipped",
            "error": (
                f"No historical data for {ticker} in normalize_data — "
                f"we don't have historical data for the company so we can't proceed."
            ),
        }

    report_call(f"  [period]  agent period detection ({ticker})")
    filing_date = state.get("filing_date")
    if isinstance(filing_date, str):
        try:
            filing_date = date.fromisoformat(filing_date)
        except ValueError:
            filing_date = None
    try:
        period = run_period_detection(
            raw_text,
            ticker=ticker,
            company_name=state.get("company_name") or ticker,
            company_industry=company.get("industry"),
            cik=company.get("cik"),
            fy_end_month=company.get("fiscal_year_end_month"),
            fy_end_code=company.get("fiscal_year_end_code"),
            filing_date=filing_date,
            document_map=state.get("document_map"),
        )
    except PeriodDetectionError as exc:
        report_call(f"  [period]  ✗ {exc}")
        logger.error("detect_period_node: %s", exc)
        return {**state, "status": "failed", "error": str(exc)}

    report_call(
        f"  [period]  ✓ {period['period_type']}  "
        f"{period['period_label'] or period['period_end'].isoformat()}"
    )
    return {
        **state,
        "cik": company["cik"],
        "company_industry": company.get("industry") or {},
        "fiscal_year_end_month": company["fiscal_year_end_month"],
        "fiscal_year_end_code": company.get("fiscal_year_end_code"),
        "detected_period_type": period["period_type"],
        "detected_quarter": period["quarter"],
        "period_label": period["period_label"],
        "sec_report_date": period["period_end"].isoformat(),
        "detected_period": {
            "period_type": period["period_type"],
            "period_end": period["period_end"].isoformat(),
            "quarter": period["quarter"],
            "period_label": period["period_label"],
        },
    }
