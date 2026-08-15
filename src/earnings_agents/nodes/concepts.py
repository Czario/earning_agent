"""Load company GAAP concepts from normalize_data before extraction.

Looks up the company by ticker in normalize_data.companies, then fetches
income-statement concepts from the appropriate normalized_concepts collection:
  - ``normalized_concepts_quarterly`` for quarterly filings (most 8-Ks)
  - ``normalized_concepts_annual``    for annual filings (Q4 / year-end 8-Ks)

The period type is NOT inferred here — it comes from the period agent
(``detect_period_node``, state field ``detected_period_type``), which read it
from the filing document.  Q4 is always annual.

Populates ``cik``, ``target_concepts``, ``fiscal_year_end_month``,
and ``recent_concept_ids`` in state so the agent pipeline
can build a targeted prompt.

Failure is always graceful: if the company is not found or the DB is
unreachable the node falls back to ``target_concepts=[]`` and lets the
generic income-statement extraction proceed.  It never sets ``status=failed``.
"""
from __future__ import annotations

import logging

from earnings_agents.config import PROMPT_HISTORY_PERIODS
from earnings_agents.integrations.normalize import (
    get_company_by_ticker,
    get_recently_valued_concept_ids,
    get_statement_concepts,
)
from earnings_agents.state import EarningsAgentState

logger = logging.getLogger(__name__)


def load_company_concepts_node(state: EarningsAgentState) -> EarningsAgentState:
    """Load GAAP concepts for targeted extraction from normalize_data.

    Targeted extraction requires stored historical concepts for the ticker.
    When the company is absent from normalize_data, the DB is unreachable, or
    no income-statement concepts are stored, the run is *skipped*
    (``status="skipped"``) with a clear error message — we do not fall back to
    generic extraction.
    """
    ticker = state["ticker"]

    def _skip(message: str, **extra: object) -> EarningsAgentState:
        from earnings_agents.hooks import report_call
        report_call(f"  [load concepts]  ✗ skipped — {message[:80]}")
        logger.info("load_company_concepts: %s", message)
        skipped = {
            **state,
            "status": "skipped",
            "error": message,
            "target_concepts": [],
            "calculated_concepts": [],
            "cik": None,
            "fiscal_year_end_month": None,
            "fiscal_year_end_code": None,
            "detected_period_type": "quarterly",
        }
        skipped.update(extra)
        return skipped  # type: ignore[return-value]

    try:
        company = get_company_by_ticker(ticker)
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"No historical data for {ticker}: normalize_data lookup failed "
            f"({exc}); we don't have historical data for the company so we "
            f"can't proceed."
        )

    if company is None:
        return _skip(
            f"No historical data for {ticker} in normalize_data — we don't have "
            f"historical data for the company so we can't proceed."
        )

    cik: str = company["cik"]
    fy_end_month: int = company["fiscal_year_end_month"]
    fy_end_code: str | None = company.get("fiscal_year_end_code")
    sec_report_date_str: str | None = state.get("sec_report_date")  # type: ignore[assignment]

    # Period type — decided upstream by the PERIOD AGENT (detect_period_node),
    # which read the current column header from the filing document.  Q4 is
    # always annual.  No inference happens here.
    period_type: str = state.get("detected_period_type") or "quarterly"

    logger.info(
        "load_company_concepts: %s (CIK %s) — period_type=%s (from period agent, "
        "sec_report_date=%s)",
        ticker, cik, period_type, sec_report_date_str,
    )

    # ── 1. Recent-value window FIRST — the rule: extract ONLY concepts that
    # had a stored value in any of the last PROMPT_HISTORY_PERIODS periods
    # (quarterly → concept_values_quarterly, annual → concept_values_annual).
    # The window is computed before any concept loading so the concept query
    # itself is restricted to it — non-recent concepts are never even loaded.
    # (system:/calculated concepts are exempt: always loaded for derivation.)
    try:
        recent = get_recently_valued_concept_ids(
            cik, period_type=period_type, n_periods=PROMPT_HISTORY_PERIODS
        )
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"Recent-value lookup failed for {ticker} ({exc}) — cannot build "
            f"the extraction target.",
            cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period_type=period_type,
        )

    recent_concept_ids: list[str] = sorted(recent)
    if not recent_concept_ids:
        return _skip(
            f"No concepts valued in the last {PROMPT_HISTORY_PERIODS} "
            f"{period_type} periods for {ticker} — nothing to extract.",
            cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period_type=period_type,
        )

    # ── 2. Load ONLY the recently-valued concepts (query-level filter) ─────
    try:
        concepts = get_statement_concepts(
            cik,
            statement_types=["income"],
            period_type=period_type,
            concept_ids=recent_concept_ids,
        )
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"No historical data for {ticker}: concept query failed ({exc}); "
            f"we don't have historical data for the company so we can't proceed.",
            cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period_type=period_type,
        )

    from earnings_agents.hooks import report_call
    report_call(
        f"  [load concepts]  loaded {len(concepts)} income-statement concept(s) "
        f"({period_type}) — last-{PROMPT_HISTORY_PERIODS}-period window "
        f"(+ system/calculated exempt)"
    )
    logger.info(
        "load_company_concepts: loaded %d income-statement concept(s) for %s "
        "(CIK %s, %s, recent-%d-period window + system/calculated exempt)",
        len(concepts), ticker, cik, period_type, PROMPT_HISTORY_PERIODS,
    )

    if not concepts:
        return _skip(
            f"No income-statement concepts stored for {ticker} in normalize_data "
            f"for the recent-value window — we can't proceed.",
            cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period_type=period_type,
        )

    return {
        **state,
        "cik": cik,
        "target_concepts": concepts,
        "recent_concept_ids": recent_concept_ids,
        "calculated_concepts": [],
        "fiscal_year_end_month": fy_end_month,
        "fiscal_year_end_code": company.get("fiscal_year_end_code"),
        "detected_period_type": period_type,
    }
