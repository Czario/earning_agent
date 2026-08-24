"""Load company GAAP concepts from normalize_data before extraction.

Looks up the company by ticker in normalize_data.companies, then fetches
income-statement concepts from the appropriate normalized_concepts collection:
  - ``normalized_concepts_quarterly`` for quarterly filings (most 8-Ks)
  - ``normalized_concepts_annual``    for annual filings (Q4 / year-end 8-Ks)

The period type is NOT inferred here — it comes from the period agent
(``detect_period_node``, canonical state field ``detected_period``), which read
it from the filing document.  Q4 is always annual.

Populates ``cik``, ``target_concepts``, ``fiscal_year_end_month``,
and ``recent_concept_ids`` in state so the agent pipeline
can build a targeted prompt.

Failure is always graceful: if the company is not found or the DB is
unreachable the node falls back to ``target_concepts=[]`` and lets the
generic income-statement extraction proceed.  It never sets ``status=failed``.
"""
from __future__ import annotations

import logging
import re

from earnings_agents.config import PROMPT_HISTORY_PERIODS
from earnings_agents.integrations.normalize import (
    get_company_by_ticker,
    get_recently_valued_concept_ids,
    get_statement_concepts,
)
from earnings_agents.agent.period import require_detected_period
from earnings_agents.state import EarningsAgentState

logger = logging.getLogger(__name__)


# Malformed-row detector for upstream normalizer pollution.  Rows whose label
# carries no words (e.g. label "404" from a page number / footnote marker —
# observed live on PDD) are not financial metrics: no filing prints them, so
# the extraction agent can never find them.  A real income-statement row always
# has an alphabetic word in its label.
_GARBAGE_LABEL_RX = re.compile(r"^[\d\s.,()/%-]+$")


def _is_garbage_concept(c: dict) -> bool:
    """True when a concept row carries no alphabetic word in its label."""
    label = (c.get("label") or "").strip()
    if not label:
        return True
    return bool(_GARBAGE_LABEL_RX.match(label))


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
            # No period fallback on a skipped path; the period agent remains
            # the only source of period throughout the pipeline.
            "detected_period": state.get("detected_period"),
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

    # The period type is decided upstream by the period agent, which read the
    # current column header from the filing.  Q4 is always annual.  No period
    # fallback or inference happens here.
    try:
        period = require_detected_period(state)
    except Exception as exc:
        return _skip(
            f"Period agent did not provide a valid period for {ticker}: {exc}",
            cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=fy_end_code,
        )
    period_type = period.period_type
    period_end_str = period.period_end.isoformat()

    logger.info(
        "load_company_concepts: %s (CIK %s) — period_type=%s (from period agent, "
        "period_end=%s)",
        ticker, cik, period_type, period_end_str,
    )

    # ── 1. Load the FULL eligible concept universe ─────────────────────────
    # Recent history is a prioritization signal, not an eligibility filter:
    # new segments, breakdowns, and newly disclosed rows must remain
    # extractable even when they had no stored value in the recent window.
    try:
        concepts = get_statement_concepts(
            cik,
            statement_types=["income"],
            period=period,
        )
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"No historical data for {ticker}: concept query failed ({exc}); "
            f"we don't have historical data for the company so we can't proceed.",
            cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period=state.get("detected_period"),
        )

    if not concepts:
        return _skip(
            f"No income-statement concepts stored for {ticker} in normalize_data "
            f"— we can't proceed.",
            cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period=state.get("detected_period"),
        )

    # ── 1b. Drop malformed concept rows (upstream normalizer pollution) ────
    n_raw = len(concepts)
    concepts = [c for c in concepts if not _is_garbage_concept(c)]
    if len(concepts) < n_raw:
        from earnings_agents.hooks import report_call
        report_call(
            f"  [load concepts]  dropped {n_raw - len(concepts)} malformed "
            "concept row(s) (numeric/garbage labels — upstream pollution)"
        )
        logger.info(
            "load_company_concepts: dropped %d malformed concept row(s) for %s",
            n_raw - len(concepts), ticker,
        )

    if not concepts:
        return _skip(
            f"No usable income-statement concepts for {ticker} in "
            "normalize_data (all rows malformed) — we can't proceed.",
            cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period=state.get("detected_period"),
        )

    # ── 2. Recent-value window → prioritization (best-effort) ─────────────
    try:
        recent = get_recently_valued_concept_ids(
            cik, period=period, n_periods=PROMPT_HISTORY_PERIODS
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_company_concepts: recent-value lookup failed for %s (%s) — "
            "falling back to the full universe", ticker, exc,
        )
        recent = set()
    recent_concept_ids: list[str] = sorted(recent)

    def _is_calculated(c: dict) -> bool:
        return bool(c.get("calculated")) or str(
            c.get("concept") or c.get("taxonomy_key") or ""
        ).lower().startswith("system:")

    def _is_dimensional(c: dict) -> bool:
        return bool(c.get("dimension") or c.get("dimension_concept"))

    # Target = recent concepts ∪ all dimensional rows ∪ system/calculated.
    # When no history exists (bootstrap), use the full universe.
    if recent_concept_ids:
        target = [
            c for c in concepts
            if c["_id"] in recent or _is_dimensional(c) or _is_calculated(c)
        ]
    else:
        target = concepts

    if not target:
        return _skip(
            f"No eligible income-statement concepts for {ticker} — nothing to extract.",
            cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period=state.get("detected_period"),
        )

    from earnings_agents.hooks import report_call
    n_recent = sum(1 for c in target if c["_id"] in recent)
    n_discovery = len(target) - n_recent
    report_call(
        f"  [load concepts]  loaded {len(target)}/{len(concepts)} income-statement "
        f"concept(s) ({period_type}) — {n_recent} recent + {n_discovery} discovery "
        f"(new segments/breakdowns + system/calculated)"
    )
    logger.info(
        "load_company_concepts: targeted %d of %d income-statement concept(s) for %s "
        "(CIK %s, %s; %d recent, %d discovery)",
        len(target), len(concepts), ticker, cik, period_type, n_recent, n_discovery,
    )

    return {
        **state,
        "cik": cik,
        "company_industry": company.get("industry") or {},
        "target_concepts": target,
        "recent_concept_ids": recent_concept_ids,
        "calculated_concepts": [],
        "fiscal_year_end_month": fy_end_month,
        "fiscal_year_end_code": company.get("fiscal_year_end_code"),
        "detected_period": state.get("detected_period"),
    }
