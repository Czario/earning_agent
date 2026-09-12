"""Guidance save node — persist `guidance_values` records + score actuals.

Runs AFTER a successful income-statement save (graph routes here only when
``mongodb_save`` returned ``status="saved"``) so the accession trace is
consistent: guidance and actuals for a filing are written in the same run.

  • upsert the agent's normalized records (``state.guidance_records``) into
    ``guidance_values`` — keyed on (cik, accession_number, metric, basis,
    period), with is_current demotion across accessions, ``supersedes``, and
    ``source="manual"`` write-protection.
  • score() the company's CURRENT guidance docs whose covered period has
    arrived (this filing reported it): resolve standard_label → concept row
    via the existing mapping vocabulary, read the stored actual, write
    ``result {outcome, delta_abs, delta_pct, ...}`` back onto the doc.

NEVER fails a run: guidance is forward-looking, independent data — any
problem is logged and summarised in ``state.guidance_save`` (observability,
mirroring ``q4_calculation``).  Disabled via ``GUIDANCE_ENABLED=0`` → the
agent never extracts guidance and this node no-ops.
"""
from __future__ import annotations

import logging
from typing import Any

from earnings_agents.state import EarningsAgentState

logger = logging.getLogger(__name__)


def save_guidance_node(state: EarningsAgentState) -> EarningsAgentState:
    from earnings_agents.hooks import report_call
    from earnings_agents.config import GUIDANCE_ENABLED as _GUIDANCE_ENABLED

    ticker = state["ticker"]
    summary: dict[str, Any] = {"status": "skipped"}

    if not _GUIDANCE_ENABLED:
        report_call(f"  [guidance]  disabled (GUIDANCE_ENABLED=0) — nothing to save")
        return {**state, "guidance_save": summary}

    records = state.get("guidance_records") or []
    if not records:
        summary["status"] = "no_records"
        return {**state, "guidance_save": summary}

    cik = state.get("cik")
    if not cik:
        summary["status"] = "no_cik"
        return {**state, "guidance_save": summary}

    try:
        from earnings_agents.agent.period import require_detected_period
        period = require_detected_period(state)
    except Exception as exc:
        summary.update({"status": "failed", "error": f"no valid detected period: {exc}"})
        logger.warning("save_guidance: %s", summary["error"])
        report_call(f"  [guidance]  ✗ {summary['error']}")
        return {**state, "guidance_save": summary}

    accession = state.get("accession_number")
    # This pipeline processes 8-K Exhibit 99.1 press releases (the worker
    # queue is sec:filings:8k); a manual URL run also carries the 8-K text.
    form_type = str(state.get("form_type") or "8-K")

    from earnings_agents.integrations.guidance import (
        upsert_guidance_records,
        score_guidance_for_cik,
    )

    try:
        upsert_summary = upsert_guidance_records(
            cik,
            records,
            accession_number=accession,
            form_type=form_type,
            filing_period=period,
        )
        summary.update({"status": "saved", **upsert_summary})
        report_call(
            f"  [guidance]  ✓ upserted {upsert_summary['upserted']}/{len(records)} "
            f"guidance record(s) for {ticker} — "
            f"{upsert_summary['skipped_manual']} manual-protected skipped, "
            f"{upsert_summary['demoted']} prior doc(s) demoted"
        )
    except Exception as exc:  # noqa: BLE001 — guidance persistence never fails the run
        summary.update({"status": "failed", "error": f"upsert failed: {exc}"})
        logger.warning("save_guidance: upsert failed for %s: %s", ticker, exc, exc_info=True)
        report_call(f"  [guidance]  ✗ upsert failed: {exc}")
        return {**state, "guidance_save": summary}

    # Score current guidance whose covered period has arrived with actuals.
    if summary["status"] == "saved":
        try:
            score_summary = score_guidance_for_cik(cik, period)
            summary["score"] = score_summary
            if score_summary["scored"]:
                report_call(
                    f"  [guidance]  ✓ scored {score_summary['scored']} guidance "
                    f"record(s) against stored actuals ({score_summary['checked']} checked)"
                )
        except Exception as exc:  # noqa: BLE001
            summary["score"] = {"status": "failed", "error": str(exc)}
            logger.warning("save_guidance: scoring failed for %s: %s", ticker, exc)

    return {**state, "guidance_save": summary}