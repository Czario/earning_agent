"""MongoDB save node — upsert concept_metrics into normalize_data.

Refuses to save when accounting identity checks failed and
``STRICT_ACCURACY`` is enabled (default).
"""
from __future__ import annotations

import logging
from earnings_agents.config import STRICT_ACCURACY
from earnings_agents.state import EarningsAgentState
from earnings_agents.agent.period import (
    format_period_label,
    require_detected_period,
)

logger = logging.getLogger(__name__)


def mongodb_save_node(state: EarningsAgentState) -> EarningsAgentState:
    """Upsert extracted concept metrics into normalize_data."""
    from earnings_agents.hooks import report_call
    ticker = state["ticker"]
    findings = state.get("findings") or []
    high_unresolved = [
        f for f in findings
        if isinstance(f, dict) and f.get("severity") == "high"
    ]

    if STRICT_ACCURACY and high_unresolved:
        parts: list[str] = [
            f"{len(high_unresolved)} unresolved high-severity finding(s): "
            + "; ".join(str(f.get("message"))[:80] for f in high_unresolved[:5])
        ]
        msg = f"Refusing to save {ticker}: " + " | ".join(parts)
        report_call(f"  [save]  ✗ refusing to save — run marked failed")
        logger.error(msg)
        return {**state, "status": "failed", "error": msg}

    try:
        period = require_detected_period(state)
    except Exception as exc:
        msg = f"Refusing to save {ticker}: invalid period-agent result: {exc}"
        report_call(f"  [save]  ✗ {msg}")
        return {**state, "status": "failed", "error": msg}

    metrics = state.get("metrics") or {}
    sec_rd = period.period_end

    if high_unresolved:
        logger.warning(
            "Saving %s with %d unresolved high-severity finding(s): %s",
            ticker,
            len(high_unresolved),
            [f.get("message") for f in high_unresolved],
        )

    concept_metrics: dict = state.get("concept_metrics") or {}
    derived_ids: set[str] = set(state.get("derived_concept_ids") or [])
    cik: str | None = state.get("cik")
    # The period label and all period identity come from the canonical period
    # agent result.  Extraction metadata is never allowed to provide a period.
    period_str: str = period.period_label or ""

    if concept_metrics and cik:
        pending = state.get("_pending_replace") or {}
        replace_note = ""
        if pending.get("cik"):
            from earnings_agents.integrations.normalize import delete_fiscal_period
            pd_cik = pending["cik"]
            n_del = delete_fiscal_period(pd_cik, period)
            if n_del:
                period_label = format_period_label(period)
                replace_note = f"replacing {period_label} — deleted {n_del} stale; "
                logger.info(
                    "save: deleted %d stale concept(s) for CIK %s %s",
                    n_del, pd_cik, period_label,
                )

        from earnings_agents.integrations.normalize import upsert_concept_values
        n_mapped = len(concept_metrics) - len(derived_ids)
        n_derived = len(derived_ids)
        report_call(
            f"  [save]  {replace_note}upserting {n_mapped} mapped + {n_derived} derived "
            f"concept(s) for CIK {cik} — {period_str or sec_rd.isoformat()}"
        )
        try:
            n = upsert_concept_values(
                cik=cik,
                company_name=state["company_name"],
                concept_metrics=concept_metrics,
                period=period,
                derived_concept_ids=derived_ids,
                accession_number=state.get("accession_number"),
            )
            logger.info(
                "normalize_data: upserted %d concept value(s) for %s", n, ticker
            )
        except Exception as exc:
            report_call(f"  [save]  ✗ upsert failed: {exc}")
            return {**state, "status": "failed", "error": f"normalize_data upsert failed: {exc}"}
    else:
        reason_parts = []
        if not concept_metrics: reason_parts.append("no concept_metrics")
        if not cik: reason_parts.append("no CIK")
        report_call(f"  [save]  skipped — {', '.join(reason_parts)}")
        logger.warning(
            "Skipping normalize_data upsert for %s — "
            "missing concept_metrics=%s cik=%s period=%r",
            ticker,
            bool(concept_metrics),
            cik,
            period_str,
        )

    return {**state, "status": "saved"}
