"""MongoDB save node — upsert concept_metrics into normalize_data.

Refuses to save when accounting identity checks failed and
``STRICT_ACCURACY`` is enabled (default).
"""
from __future__ import annotations

import logging
from datetime import date as _date

from earnings_agents.config import STRICT_ACCURACY
from earnings_agents.state import EarningsAgentState

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

    metrics = state.get("metrics") or {}
    sec_report_date_str: str | None = state.get("sec_report_date")
    sec_rd: _date | None = None
    if sec_report_date_str:
        try:
            sec_rd = _date.fromisoformat(sec_report_date_str)
        except ValueError:
            pass

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
    fy_end_month: int | None = state.get("fiscal_year_end_month")
    fy_end_code: str = str(state.get("fiscal_year_end_code") or "1231")
    # Period label comes from the PERIOD AGENT (detected from the document
    # header); the extraction agent's __period__ is only a fallback.
    period_str: str = str(
        state.get("period_label") or (metrics.get("__period__") or "")
    )
    detected_period_type: str | None = state.get("detected_period_type")

    if concept_metrics and cik and fy_end_month and (period_str or sec_rd):
        pending = state.get("_pending_replace") or {}
        if pending.get("cik") and pending.get("fiscal_year"):
            from earnings_agents.integrations.normalize import delete_fiscal_period
            pd_cik = pending["cik"]
            pd_fy = pending["fiscal_year"]
            pd_q = pending.get("quarter")
            n_del = delete_fiscal_period(pd_cik, pd_fy, pd_q)
            if n_del:
                period_label = (
                    f"FY{pd_fy} Q{pd_q}" if pd_q is not None else f"FY{pd_fy} (annual)"
                )
                report_call(
                    f"  [save]  deleted {n_del} stale concept(s) for {period_label}"
                )
                logger.info(
                    "save: deleted %d stale concept(s) for CIK %s %s",
                    n_del, pd_cik, period_label,
                )

        from earnings_agents.integrations.normalize import upsert_concept_values
        n_mapped = len(concept_metrics) - len(derived_ids)
        n_derived = len(derived_ids)
        report_call(
            f"  [save]  upserting {n_mapped} mapped + {n_derived} derived "
            f"concept(s) for CIK {cik} — {period_str or sec_report_date_str or '?'}"
        )
        try:
            n = upsert_concept_values(
                cik=cik,
                company_name=state["company_name"],
                concept_metrics=concept_metrics,
                period_str=period_str,
                fiscal_year_end_month=fy_end_month,
                fiscal_year_end_code=fy_end_code,
                report_date=sec_rd,
                period_type_override=detected_period_type,
                derived_concept_ids=derived_ids,
                accession_number=state.get("accession_number"),
            )
            report_call(f"  [save]  ✓ {n} concept value(s) upserted")
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
        if not fy_end_month: reason_parts.append("no fy_end_month")
        if not period_str and not sec_rd: reason_parts.append("no period date")
        report_call(f"  [save]  skipped — {', '.join(reason_parts)}")
        logger.warning(
            "Skipping normalize_data upsert for %s — "
            "missing concept_metrics=%s cik=%s fy_end_month=%s period=%r",
            ticker,
            bool(concept_metrics),
            cik,
            fy_end_month,
            period_str,
        )

    return {**state, "status": "saved"}
