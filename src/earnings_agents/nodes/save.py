"""MongoDB save node — upsert concept_metrics into normalize_data.

Refuses to save when accounting identity checks failed and
``STRICT_ACCURACY`` is enabled (default).
"""
from __future__ import annotations

import logging
from earnings_agents.state import EarningsAgentState
from earnings_agents.agent.period import (
    format_period_label,
    require_detected_period,
)

logger = logging.getLogger(__name__)

# Findings that concern metrics ABSENT from the extraction (the agent searched
# but could not locate them) must NEVER block the save — whatever WAS found is
# still correct and gets persisted; a few not-found metrics must not drop the
# whole period.  Only findings that corrupt the values being stored (wrong
# number/scale/currency, wrong segment parent, wrong-company document,
# truncated exhibit, non-USD currency) remain blocking.
_ABSENCE_ONLY_FINDING_TYPES = frozenset({
    "missing_concept",         # agent: searched but could not locate
})


def mongodb_save_node(state: EarningsAgentState) -> EarningsAgentState:
    """Upsert extracted concept metrics into normalize_data."""
    from earnings_agents.hooks import report_call
    # Read the flag LAZILY from the config module so a runtime override (the
    # CLI's --allow-inconsistent flag sets config.STRICT_ACCURACY = False)
    # takes effect here — an import-time copy would silently ignore it.
    from earnings_agents import config as _config

    ticker = state["ticker"]
    findings = state.get("findings") or []
    high_unresolved = [
        f for f in findings
        if isinstance(f, dict)
        and f.get("severity") == "high"
        and f.get("type") not in _ABSENCE_ONLY_FINDING_TYPES
    ]

    # USD-only is a hard invariant, not bypassable by relaxed accuracy:
    # a confirmed foreign or mixed-currency document is never saved as USD.
    currency_meta = state.get("currency_metadata") or {}
    currency = currency_meta.get("currency")
    if currency != "USD":
        msg = (
            f"Refusing to save {ticker}: currency is {currency or 'unknown'}, "
            "not explicitly confirmed as USD"
        )
        report_call(f"  [save]  ✗ {msg}")
        logger.error(msg)
        return {**state, "status": "failed", "error": msg}

    if _config.STRICT_ACCURACY and high_unresolved:
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
        # The replace is ATOMIC and handled inside upsert_concept_values:
        # all values are upserted FIRST (with a unique per-save save_token),
        # then a stale sweep deletes any doc of this exact fiscal period that
        # does not carry the new token.  There is deliberately NO
        # delete-before-write step here — a failed/interrupted write must never
        # leave the period empty.
        pending = state.get("_pending_replace") or {}
        replace_note = ""
        if pending.get("cik"):
            replace_note = f"replacing {format_period_label(period)}; "

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
                value_metadata_by_id=state.get("value_metadata_by_id"),
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
