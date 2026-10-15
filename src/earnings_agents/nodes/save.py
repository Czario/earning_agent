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
# number/scale, wrong segment parent, wrong-company document, truncated exhibit)
# remain blocking.  Currency is a per-value filter below, not a blocker.
_ABSENCE_ONLY_FINDING_TYPES = frozenset({
    "missing_concept",         # agent: searched but could not locate
    "cashflow_not_deaccumulated",  # un-deaccumulated cash flow concept never blocks save
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

    # Currency is resolved per value by the extraction pipeline (USD preferred,
    # native fallback, no conversion).  A value whose currency cannot be
    # confirmed is dropped just below — never written as USD and never a
    # whole-run blocker.  `currency` is only the document-level default used
    # when a value carries no per-value currency.
    currency_meta = state.get("currency_metadata") or {}
    currency = str(currency_meta.get("currency") or "").strip().upper() or None

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

    concept_metrics: dict = dict(state.get("concept_metrics") or {})
    value_metadata_by_id: dict = dict(state.get("value_metadata_by_id") or {})
    derived_ids: set[str] = set(state.get("derived_concept_ids") or [])
    cik: str | None = state.get("cik")
    # The period label and all period identity come from the canonical period
    # agent result.  Extraction metadata is never allowed to provide a period.
    period_str: str = period.period_label or ""

    # Materialize only agent-proposed, grounded new rows, and only after the
    # save/currency gates above pass. Existing concepts were already checked by
    # the extraction pipeline; this save-time upsert is the idempotent backstop.
    new_candidate_issues = list(state.get("new_concept_issues") or [])
    candidates = state.get("new_concept_candidates") or []
    if candidates and cik:
        from earnings_agents.integrations.normalize import persist_new_concept_candidates
        try:
            persisted_new, persist_issues = persist_new_concept_candidates(
                cik=cik,
                period=period,
                candidates=candidates,
            )
            new_candidate_issues.extend(persist_issues)
            for candidate in candidates:
                metric_key = str(candidate.get("metric_key") or "")
                persisted = persisted_new.get(metric_key)
                if not persisted:
                    continue
                concept_id = persisted["concept_id"]
                concept_metrics[concept_id] = float(candidate["value"])
                value_metadata_by_id[concept_id] = {
                    "statement_type": candidate.get("statement_type") or "income",
                    "dimension": False,
                    "dimension_concept": bool(candidate.get("dimension_concept")),
                    "dimension_member": (
                        persisted["concept"].get("dimension_member") or ""
                    ),
                    "dimension_member_label": (
                        candidate.get("dimension_member_label") or ""
                    ),
                    "dimension_axis": candidate.get("dimension_axis") or "",
                    "calculated": False,
                    "currency": candidate.get("currency") or currency or "",
                    "scale": candidate.get("scale") or "as-is",
                    "source_lines": candidate.get("source_lines") or [],
                    "status": "new_concept",
                }
                if candidate.get("source_snippet"):
                    value_metadata_by_id[concept_id]["source_snippet"] = candidate["source_snippet"]
        except Exception as exc:  # noqa: BLE001 — new concepts must not lose mapped values
            logger.exception("Could not persist new concept candidates for %s", ticker)
            new_candidate_issues.append(f"new concept persistence failed: {exc}")

        if new_candidate_issues:
            report_call(
                f"  [save]  ⚠ {len(new_candidate_issues)} new concept candidate(s) "
                "not created; other mapped values continue"
            )

    # ── Currency backstop (never persist an unconfirmed currency) ────────
    # The extraction pipeline already drops values that do not match their
    # statement's resolved currency.  This is the write-time guarantee: no FX
    # conversion exists anywhere, and USD is never assumed, so a value without
    # a confirmed ISO code is omitted instead of being stored under a guess.
    from earnings_agents.agent.currency import is_currency_code
    dropped_currency: list[str] = []
    for cid in list(concept_metrics):
        meta = value_metadata_by_id.get(cid) or {}
        code = str(meta.get("currency") or "").strip().upper()
        if is_currency_code(code):
            continue
        if is_currency_code(currency):
            meta["currency"] = currency
            value_metadata_by_id[cid] = meta
            continue
        concept_metrics.pop(cid, None)
        derived_ids.discard(cid)
        value_metadata_by_id.pop(cid, None)
        dropped_currency.append(cid)
    if dropped_currency:
        report_call(
            f"  [save]  ⚠ omitted {len(dropped_currency)} value(s) with an "
            "unconfirmed currency (no conversion, USD never assumed)"
        )
        findings = findings + [{
            "type": "currency_unresolved",
            "severity": "medium",
            "message": (
                f"{len(dropped_currency)} value(s) omitted at save — currency "
                "could not be confirmed"
            ),
            "evidence": {"concept_ids": dropped_currency[:20]},
        }]

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
                target_statements=state.get("target_statements"),
                derived_concept_ids=derived_ids,
                value_metadata_by_id=value_metadata_by_id,
                accession_number=state.get("accession_number"),
                source_form_type=state.get("source_form_type"),
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

    return {
        **state,
        "status": "saved",
        "findings": findings,
        "concept_metrics": concept_metrics,
        "value_metadata_by_id": value_metadata_by_id,
        "new_concept_issues": new_candidate_issues,
    }
