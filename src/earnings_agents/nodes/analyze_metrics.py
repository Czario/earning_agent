"""``analyze_metrics`` — single point of post-extraction quality control.

Runs pure-Python checkers from ``earnings_agents.analysis`` and produces:

  * ``state["findings"]``  — list of ``Finding.to_dict()`` for downstream nodes
                              and persistence.
  * ``state["extraction_notes"]`` — structured hint string consumed by
                              ``extract_financial_metrics`` on re-extract.
  * ``state["needs_reextract"]`` — True when a high-severity finding exists
                              AND ``extraction_attempts`` remain below
                              ``MAX_EXTRACTION_ATTEMPTS`` AND new findings
                              differ from the previous pass (progress detected).

This node *replaces* ``reflect_metrics`` in the graph.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from earnings_agents.analysis.critical_metrics import check_presence as presence_summary
from earnings_agents.analysis.findings import (
    Finding,
    derive_corrected_total_opex,
    check_presence,
    check_source_grounding,
)
from earnings_agents.analysis.skills import compute_skill_effectiveness, iter_detectors
from earnings_agents.config import MAX_EXTRACTION_ATTEMPTS
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)


def _apply_identity_corrections(
    findings: list[Finding],
    metrics: dict[str, Any],
    state: EarningsAgentState,
) -> list[Finding]:
    """Correct extracted values that fail accounting-identity checks.

    When the LLM extracts a value that violates a deterministic accounting
    identity, compute the correct value and update both ``metrics`` and
    ``concept_metrics`` so the corrected value is saved.

    Identities:
        1. Gross Profit      = Revenue − Cost of Revenue
        2. Operating Income  = Gross Profit − Total OpEx
        3. Net Income        = Pre-tax Income − Tax Expense

    Returns a new findings list with identity violations replaced by
    ``auto_corrected`` entries.
    """
    import re as _re
    from dataclasses import replace as _dc_replace
    from earnings_agents.analysis.calculators import identify_role as _identify_concept_role

    _out_findings: list[Finding] = [f for f in findings]

    # ── Helpers: find metric keys in raw metrics by regex ──────────────────
    def _find_key(pattern: str) -> str | None:
        for key in metrics:
            if _re.search(pattern, key, _re.I):
                return key
        return None

    def _find_concept_id(label_pattern: str) -> str | None:
        for c in (state.get("target_concepts") or []):
            if _re.search(label_pattern, c.get("label", ""), _re.I):
                return c.get("_id")
        return None

    def _apply_correction(metric_key: str | None, computed_val: float,
                          label_regex: str, method: str) -> None:
        """Update both raw ``metrics`` and ``concept_metrics`` with corrected value."""
        _old_val = None
        if metric_key:
            _old_val = metrics.get(metric_key)
            metrics[metric_key] = computed_val
        _cm: dict[str, Any] | None = state.get("concept_metrics")  # type: ignore[assignment]
        _cid = _find_concept_id(label_regex) if label_regex else None
        if _cm and _cid and _cid in _cm:
            _cm[_cid] = computed_val
            _derived: list[str] = list(state.get("derived_concept_ids") or [])
            if _cid not in _derived:
                _derived.append(_cid)
                state["derived_concept_ids"] = _derived  # type: ignore[typeddict-unknown-key]
        logger.info(
            "analyze_metrics %s: corrected %s %s → %.0f (%s)",
            state.get("ticker", "?"),
            metric_key or label_regex or "?",
            f"{_old_val:,.0f}" if isinstance(_old_val, (int, float)) else "missing",
            computed_val, method,
        )
        _out_findings.append(
            Finding(
                type="auto_corrected",
                severity="low",
                message=f"{metric_key or label_regex or 'metric'} corrected to {computed_val:,.0f} ({method}).",
                keys=(metric_key,) if metric_key else (),
                evidence={"corrected_value": computed_val, "method": method},
            )
        )

    # ═══════════════════════════════════════════════════════════════════════
    # A. RAW METRICS CORRECTION — regex-based (pre-industry, general path)
    #    Searches the raw LLM-extracted keys for GP/OI/NI patterns and
    #    corrects both metrics and concept_metrics.
    # ═══════════════════════════════════════════════════════════════════════

    # ── A1. Gross Profit = Revenue − Cost of Revenue ──────────────────────
    _gp_key = _find_key(r"^(?:gross (?:margin|profit))")
    _rev_key = _find_key(r"^(?:(?:total|net)\s+)?(?:revenue|sales)")
    _cor_key = _find_key(r"^cost\s+of\s+(?:revenue|sales|goods)")
    if _gp_key and _rev_key and _cor_key:
        _rev = metrics.get(_rev_key)
        _cor = metrics.get(_cor_key)
        _gp = metrics.get(_gp_key)
        if isinstance(_rev, (int, float)) and isinstance(_cor, (int, float)):
            _computed = _rev - _cor
            if not isinstance(_gp, (int, float)) or abs(_gp - _computed) > 1.0:
                _apply_correction(_gp_key, _computed, r"gross.profit",
                                  "Revenue − Cost of Revenue")
                _out_findings = [
                    f for f in _out_findings
                    if not (f.type == "identity_violation"
                            and "gross_profit" in (f.evidence or {}))
                ]

    # ── A2. Operating Income = Gross Profit − Total OpEx ───────────────────
    _oi_key = _find_key(r"^(?:income\s+from\s+operations|operating\s+(?:income|profit|loss))")
    _opex_key = _find_key(r"(?:total\s+)?operating\s+(?:expenses?|costs?)")
    _gp_key2 = _find_key(r"^(?:gross (?:margin|profit))")
    if _oi_key and _opex_key and _gp_key2:
        _gp2 = metrics.get(_gp_key2)
        _opex = metrics.get(_opex_key)
        _oi = metrics.get(_oi_key)
        if isinstance(_gp2, (int, float)) and isinstance(_opex, (int, float)):
            _computed = _gp2 - _opex
            if not isinstance(_oi, (int, float)) or abs(_oi - _computed) > 1.0:
                _apply_correction(_oi_key, _computed, r"operat(?:ing|ional)\s+income",
                                  "Gross Profit − Total Operating Expenses")
                _out_findings = [
                    f for f in _out_findings
                    if not (f.type == "suspect_value"
                            and any("operating" in k.lower()
                                    for k in (f.keys or ())))
                ]

    # ── A3. Net Income = Pre-tax Income − Tax Expense ─────────────────────
    _ni_key = _find_key(r"^net\s+(?:income|earnings|loss)\s*$")
    _pt_key = _find_key(r"(?:income\s+before.*tax|pre.?tax\s+income)")
    _tax_key = _find_key(
        r"(?:provision\s+for\s+income\s+tax"
        r"|income\s+tax\s+(?:expense|provision)"
        r")"
    )
    if _ni_key and _pt_key and _tax_key:
        _pt = metrics.get(_pt_key)
        _tax = metrics.get(_tax_key)
        _ni = metrics.get(_ni_key)
        if isinstance(_pt, (int, float)) and isinstance(_tax, (int, float)):
            _computed = _pt - _tax
            if not isinstance(_ni, (int, float)) or abs(_ni - _computed) > 1.0:
                _apply_correction(_ni_key, _computed, r"net.income",
                                  "Pre-tax Income − Tax Expense")

    # ═══════════════════════════════════════════════════════════════════════
    # B. CONCEPT-LEVEL CORRECTION — role-based (industry-aware path)
    #    Checks identities on concept_metrics (concept_id → value), catching
    #    cases where Tier 2 mapping assigned a wrong key to a concept.
    # ═══════════════════════════════════════════════════════════════════════
    _cm: dict[str, Any] | None = state.get("concept_metrics")  # type: ignore[assignment]
    _targets = state.get("target_concepts") or []

    _cid_to_role: dict[str, str] = {}
    for c in _targets:
        role = _identify_concept_role(
            c.get("label", ""), c.get("taxonomy_key") or c.get("concept") or "")
        if role:
            _cid_to_role[c["_id"]] = role

    def _cm_val(role: str) -> float | None:
        for cid, r in _cid_to_role.items():
            if r == role and cid in (_cm or {}):
                v = _cm[cid]
                if isinstance(v, (int, float)):
                    return v
        return None

    if _cm:
        # ── B1. Gross Profit = Revenue − Cost of Revenue ──────────────────
        _rev = _cm_val("revenue")
        _cor = _cm_val("cost_of_revenue")
        if _rev is not None and _cor is not None:
            _gp_computed = _rev - _cor
            for cid, role in _cid_to_role.items():
                if role == "gross_profit" and cid in _cm:
                    _gp_current = _cm[cid]
                    if not isinstance(_gp_current, (int, float)) or abs(_gp_current - _gp_computed) > 1.0:
                        _cm[cid] = _gp_computed
                        _derived = list(state.get("derived_concept_ids") or [])
                        if cid not in _derived:
                            _derived.append(cid)
                            state["derived_concept_ids"] = _derived  # type: ignore[typeddict-unknown-key]
                        logger.info(
                            "analyze_metrics %s: corrected GP (concept) %.0f → %.0f (Revenue − CoR)",
                            state.get("ticker", "?"), _gp_current or 0, _gp_computed)
                        _out_findings.append(
                            Finding(
                                type="auto_corrected", severity="low",
                                message=f"Gross Profit corrected to {_gp_computed:,.0f} (Revenue − Cost of Revenue).",
                                evidence={"corrected_value": _gp_computed, "method": "Revenue − Cost of Revenue"},
                            ))
                    break

        # ── B2. Operating Income = Gross Profit − Total OpEx ──────────────
        _gp2 = _cm_val("gross_profit")
        _opex = _cm_val("total_opex")
        if _gp2 is not None and _opex is not None:
            _oi_computed = _gp2 - _opex
            for cid, role in _cid_to_role.items():
                if role == "operating_income" and cid in _cm:
                    _oi_current = _cm[cid]
                    if not isinstance(_oi_current, (int, float)) or abs(_oi_current - _oi_computed) > 1.0:
                        _cm[cid] = _oi_computed
                        _derived = list(state.get("derived_concept_ids") or [])
                        if cid not in _derived:
                            _derived.append(cid)
                            state["derived_concept_ids"] = _derived  # type: ignore[typeddict-unknown-key]
                        logger.info(
                            "analyze_metrics %s: corrected OI (concept) %.0f → %.0f (GP − OpEx)",
                            state.get("ticker", "?"), _oi_current or 0, _oi_computed)
                        _out_findings.append(
                            Finding(
                                type="auto_corrected", severity="low",
                                message=f"Operating Income corrected to {_oi_computed:,.0f} (Gross Profit − OpEx).",
                                evidence={"corrected_value": _oi_computed, "method": "Gross Profit − Total OpEx"},
                            ))
                    break

        # ── B3. Net Income = Pre-tax Income − Tax Expense ─────────────────
        _pt = _cm_val("pretax_income")
        _tax = _cm_val("tax_expense")
        if _pt is not None and _tax is not None:
            _ni_computed = _pt - _tax
            for cid, role in _cid_to_role.items():
                if role == "net_income" and cid in _cm:
                    _ni_current = _cm[cid]
                    if not isinstance(_ni_current, (int, float)) or abs(_ni_current - _ni_computed) > 1.0:
                        _cm[cid] = _ni_computed
                        _derived = list(state.get("derived_concept_ids") or [])
                        if cid not in _derived:
                            _derived.append(cid)
                            state["derived_concept_ids"] = _derived  # type: ignore[typeddict-unknown-key]
                        logger.info(
                            "analyze_metrics %s: corrected NI (concept) %.0f → %.0f (Pre-tax − Tax)",
                            state.get("ticker", "?"), _ni_current or 0, _ni_computed)
                        _out_findings.append(
                            Finding(
                                type="auto_corrected", severity="low",
                                message=f"Net Income corrected to {_ni_computed:,.0f} (Pre-tax Income − Tax Expense).",
                                evidence={"corrected_value": _ni_computed, "method": "Pre-tax Income − Tax Expense"},
                            ))
                    break

    # ═══════════════════════════════════════════════════════════════════════
    # C. INDUSTRY-SPECIFIC CHECKS
    # ═══════════════════════════════════════════════════════════════════════
    _ind = state.get("company_industry") or {}
    _cat = _ind.get("category", "general") if isinstance(_ind, dict) else "general"

    # Banking: NII = Interest Income − Interest Expense
    if _cat == "banking" and _cm:
        _int_inc = _cm_val("interest_income")
        _int_exp = _cm_val("interest_expense")
        if _int_inc is not None and _int_exp is not None:
            _nii_computed = _int_inc - abs(_int_exp)
            for cid, role in _cid_to_role.items():
                if role == "gross_profit" and cid in _cm:
                    _gp_current = _cm[cid]
                    if not isinstance(_gp_current, (int, float)) or abs(_gp_current - _nii_computed) > 1.0:
                        _cm[cid] = _nii_computed
                        _derived = list(state.get("derived_concept_ids") or [])
                        if cid not in _derived:
                            _derived.append(cid)
                            state["derived_concept_ids"] = _derived  # type: ignore[typeddict-unknown-key]
                        logger.info(
                            "analyze_metrics %s: corrected NII/GP %.0f → %.0f (Interest Income − Interest Expense, banking)",
                            state.get("ticker", "?"), _gp_current or 0, _nii_computed)
                        _out_findings.append(
                            Finding(
                                type="auto_corrected", severity="low",
                                message=f"Gross Profit corrected to {_nii_computed:,.0f} (Interest Income − Interest Expense — banking).",
                                evidence={"corrected_value": _nii_computed, "method": "Interest Income − Interest Expense"},
                            ))
                    break

    # ── Cleanup: demote high findings for identities we just corrected ────
    _corrected_roles: set[str] = set()
    for f in _out_findings:
        if f.type == "auto_corrected":
            ev = f.evidence or {}
            if isinstance(ev, dict):
                _corrected_roles.add(ev.get("method", ""))
    if _corrected_roles:
        _out_findings = [
            _dc_replace(f, severity="medium")
            if f.severity == "high"
            and (
                (f.type == "identity_violation"
                 and "Revenue − Cost of Revenue" in _corrected_roles)
                or (f.type == "suspect_value"
                    and any("operating" in k.lower() for k in (f.keys or ()))
                    and "Gross Profit − Total OpEx" in _corrected_roles)
            )
            else f
            for f in _out_findings
        ]

    return _out_findings


def _build_extraction_notes(
    findings: list[Finding],
    attempt_num: int,
    prev_notes: str,
    missing_toplevel: list[str] | None = None,
    missing_segments: list[str] | None = None,
) -> str:
    """Build a focused hint block for the next extraction pass.

    Only ``high`` and ``medium`` findings are surfaced — ``low`` findings
    (e.g. case duplicates) are handled deterministically downstream.

    The previous pass's notes are prepended so the model has cumulative
    context on what has already been attempted and still failed.
    """
    high = [f for f in findings if f.severity == "high"]
    medium = [f for f in findings if f.severity == "medium"]

    lines: list[str] = []
    if prev_notes and attempt_num > 1:
        lines.append(
            f"[Attempt {attempt_num - 1} history] {prev_notes.strip()}"
        )
        lines.append("")

    lines.append(
        f"Attempt {attempt_num} still incomplete. Prioritise finding these metrics "
        f"(search the income statement and supplementary FINANCIAL DATA tables only):"
    )
    for f in high:
        lines.append(f"  - [REQUIRED] {f.message}")
        if f.suggested_action:
            lines.append(f"      → {f.suggested_action}")
        # For income-statement identity violations, tell the LLM exactly
        # which value it got wrong and what the implied correct value is.
        # This turns a vague "identity broken" message into a concrete hint:
        # "look for a Cost of revenue row close to X, not Y".
        if f.type == "identity_violation":
            ev = f.evidence or {}
            rev = ev.get("revenue")
            cor = ev.get("cost_of_revenue")
            gp  = ev.get("gross_profit")
            if (
                isinstance(rev, (int, float)) and rev
                and isinstance(gp, (int, float))
            ):
                implied_cor = rev - gp
                lines.append(f"      DIAGNOSIS — your previous extraction returned:")
                lines.append(f"        Revenue         = {rev:>22,.0f}  (reference)")
                if isinstance(cor, (int, float)):
                    lines.append(
                        f"        Cost of revenue = {cor:>22,.0f}  ← WRONG — taken from wrong column/row"
                    )
                lines.append(f"        Gross profit    = {gp:>22,.0f}  (reference)")
                lines.append(
                    f"      Revenue − Gross profit = {rev:,.0f} − {gp:,.0f} = {implied_cor:,.0f}"
                )
                lines.append(
                    f"      That means Cost of revenue in the current-period column should be"
                )
                lines.append(
                    f"      approximately {implied_cor:,.0f} — locate that row and use it."
                )
    if medium:
        lines.append("Also locate these if reported:")
        for f in medium:
            lines.append(f"  - {f.message}")

    # Concept-level gap hints: tell the LLM specifically which concepts had no
    # value mapped in the previous pass so it knows exactly what is still missing.
    missing_toplevel = missing_toplevel or []
    missing_segments = missing_segments or []
    if missing_toplevel:
        lines.append(
            "MISSING top-level IS concepts (look in the primary income statement):"
        )
        for lbl in missing_toplevel[:10]:  # cap to avoid runaway hints
            lines.append(f"  • {lbl}")
    if missing_segments:
        lines.append(
            "MISSING segment/dimensional concepts (look in FINANCIAL DATA tables — "
            "segment, geographic, or product breakdowns):"
        )
        for lbl in missing_segments[:20]:
            lines.append(f"  • {lbl}")

    lines.append(
        "Preserve the exact wording each metric uses in the document."
    )
    return "\n".join(lines)


def analyze_metrics_node(state: EarningsAgentState) -> EarningsAgentState:
    """Run all deterministic analysis checkers and decide on re-extract."""
    from earnings_agents.hooks import report_call
    if state.get("status") == "failed":
        return state

    metrics: dict[str, Any] = state.get("metrics") or {}
    ticker = state.get("ticker", "?")

    # Findings from the previous analysis pass (carried on state across the
    # extract↔analyze loop). Used purely for skill-effectiveness observation;
    # never feeds back into routing (ADR-0001).
    prev_findings: list[dict[str, Any]] = state.get("findings") or []

    if not metrics:
        logger.warning("analyze_metrics: no metrics on state for %s", ticker)
        return {**state, "needs_reextract": False}

    presence = presence_summary(metrics.keys())
    findings: list[Finding] = []
    findings.extend(check_presence(metrics, presence))

    # Registry loop — pure observers only (ADR-0003). check_presence is called
    # separately above because it requires a pre-computed presence summary.
    for checker in iter_detectors():
        findings.extend(checker(metrics))

    # Source-grounding ("show me") verification. Called explicitly (not via the
    # registry) because it needs the per-metric source snippets and the source
    # text in addition to the metrics dict. Degrades to a no-op when no
    # snippets were captured.
    findings.extend(
        check_source_grounding(
            metrics,
            state.get("metric_source_snippets"),
            state.get("raw_text") or "",
        )
    )

    # ── Momentum check for Net Interest Income (banks) ────────────────────
    # Banks present Net Interest Income as a primary revenue component.
    # A >50% QoQ swing is almost always a column-mixup error (reading from
    # a prior-period or YTD column).  This check queries the database for the
    # prior period's value and flags implausible changes.
    _net_interest_keys = [k for k in metrics if re.search(r'net\s+interest', k, re.I)]
    if _net_interest_keys and state.get('company_cik'):
        try:
            from earnings_agents.tools.normalize_data_client import _get_client, _NORMALIZE_DB
            _db = _get_client()[_NORMALIZE_DB]
            _cik = state['company_cik']
            _period_type = state.get('detected_period_type', 'quarterly')
            _col_name = 'concept_values_quarterly' if _period_type == 'quarterly' else 'concept_values_annual'
            # Find the target_concept_ids that map to these net interest keys
            _target = state.get('target_concepts') or []
            _ni_concept_ids = [
                c['_id'] for c in _target
                for k in _net_interest_keys
                if re.search(c.get('label', ''), k, re.I)
                or re.search(k, c.get('label', ''), re.I)
            ]
            if _ni_concept_ids:
                # Find the most recent prior period's values
                _periods = sorted(
                    _db[_col_name].distinct(
                        'reporting_period.end_date',
                        {'company_cik': _cik, 'statement_type': 'income_statement'},
                    ),
                    reverse=True,
                )
                if len(_periods) >= 2:
                    _prior_period = _periods[0]  # The most recent saved period
                    # Check if the current filing's period is already stored
                    _current_filing_date = state.get('sec_report_date')
                    if _current_filing_date:
                        try:
                            from datetime import date as _dd
                            _cfd_date = _dd.fromisoformat(_current_filing_date) if isinstance(_current_filing_date, str) else _current_filing_date
                            # If current period is already the most recent, use the next one back
                            if _cfd_date and _periods[0] == _cfd_date:
                                _prior_period = _periods[1] if len(_periods) > 1 else None
                            else:
                                _prior_period = _periods[0]
                        except (ValueError, TypeError):
                            pass

                    if _prior_period:
                        _prior_vals = list(_db[_col_name].find({
                            'company_cik': _cik,
                            'statement_type': 'income_statement',
                            'concept_id': {'$in': _ni_concept_ids},
                            'reporting_period.end_date': _prior_period,
                        }))
                        if _prior_vals:
                            _prior_nii = max(v.get('value', 0) or 0 for v in _prior_vals)
                            for _nk in _net_interest_keys:
                                _current_val = metrics.get(_nk)
                                if isinstance(_current_val, (int, float)) and _current_val and _prior_nii:
                                    _ratio = _current_val / _prior_nii
                                    if _ratio < 0.5 or _ratio > 1.5:
                                        findings.append(
                                            Finding(
                                                type='suspect_value',
                                                severity='high',
                                                message=(
                                                    f"Net Interest Income QoQ swing: {_current_val:,.0f} "
                                                    f"vs prior period {_prior_nii:,.0f} "
                                                    f"({_ratio*100:.0f}%) — likely a column-mixup error. "
                                                    f"Prior NII (period ending {_prior_period}) was "
                                                    f"{_prior_nii:,.0f}."
                                                ),
                                                keys=(_nk,),
                                                evidence={
                                                    'current_value': _current_val,
                                                    'prior_value': _prior_nii,
                                                    'prior_period_end': str(_prior_period),
                                                    'ratio': _ratio,
                                                },
                                                suggested_action=(
                                                    f"Find the Net Interest Income row and read ONLY from the "
                                                    f"current-quarter column. The prior-period value was "
                                                    f"{_prior_nii:,.0f}, so the current value should be close "
                                                    f"to that range."
                                                ),
                                            )
                                        )
                                        logger.warning(
                                            'analyze_metrics %s: NII momentum check — current %s vs prior %s '
                                            '(ratio %.2f) on %s',
                                            ticker,
                                            f'{_current_val:,.0f}',
                                            f'{_prior_nii:,.0f}',
                                            _ratio,
                                            _nk,
                                        )
        except Exception as _exc:
            logger.debug('analyze_metrics %s: NII momentum check failed — %s', ticker, _exc)

    # Corrector post-pass — kept explicitly separate from the observer loop so
    # it is easy to audit: only derive_corrected_total_opex mutates metrics.
    opex_collision = any(
        f.type == "suspect_value"
        and any("total operating expenses" in k.lower() for k in (f.keys or ()))
        for f in findings
    )
    if opex_collision:
        corrected_key, corrected_value = derive_corrected_total_opex(metrics)
        if corrected_key is not None and corrected_value is not None:
            old_value = metrics[corrected_key]
            metrics = {**metrics, corrected_key: corrected_value}
            findings.append(
                Finding(
                    type="auto_corrected",
                    severity="low",
                    message=(
                        f"Auto-corrected '{corrected_key}' from {old_value:,.0f} "
                        f"to {corrected_value:,.0f} "
                        f"(Cost of revenue + Operating expenses)."
                    ),
                    keys=(corrected_key,),
                    evidence={
                        "old_value": old_value,
                        "corrected_value": corrected_value,
                        "method": "Cost_of_revenue + Operating_expenses",
                    },
                )
            )
            logger.info(
                "analyze_metrics %s: auto-corrected '%s' %s → %s",
                ticker, corrected_key, old_value, corrected_value,
            )

    # ── Industry-aware TIER adjustments ──────────────────────────────────────
    # Promote/demote concept criticality based on the company's industry.
    # e.g. banks need Interest Income as TIER1; insurance skips Gross Profit.
    _ind = state.get("company_industry") or {}
    _cat = _ind.get("category", "general") if isinstance(_ind, dict) else "general"
    if _cat in ("banking", "insurance"):
        from dataclasses import replace as _dcr

        # Banks + insurance: CoGS/Gross Profit not applicable → demote
        findings = [
            _dcr(f, severity="medium")
            if f.severity == "high" and f.type == "missing_critical"
            and any(k and "gross" in k.lower()
                    for k in (f.keys or ()))
            else f
            for f in findings
        ]
        # Banks: promote interest income/expense from tier2 → tier1
        findings = [
            _dcr(f, severity="high")
            if f.severity == "medium" and f.type == "missing_expected"
            and any(k and ("interest income" in k.lower() or "interest expense" in k.lower())
                    for k in (f.keys or ()))
            else f
            for f in findings
        ]

    # ── Identity-rule corrections ─────────────────────────────────────────
    # After all observers have run, correct any values that violate basic
    # accounting identities (e.g. GP ≠ Revenue − CoR).  The LLM may have
    # extracted a hallucinated or misread value; the identity is unambiguous.
    findings = _apply_identity_corrections(findings, metrics, state)

    # Log a compact summary.
    by_type: dict[str, int] = {}
    for f in findings:
        by_type[f.type] = by_type.get(f.type, 0) + 1
    logger.info(
        "analyze_metrics %s: tier1_missing=%d tier2_missing=%d tier3_present=%d findings=%s",
        ticker,
        len(presence["tier1_missing"]),
        len(presence["tier2_missing"]),
        len(presence["tier3_present"]),
        by_type or "{}",
    )

    out: dict[str, Any] = {
        **state,
        "metrics": metrics,
        "findings": [f.to_dict() for f in findings],
        "needs_reextract": False,
    }

    # In targeted mode (normalize_data) the truth set is ``target_concepts``,
    # not the hardcoded TIER1 registry.  Looping based on TIER1 misses is
    # futile when the company simply doesn't report that metric (e.g. BJ
    # Wholesale Club never reports a standalone Gross Profit line) AND when
    # the targeted prompt was only asked about a subset of statements.
    # Demote ``missing_critical`` findings to medium severity so they no
    # longer trigger re-extract; keep them in ``findings`` for visibility.
    if state.get("target_concepts"):
        from dataclasses import replace as _dc_replace
        findings = [
            _dc_replace(f, severity="medium")
            if f.severity == "high" and f.type == "missing_critical"
            else f
            for f in findings
        ]
        out["findings"] = [f.to_dict() for f in findings]

    # Skill-effectiveness tracking (observability, ADR-0006). On every pass
    # after the first we have the previous pass's findings, so we can record
    # which skills' findings were resolved/persisted/new. This is pure
    # observation: it is logged and accumulated on state for later inspection,
    # and never influences routing.
    attempts = state.get("extraction_attempts", 0)
    if prev_findings:
        deltas = compute_skill_effectiveness(prev_findings, out["findings"])
        if deltas:
            effectiveness_log = list(state.get("skill_effectiveness") or [])
            effectiveness_log.append({"to_attempt": attempts, "deltas": deltas})
            out["skill_effectiveness"] = effectiveness_log
            resolved_by_type = {
                d["finding_type"]: d["resolved"] for d in deltas if d["resolved"]
            }
            logger.info(
                "analyze_metrics %s: skill effectiveness — resolved=%s deltas=%s",
                ticker,
                resolved_by_type or "{}",
                deltas,
            )

    high = [f for f in findings if f.severity == "high"]

    if not high or attempts >= MAX_EXTRACTION_ATTEMPTS:
        if high:
            n_msg = min(len(high), 3)
            samples = "; ".join(f.message[:60] for f in high[:n_msg])
            report_call(
                f"  [analyze]  {len(high)} high finding(s) unresolvable after "
                f"{attempts}/{MAX_EXTRACTION_ATTEMPTS} attempts — proceeding to save"
            )
            logger.warning(
                "analyze_metrics %s: %d critical metric(s) still missing after "
                "%d attempt(s) — proceeding to cleanup/save",
                ticker, len(high), attempts,
            )
        else:
            report_call(f"  [analyze]  ✓ all checks passed")
        return out  # type: ignore[return-value]

    # No-progress detection: if the set of high-severity messages is identical
    # to the previous pass, the LLM gained nothing — break the loop early.
    current_high_keys = sorted(f.message for f in high)
    prev_high_keys: list[str] = state.get("previous_high_finding_keys") or []
    if attempts > 0 and current_high_keys == prev_high_keys:
        report_call(
            f"  [analyze]  same {len(high)} high finding(s) as previous pass — "
            f"no progress, skipping re-extract"
        )
        logger.warning(
            "analyze_metrics %s: same %d high finding(s) as previous pass — "
            "no progress detected, skipping re-extract",
            ticker, len(high),
        )
        return out  # type: ignore[return-value]

    # Loop back: build cumulative extraction notes.
    prev_notes = state.get("extraction_notes") or ""
    out["extraction_notes"] = _build_extraction_notes(
        findings,
        attempts + 1,
        prev_notes,
        missing_toplevel=state.get("missing_toplevel_labels") or [],
        missing_segments=state.get("missing_segment_labels") or [],
    )
    out["needs_reextract"] = True
    out["previous_high_finding_keys"] = current_high_keys
    msg = f"  [analyze]  {len(high)} high finding(s) — re-extracting (attempt {attempts + 1}/{MAX_EXTRACTION_ATTEMPTS})"
    report_call(msg)
    logger.info(
        "analyze_metrics %s: %d critical metric(s) missing — looping back "
        "(attempt %d/%d)",
        ticker, len(high), attempts + 1, MAX_EXTRACTION_ATTEMPTS,
    )
    return out  # type: ignore[return-value]

