"""Concept mapping, prior-value lookup, and document pre-scanning."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── Document pre-scan ────────────────────────────────────────────────────────

_PRESCAN_HEADING_PREFIX = (
    r"^[^\S\n]*\$?[^\S\n]*"
    r"(?:(?:u\.?[^\S\n]?s\.?[^\S\n]+)?(?:dollars|amounts|all[^\S\n]+figures|figures)[^\S\n]+)?"
    r"in[^\S\n]+"
)

_PRESCAN_SCALE_PARENS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\([^)]{0,30}?\bin millions\b", re.I), "millions"),
    (re.compile(r"\([^)]{0,30}?\bin thousands\b", re.I), "thousands"),
    (re.compile(r"\([^)]{0,30}?\bin billions\b", re.I), "billions"),
]

_PRESCAN_SCALE_HEADINGS: list[tuple[re.Pattern, str]] = [
    (re.compile(_PRESCAN_HEADING_PREFIX + r"millions\b", re.I | re.M), "millions"),
    (re.compile(_PRESCAN_HEADING_PREFIX + r"thousands\b", re.I | re.M), "thousands"),
    (re.compile(_PRESCAN_HEADING_PREFIX + r"billions\b", re.I | re.M), "billions"),
]

_PRESCAN_SHARES_IN_THOUSANDS_RX = re.compile(
    r"shares\s+(?:which\s+are\s+)?(?:reflected\s+)?in\s+thousands"
    r"|number\s+of\s+shares[^)]{0,60}in\s+thousands"
    r"|except[^)]{0,60}shares[^)]{0,60}thousands",
    re.I,
)

def prescan_document(raw_text: str) -> tuple[str | None, str | None]:
    """Scan the full document once for scale (thousands/millions/billions).

    Returns ``(scale, shares_scale)`` — either may be None if not detected.
    Period detection is NOT done here — the period agent reads the document
    header itself.
    """
    text = re.sub(r"[^\S\n]+", " ", raw_text)

    def _dominant_scale(patterns: list[tuple[re.Pattern, str]]) -> str | None:
        counts: dict[str, int] = {}
        first_pos: dict[str, int] = {}
        for pattern, scale_name in patterns:
            for m in pattern.finditer(text):
                counts[scale_name] = counts.get(scale_name, 0) + 1
                if scale_name not in first_pos or m.start() < first_pos[scale_name]:
                    first_pos[scale_name] = m.start()
        if not counts:
            return None
        max_count = max(counts.values())
        candidates = [s for s, c in counts.items() if c == max_count]
        return min(candidates, key=lambda s: first_pos[s])

    scale: str | None = _dominant_scale(_PRESCAN_SCALE_PARENS) or _dominant_scale(
        _PRESCAN_SCALE_HEADINGS
    )

    shares_scale: str | None = None
    if _PRESCAN_SHARES_IN_THOUSANDS_RX.search(text):
        shares_scale = "thousands"

    return scale, shares_scale


# ── Scale multipliers ────────────────────────────────────────────────────────

SCALE_MULTIPLIERS: dict[str, int] = {
    "millions": 1_000_000,
    "thousands": 1_000,
    "billions": 1_000_000_000,
}


# ── Tier 0 + Tier 1 concept mapping ─────────────────────────────────────────

def map_concepts(
    metrics: dict[str, Any],
    target_concepts: list[dict],
) -> tuple[dict[str, float], dict[str, str], set[str]]:
    """Map extracted metric keys to concept_ids via Tier 0 (bracket/taxonomy key)
    and Tier 1 (deterministic label match).

    Returns:
        concept_metrics:  concept_id → float
        reverse_map:      concept_id → metric_key (for mapped_metric_keys)
        mapped_keys:      set of metric keys that were successfully mapped
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    taxonomy_key_to_id: dict[str, str] = {}
    bracket_key_to_id: dict[str, str] = {}
    exact_label_to_id: dict[str, str] = {}
    norm_label_to_id: dict[str, str] = {}

    for c in target_concepts:
        cid = c["_id"]
        exact_label_to_id[c["label"]] = cid
        norm_label_to_id[_norm(c["label"])] = cid
        key = c.get("taxonomy_key") or c.get("concept") or ""
        if key:
            taxonomy_key_to_id[key] = cid
            bracket_key_to_id[f"[{key}]"] = cid

    concept_metrics: dict[str, float] = {}
    reverse_map: dict[str, str] = {}
    mapped_keys: set[str] = set()

    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if key in taxonomy_key_to_id:
            cid = taxonomy_key_to_id[key]
        elif key in bracket_key_to_id:
            cid = bracket_key_to_id[key]
        elif key in exact_label_to_id:
            cid = exact_label_to_id[key]
        elif _norm(key) in norm_label_to_id:
            cid = norm_label_to_id[_norm(key)]
        else:
            continue
        concept_metrics[cid] = float(value)
        reverse_map[cid] = key
        mapped_keys.add(key)

    return concept_metrics, reverse_map, mapped_keys


# ── Agent-based post-extraction derivation ───────────────────────────────────

_DERIVATION_PROMPT = """\
You are a financial derivation agent.  Given values extracted from an SEC
earnings filing and a concept hierarchy, compute missing derived values.

EXTRACTED VALUES (verbatim from the filing):
{extracted_block}

CONCEPT HIERARCHY — each parent is computed from its children:
{hierarchy_block}

RULES:
  • Gross Profit = Revenue − Cost of Revenue (if GP is missing but both
    Revenue and Cost of Revenue are present in the extracted or computable
    values).
  • Every other parent = sum of its children.
  • If a combined "Costs and Expenses" line was extracted, it covers BOTH
    Cost of Revenue AND Operating Expenses.  Split it using the hierarchy:
    the children under each parent tell you how to allocate the total.
  • NEVER compute a value that is already in EXTRACTED VALUES.
  • Only return values for the parent concepts listed in the HIERARCHY.
  • If you can't compute a value (missing children, etc.), OMIT it.
  • Use the EXACT amounts from the extracted values.  Ignore the
    "Sum if all available" hint — it is informational only.

Return ONLY a JSON object mapping concept_id to computed value:
  {{"concept_id_1": 123456, "concept_id_2": 789012}}
"""

_SYSTEM_PREFIX_RX = re.compile(r"^system:", re.I)


def _build_hierarchy(
    target_concepts: list[dict],
) -> dict[str, list[str]]:
    """Build parent-concept-id → list-of-child-concept-ids from the path hierarchy."""
    path_to_id: dict[str, str] = {}
    for c in target_concepts:
        p = (c.get("path") or "").strip()
        # First concept at a path wins — sibling rows can share a path
        # (disambiguated by order_key), so last-write would be arbitrary.
        if p and p not in path_to_id:
            path_to_id[p] = c["_id"]

    parent_children: dict[str, list[str]] = {}
    for parent_path, parent_id in path_to_id.items():
        prefix = parent_path + "."
        children: list[str] = []
        for child_path, child_id in path_to_id.items():
            if child_path.startswith(prefix):
                children.append(child_id)
        if children:
            parent_children[parent_id] = children

    return parent_children


def _build_id_label_map(target_concepts: list[dict]) -> dict[str, str]:
    """Build concept_id → label."""
    return {c["_id"]: c.get("label", "?") for c in target_concepts}


def _build_derivation_prompt(
    concept_metrics: dict[str, float],
    target_concepts: list[dict],
) -> str:
    """Build the derivation prompt: extracted values + hierarchy."""
    id_label = _build_id_label_map(target_concepts)
    parent_children = _build_hierarchy(target_concepts)

    # ── Extracted block ──────────────────────────────────────────────
    extracted_lines: list[str] = []
    for cid, val in sorted(concept_metrics.items(), key=lambda x: str(x[0])):
        label = id_label.get(cid, cid)
        extracted_lines.append(f"  • {label} = {val:,.0f}")
    extracted_block = "\n".join(extracted_lines) if extracted_lines else "  (none)"

    # ── Hierarchy block ──────────────────────────────────────────────
    # Only show CALC (system:) concepts that have children.
    hierarchy_lines: list[str] = []
    for c in target_concepts:
        cid = c["_id"]
        concept = (c.get("concept") or c.get("taxonomy_key") or "").strip()
        if not _SYSTEM_PREFIX_RX.match(concept):
            continue
        label = c.get("label", "?")
        child_ids = parent_children.get(cid, [])

        # Determine if GP (special formula) or regular parent.
        # Margin/ratio concepts are NEVER the dollar Gross Profit subtotal —
        # computing Rev − CoR for "Gross Profit Margin" would store a dollar
        # value in a percentage concept (observed live: 16,800,000 stored for
        # system:GrossProfitMargin).  Margins stay underived until the proper
        # calculated_row_formulas wiring.
        label_lower = label.lower()
        is_margin_or_ratio = (
            "margin" in label_lower or "ratio" in label_lower or "%" in label_lower
        )
        if "gross" in label_lower and "profit" in label_lower and not is_margin_or_ratio:
            hierarchy_lines.append(
                f"  {cid} — \"{label}\"  ← Gross Profit = Revenue − Cost of Revenue"
            )
            continue

        if not child_ids:
            continue

        hierarchy_lines.append(f"  {cid} — \"{label}\"  ← sum of:")
        child_sum = 0.0
        all_available = True
        for child_id in child_ids:
            child_label = id_label.get(child_id, child_id)
            if child_id in concept_metrics:
                child_val = concept_metrics[child_id]
                child_sum += child_val
                hierarchy_lines.append(
                    f"      ✓ {child_label} = {child_val:,.0f}"
                )
            else:
                hierarchy_lines.append(f"      ✗ {child_label} (not extracted)")
                all_available = False
        if all_available:
            hierarchy_lines.append(f"      → Sum = {child_sum:,.0f}")

    hierarchy_block = "\n".join(hierarchy_lines) if hierarchy_lines else "  (no derived concepts)"

    return _DERIVATION_PROMPT.format(
        extracted_block=extracted_block,
        hierarchy_block=hierarchy_block,
    )


def derive_missing_concepts(
    concept_metrics: dict[str, float],
    target_concepts: list[dict],
) -> tuple[dict[str, float], set[str]]:
    """Compute missing CALC concepts via a lightweight LLM call.

    The agent receives extracted values + the concept hierarchy (which
    children roll up to which parent) and returns JSON with computed
    values.  This is more flexible than deterministic heuristics — the
    LLM can reason about combined-costs splits, different naming
    conventions, and partial-child scenarios.

    Returns ``(updated_concept_metrics, derived_concept_ids)``.
    """
    # ── Quick check: is there any work to do? ────────────────────────
    parent_children = _build_hierarchy(target_concepts)
    calc_ids: set[str] = set()
    for c in target_concepts:
        concept = (c.get("concept") or c.get("taxonomy_key") or "").strip()
        if _SYSTEM_PREFIX_RX.match(concept):
            calc_ids.add(c["_id"])

    missing_calc = [cid for cid in calc_ids if cid not in concept_metrics]
    # Also check for combined costs (non-system concepts that contain
    # both CoR and OpEx).
    has_combined = any(
        "costsandexpenses" in (c.get("concept") or "").lower()
        or "costs and expenses" in (c.get("label") or "").lower()
        for c in target_concepts
        if c["_id"] in concept_metrics
    )

    if not missing_calc and not has_combined:
        # Nothing to derive — all CALC concepts already present.
        return concept_metrics, set()

    # ── Build prompt and call LLM ────────────────────────────────────
    prompt = _build_derivation_prompt(concept_metrics, target_concepts)

    try:
        from earnings_agents.hooks import report_call as _report_call
        from earnings_agents.llm import build_llm
        from earnings_agents.config import LLM_PROVIDER as _LP
        labels = _build_id_label_map(target_concepts)
        names = ", ".join(labels.get(cid, cid) for cid in missing_calc[:6])
        if names:
            extra = (
                f" (+{len(missing_calc) - 6} more)" if len(missing_calc) > 6 else ""
            )
            derive_msg = f"derive missing CALC — {names}{extra}"
        else:
            derive_msg = "derive (combined-costs split only)"
        _report_call(
            f"  [llm]  {derive_msg}  → calling llm  ({_LP or 'llm'})"
        )
        llm = build_llm()
        response = llm.invoke(prompt)
    except Exception as exc:
        logger.warning("Derivation LLM call failed: %s", exc)
        return concept_metrics, set()

    # ── Parse JSON response ──────────────────────────────────────────
    cleaned = (
        response.strip()
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
        import json
        computed: dict[str, float] = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Derivation JSON parse failed: %s", exc)
        return concept_metrics, set()

    # ── Merge computed values (never override extracted) ─────────────
    derived: set[str] = set()
    metrics = dict(concept_metrics)
    for cid, val in computed.items():
        if cid in metrics:
            logger.debug("derive: skipping %s — already extracted", cid)
            continue
        if not isinstance(val, (int, float)):
            continue
        if cid not in calc_ids:
            logger.debug("derive: skipping %s — not a known CALC concept", cid)
            continue
        metrics[cid] = float(val)
        derived.add(cid)
        label = _build_id_label_map(target_concepts).get(cid, cid)
        logger.info("derive: %s = %.0f (agent-computed)", label, val)

    # ── Report what was computed / omitted, with concept names ───────
    from earnings_agents.hooks import report_call
    labels = _build_id_label_map(target_concepts)
    for cid in sorted(derived):
        report_call(
            f"  [derived]  ✓ {labels.get(cid, cid)} = {metrics[cid]:,.0f}"
        )
    for cid in missing_calc:
        if cid not in derived:
            report_call(
                f"  [derived]  ✗ {labels.get(cid, cid)} — not computable (no data)"
            )

    return metrics, derived


# ── Prior-value loader ───────────────────────────────────────────────────────

def load_prior_values(
    target_concepts: list[dict],
    cik: str | None,
    detected_period_type: str | None,
    sec_report_date: str | None,
) -> dict[str, float]:
    """Load prior-period values from normalize_data for agent reference."""
    if not cik or not target_concepts:
        return {}
    try:
        from earnings_agents.integrations.normalize import _get_client, _NORMALIZE_DB
        from datetime import date as _date

        db = _get_client()[_NORMALIZE_DB]
        period_type = detected_period_type or "quarterly"
        col_name = (
            "concept_values_quarterly" if period_type == "quarterly"
            else "concept_values_annual"
        )

        all_periods = sorted(
            db[col_name].distinct(
                "reporting_period.end_date",
                {"cik": cik, "statement_type": "income"},
            ),
            reverse=True,
        )

        if sec_report_date:
            try:
                cfd = _date.fromisoformat(sec_report_date) if isinstance(sec_report_date, str) else sec_report_date
                if all_periods and all_periods[0] == cfd:
                    all_periods = all_periods[1:]
            except (ValueError, TypeError):
                pass

        prior_end = all_periods[0] if all_periods else None
        if not prior_end:
            return {}

        concept_ids = [c["_id"] for c in target_concepts if c.get("_id")]
        if not concept_ids:
            return {}

        prior_vals = list(db[col_name].find({
            "cik": cik,
            "statement_type": "income",
            "concept_id": {"$in": concept_ids},
            "reporting_period.end_date": prior_end,
        }))

        id_to_label = {str(c["_id"]): c.get("label", "") for c in target_concepts}
        result: dict[str, float] = {}
        for pv in prior_vals:
            cid = str(pv["concept_id"])
            val = pv.get("value")
            if isinstance(val, (int, float)) and cid in id_to_label:
                label = id_to_label[cid]
                if label:
                    result[label] = float(val)

        logger.info("Prior values: loaded %d from %s", len(result), prior_end.date())
        return result
    except Exception:
        logger.debug("Prior values unavailable", exc_info=True)
        return {}
