"""Concept mapping, prior-value lookup, and document pre-scanning."""

from __future__ import annotations

import json
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


def semantically_map_unmapped_metrics(
    metrics: dict[str, Any],
    target_concepts: list[dict],
    concept_metrics: dict[str, float],
) -> tuple[dict[str, float], set[str]]:
    """Resolve numeric extraction keys that did not map exactly.

    Press releases frequently use a business label that is semantically the
    same as an XBRL-normalized label but not textually identical (for example,
    ``Revenue`` vs ``Total revenue`` or ``Cloud and software`` vs a longer
    normalized expense label).  Asking the model to resolve only these leftover
    *keys* is safer than fuzzy string matching: values are never changed, and
    the model may only choose from the supplied concept IDs.

    This is a repair step, not an extraction step.  Exact taxonomy/label
    mapping always wins, one concept can be selected at most once, and only
    high-confidence mappings returned by the resolver are accepted.  If the
    resolver is unavailable or uncertain, the metric remains observable in the
    existing missing-concept fields rather than being guessed.
    """
    mapped_ids = set(concept_metrics)
    unmapped: dict[str, float] = {
        key: float(value)
        for key, value in metrics.items()
        if not key.startswith("__")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key not in {"__scale__", "__derived__"}
    }
    # Remove keys already handled by exact Tier 0/Tier 1 mapping.  The
    # resolver should see only genuine leftovers, both to reduce prompt size
    # and to prevent it from remapping a canonical extraction key.
    direct_keys: set[str] = set()
    for key, value in unmapped.items():
        for concept in target_concepts:
            cid = concept.get("_id")
            taxonomy_key = concept.get("taxonomy_key") or concept.get("concept") or ""
            label = concept.get("label") or ""
            if (
                key in {taxonomy_key, f"[{taxonomy_key}]", label}
                or re.sub(r"\s+", " ", key).strip().lower()
                == re.sub(r"\s+", " ", label).strip().lower()
            ) and cid in mapped_ids:
                direct_keys.add(key)
                break
    unresolved = {key: value for key, value in unmapped.items() if key not in direct_keys}
    if not unresolved:
        return concept_metrics, set()

    candidates = [
        {
            "concept_id": str(c.get("_id", "")),
            "label": c.get("label", ""),
            "taxonomy_key": c.get("taxonomy_key") or c.get("concept", ""),
            "path": c.get("path", ""),
        }
        for c in target_concepts
        if c.get("_id") not in mapped_ids
        and not str(c.get("concept") or c.get("taxonomy_key") or "").lower().startswith("system:")
        and c.get("_id")
    ]
    if not candidates:
        return concept_metrics, set()

    prompt = """\
You are a conservative accounting concept mapper. Map extracted filing
metric names to the normalized concepts below by MEANING, not by substring or
keyword overlap. Filing labels may be abbreviated, reordered, or use a normal
business synonym; use the surrounding accounting meaning and statement
hierarchy (path) to distinguish similarly named rows.

Do not calculate, alter, rescale, or rename any value. Do not map a metric just
because one word overlaps. If the meaning is not clearly the same, omit it.
Each metric may map to at most one concept, and each concept may be used at most
once. Return only HIGH-confidence mappings.

UNMAPPED EXTRACTED METRICS (key: value):
{metrics}

AVAILABLE TARGET CONCEPTS:
{targets}

Return strict JSON only:
{{"mappings": [{{"metric_key": "exact extracted key", "concept_id":
"exact target concept_id", "confidence": "high"}}]}}
""".format(
        metrics=json.dumps(unresolved, ensure_ascii=False),
        targets=json.dumps(candidates, ensure_ascii=False),
    )

    try:
        from earnings_agents.hooks import report_call
        report_call("  [llm]  semantic concept mapping  → calling llm")
        from earnings_agents.llm import build_llm
        response = build_llm(format_json=True, max_retries=0).invoke(prompt)
        if not isinstance(response, str):
            response = str(response)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", cleaned).strip()
        # JSON mode normally returns the object directly, but tolerate a
        # short explanatory prefix/suffix without accepting arbitrary text.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end >= start:
            cleaned = cleaned[start : end + 1]
        parsed = json.loads(cleaned)
        mappings = parsed.get("mappings", []) if isinstance(parsed, dict) else []
    except Exception as exc:  # noqa: BLE001 — semantic repair is best effort
        logger.warning("semantic concept mapping unavailable: %s", exc)
        return concept_metrics, set()

    if not isinstance(mappings, list):
        return concept_metrics, set()
    candidate_ids = {c["concept_id"] for c in candidates}
    used_ids = set(mapped_ids)
    resolved: set[str] = set()
    updated = dict(concept_metrics)
    for item in mappings:
        if not isinstance(item, dict):
            continue
        key = item.get("metric_key")
        cid = str(item.get("concept_id", ""))
        if (
            not isinstance(key, str)
            or key not in unresolved
            or cid not in candidate_ids
            or cid in used_ids
            or str(item.get("confidence", "")).lower() != "high"
        ):
            continue
        updated[cid] = unresolved[key]
        used_ids.add(cid)
        resolved.add(key)
        logger.info(
            "semantic concept mapping: %r → %s (high confidence)", key, cid
        )
    return updated, resolved


# ── Agent-based post-extraction derivation ───────────────────────────────────
_DERIVATION_PROMPT = """\
You are a financial derivation agent.  Given values extracted from an SEC
earnings filing and a concept hierarchy, compute missing derived values.

EXTRACTED VALUES (verbatim from the filing):
{extracted_block}

CONCEPT HIERARCHY — each parent is computed from its children:
{hierarchy_block}

RULES:
  • Gross Profit = Revenue − Cost of Revenue (the Revenue value is in
    EXTRACTED VALUES; compute Cost of Revenue from its own children in the
    HIERARCHY, then subtract it from that Revenue).
  • Every other parent = sum of its children.
  • If ALL children of a parent are present in the extracted or computed
    values, ALWAYS compute the parent as their sum — do not omit a value
    you can compute exactly.
  • Every parent whose children are ALL present MUST appear in your JSON —
    a missing computable parent is an error, not an omission.
  • When the filing printed ONE combined row for a parent that the HIERARCHY
    splits into children (e.g. "Restructuring and other" = 1,838 splits into
    "Restructuring Charges" + "Acquisition related and other"; or "Costs
    and Expenses" = Cost of Revenue + Operating Expenses), ALLOCATE the
    combined total across the children: a missing child = combined total −
    sum of the extracted children, then compute the parent as the children's
    sum.  Do not leave such a parent uncomputed.
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
    """Build parent-concept-id → list-of-child-concept-ids from the path hierarchy.

    Children are DIRECT only — exactly one path segment deeper.  Matching all
    descendants would double-count nested parents in higher-level sums (e.g.
    Operating Expenses must list "Restructure and Other" as one child, not
    also its grandchildren Restructuring Charges / Acquisition related and
    other, which roll up into Restructure and Other first).
    """
    # Keep every row at a path.  ``order_key`` is part of the row identity;
    # path alone is not unique in normalized XBRL data (geographic members,
    # alternate concepts, and same-path statement rows are common).
    nodes_by_path: dict[str, list[dict]] = {}
    for c in target_concepts:
        p = (c.get("path") or "").strip()
        if p:
            nodes_by_path.setdefault(p, []).append(c)

    def _order_value(c: dict) -> tuple[int, str]:
        value = c.get("order_key")
        return (value is None, "" if value is None else str(value))

    for nodes in nodes_by_path.values():
        nodes.sort(key=_order_value)

    parent_children: dict[str, list[str]] = {}
    for parent_path, parent_nodes in nodes_by_path.items():
        prefix = parent_path + "."
        parent_depth = parent_path.count(".")
        child_ids: list[str] = []
        for child_path, child_nodes in nodes_by_path.items():
            if (
                child_path.startswith(prefix)
                and child_path.count(".") == parent_depth + 1
            ):
                # Preserve every child row, ordered by (path, order_key).
                child_ids.extend(c["_id"] for c in child_nodes)
        if child_ids:
            # A same-path parent row is a distinct (path, order_key) node, so
            # retain the direct child group for each parent rather than
            # arbitrarily attaching it to the first row only.
            for parent in parent_nodes:
                parent_children[parent["_id"]] = list(child_ids)

    return parent_children


def _build_id_label_map(target_concepts: list[dict]) -> dict[str, str]:
    """Build concept_id → label."""
    return {c["_id"]: c.get("label", "?") for c in target_concepts}


def _build_derivation_prompt(
    concept_metrics: dict[str, float],
    target_concepts: list[dict],
) -> str:
    """Build the derivation prompt: extracted values + hierarchy.

    LEAN PROMPT: the extracted block carries only the values the derivation
    actually references — descendants of the missing CALC parents, the
    Revenue operand when Gross Profit is missing, and the combined-costs
    concept for row-allocation.  The hierarchy block already embeds every
    missing parent's child values, so the rest is history-size noise that
    only slows the LLM call (observed live: 33 values → 49.9s derive).
    """
    id_label = _build_id_label_map(target_concepts)
    parent_children = _build_hierarchy(target_concepts)
    present_ids = set(concept_metrics)

    # ── Extracted block — restricted to referenced values ─────────────
    missing_calc_ids = {
        c["_id"] for c in target_concepts
        if _SYSTEM_PREFIX_RX.match(
            (c.get("concept") or c.get("taxonomy_key") or "").strip()
        ) and c["_id"] not in present_ids
    }

    referenced: set[str] = set()
    dependency_uncertain = False
    uncertainty_reasons: list[str] = []
    path_counts: dict[str, int] = {}
    for c in target_concepts:
        p = (c.get("path") or "").strip()
        if p:
            path_counts[p] = path_counts.get(p, 0) + 1

    if missing_calc_ids:
        # (a) All descendants of the missing parents — nested parents and
        # their leaves (e.g. Restructuring Charges under Restructure and
        # Other under Operating Expenses).
        missing_paths = {
            (c.get("path") or "").strip()
            for c in target_concepts if c["_id"] in missing_calc_ids
        }
        if not all(missing_paths):
            dependency_uncertain = True
            uncertainty_reasons.append("missing CALC parent has no path")
        for c in target_concepts:
            p = (c.get("path") or "").strip()
            if any(p.startswith(mp + ".") for mp in missing_paths if mp):
                referenced.add(c["_id"])
                if path_counts.get(p, 0) > 1:
                    dependency_uncertain = True
                    uncertainty_reasons.append(f"duplicate dependency path {p}")
        for mp in missing_paths:
            if mp and path_counts.get(mp, 0) > 1:
                dependency_uncertain = True
                uncertainty_reasons.append(f"duplicate CALC path {mp}")

        # (b) Revenue operand when Gross Profit is missing — GP = Revenue −
        # CoR, and only Revenue's VALUE is pre-known (CoR is computed by the
        # derivation itself, never pre-shown).
        for c in target_concepts:
            if c["_id"] not in missing_calc_ids:
                continue
            ll = (c.get("label") or "").lower()
            if "gross" in ll and "profit" in ll and not (
                "margin" in ll or "ratio" in ll or "%" in ll
            ):
                revenue_candidates = [
                    t for t in target_concepts
                    if "RevenueFromContract" in (
                        t.get("taxonomy_key") or t.get("concept") or ""
                    )
                ]
                if len(revenue_candidates) != 1:
                    dependency_uncertain = True
                    uncertainty_reasons.append("Revenue operand is ambiguous or absent")
                for t in revenue_candidates:
                    referenced.add(t["_id"])

    # (c) Combined costs concept — the row-allocation rule needs its total.
    combined_candidates = [
        c for c in target_concepts
        if (c.get("taxonomy_key") or c.get("concept") or "").strip()
        == "us-gaap:CostsAndExpenses"
        or "costs and expenses" in (c.get("label") or "").lower()
    ]
    if len(combined_candidates) > 1:
        dependency_uncertain = True
        uncertainty_reasons.append("combined-costs operand is ambiguous")
    referenced.update(c["_id"] for c in combined_candidates)

    allowed = referenced & present_ids
    if dependency_uncertain:
        # Full context is a safety fallback only when the dependency graph is
        # ambiguous or incomplete — never merely because the referenced set
        # happens to contain fewer than an arbitrary number of values.
        allowed = present_ids
        logger.debug(
            "derive prompt: dependency uncertainty; keeping full block (%s)",
            "; ".join(dict.fromkeys(uncertainty_reasons)),
        )

    extracted_lines: list[str] = []
    for cid, val in sorted(concept_metrics.items(), key=lambda x: str(x[0])):
        if cid not in allowed:
            continue
        label = id_label.get(cid, cid)
        extracted_lines.append(f"  • {label} = {val:,.0f}")
    extracted_block = "\n".join(extracted_lines) if extracted_lines else "  (none)"
    if len(allowed) < len(present_ids):
        logger.debug(
            "derive prompt: %d of %d extracted values referenced (%s)",
            len(allowed), len(present_ids),
            ", ".join(sorted(id_label.get(i, i) for i in allowed))[:200],
        )

    # ── Hierarchy block ──────────────────────────────────────────────
    # Only show CALC (system:) concepts that are MISSING and have children —
    # parents already present are not the LLM's problem, and a leaner prompt
    # means a faster derivation call (observed live: a bloated prompt riding
    # LangChain's x3 retries took 5m01s).
    hierarchy_lines: list[str] = []
    for c in target_concepts:
        cid = c["_id"]
        concept = (c.get("concept") or c.get("taxonomy_key") or "").strip()
        if not _SYSTEM_PREFIX_RX.match(concept) or cid in present_ids:
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
        # Bounded, self-retried derive call: max_retries=0 disables LangChain's
        # built-in x3 retries (a hung request would otherwise stall for
        # timeout×3 — observed live: 120s × 3 ≈ 5m01s).  Our single retry
        # bounds the worst case to ~2 × timeout while still surviving one
        # transient failure.
        response: str | None = None
        for attempt in (1, 2):
            try:
                llm = build_llm(max_retries=0)
                response = llm.invoke(prompt)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Derivation LLM call failed (attempt %d/2): %s", attempt, exc,
                )
        if response is None:
            return concept_metrics, set()
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
    period: Any,
) -> dict[str, float]:
    """Load prior-period values from normalize_data for agent reference."""
    if not cik or not target_concepts:
        return {}
    try:
        from earnings_agents.integrations.normalize import (
            _get_client,
            _NORMALIZE_DB,
            _values_collection,
        )
        from earnings_agents.agent.period import parse_iso_period_end
        db = _get_client()[_NORMALIZE_DB]
        col_name = _values_collection(period)

        all_periods = sorted(
            db[col_name].distinct(
                "reporting_period.end_date",
                {"cik": cik, "statement_type": "income"},
            ),
            reverse=True,
        )

        if (
            all_periods
            and parse_iso_period_end(all_periods[0]) == period.period_end
        ):
            all_periods = all_periods[1:]

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

        logger.info(
            "Prior values: loaded %d from %s",
            len(result), parse_iso_period_end(prior_end) or prior_end,
        )
        return result
    except Exception:
        logger.debug("Prior values unavailable", exc_info=True)
        return {}
