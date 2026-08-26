"""Concept mapping, prior-value lookup, and document pre-scanning."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _norm_label(s: str) -> str:
    """Whitespace-normalized lowercase form of *s* (shared by mapping code)."""
    return re.sub(r"\s+", " ", s).strip().lower()


def format_value_for_llm(val: float) -> str:
    """Human-readable value that preserves decimals for small as-is values.

    ``f\"{val:,.0f}\"`` destroys per-share / ratio values (0.32 → "0",
    8.94 → "9"); an LLM auditor then flags a CORRECT extraction as a
    mismatch because its prompt disagrees with the printed document
    (observed live: EPS false-positive loop on PDD).  Integral monetary
    values keep thousands separators with no decimals.
    """
    if val == int(val) and abs(val) < 1e15:
        return f"{val:,.0f}"
    return f"{val:,.4f}".rstrip("0").rstrip(".")


# ── As-is (never-scaled) concept labels ───────────────────────────────────────

# Concepts whose values are inherently as-is — per-share amounts, percentages,
# ratios, and share counts.  Applied to concept LABELS (not just metric keys)
# because member-tagged taxonomy keys such as ``custom:Basic|014.001`` carry no
# "per share" signal while their labels ("Basic (Earnings per ordinary share)")
# do — the parser's key-based guardrail alone lets those keys get scaled
# (observed live: EPS 4.85 stored as 4,850,000 under a millions multiplier).
_LABEL_AS_IS_PATTERNS = re.compile(
    r"per\s+(?:ordinary\s+|basic\s+|diluted\s+|common\s+)?(?:share|ads)\b"
    r"|per-share|\beps\b|earnings\s+per"
    r"|%|percent|\bpct\b|basis\s+points|percentage\s+points"
    r"|margin|yield|growth\b|ratio|ratios\b|\brate\b"
    r"|\bshares?\s+(?:outstanding|used|weighted|issued)\b"
    r"|number\s+of\s+shares|weighted.{0,20}average.{0,15}shares"
    r"|employee|headcount"
    r"|production|deliveries\b|delivered"
    r"|(?:super)?charger.{0,12}(?:station|connector)"
    r"|\bstations?\b|\bconnectors?\b"
    r"|\bdays.{0,5}supply\b|\blease count\b"
    r"|\bactive\b.{0,20}\bsubscriptions?\b|\bfsd subscriptions?\b",
    re.IGNORECASE,
)


def build_no_scale_keys(target_concepts: list[dict]) -> set[str]:
    """Return bracket/raw metric keys whose values must NEVER be scaled.

    The parser's key-based per-share/percentage guardrail cannot see concept
    labels, so member-tagged keys whose as-is signal lives in the label
    (``[custom:Basic|014.001]`` → "Basic (Earnings per ordinary share)") escape
    it and get scaled.  This builds the as-is key set from the labels so the
    parser can skip them regardless of the key string.  Both the raw taxonomy
    key and its ``[bracketed]`` form are included (the agent emits the bracket
    form from the concept list).
    """
    out: set[str] = set()
    for c in target_concepts:
        label = (c.get("label") or "").strip()
        if not label or not _LABEL_AS_IS_PATTERNS.search(label):
            continue
        key = (c.get("taxonomy_key") or c.get("concept") or "").strip()
        if key:
            out.add(key)
            out.add(f"[{key}]")
    return out


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

# Income-statement keyword sets used for per-exhibit routing hints.
# A range that hits both a primary and a secondary keyword likely contains
# the income statement.
_IS_PRIMARY_KW = re.compile(
    r"\b(?:revenue|total\s+revenue|net\s+revenue|operating\s+revenue"
    r"|net\s+income|net\s+earnings|net\s+loss|operating\s+income"
    r"|operating\s+loss|income\s+before\s+tax|income\s+from\s+operations"
    r"|gross\s+profit|gross\s+margin|cost\s+of\s+revenue|cost\s+of\s+sales"
    r"|cost\s+of\s+goods\s+sold|interest\s+income|interest\s+expense"
    r"|provision\s+for\s+credit\s+losses|loan\s+loss\s+provision"
    r"|noninterest\s+income|noninterest\s+expense)",
    re.I,
)
_IS_SECONDARY_KW = re.compile(
    r"\b(?:income\s+tax|income\s+taxes|provision\s+for\s+taxes"
    r"|effective\s+tax\s+rate|diluted\s+eps|basic\s+eps"
    r"|earnings\s+per\s+share|weighted.average\s+shares"
    r"|selling.?general|sg%?a|r%?d|research\s+and\s+development"
    r"|depreciation|amortization|stock.based\s+compensation"
    r"|non.operating|other\s+income|other\s+expense"
    r"|consolidated\s+statement|results\s+of\s+operations)",
    re.I,
)

def prescan_document(
    raw_text: str,
    document_map: list[dict] | None = None,
) -> tuple[str | None, str | None, dict | None]:
    """Scan the full document once for scale and (optionally) exhibit-level
    income-statement routing hints.

    Returns ``(scale, shares_scale, is_hints)`` where *is_hints* is
    ``{"exhibit": str, "lines": [start, end], "primary_count": int,
    "secondary_count": int}`` for each exhibit range that appears to contain
    income-statement content.  ``None`` when no document_map is provided
    (backward-compatible with callers that only want scale).

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

    # ── Per-exhibit income-statement routing hints ──────────────────────
    is_hints: list[dict] | None = None
    if document_map:
        lines = raw_text.split("\n")
        for doc in document_map:
            ls = doc.get("line_start")
            le = doc.get("line_end")
            if ls is None or le is None:
                continue
            exhibit_text = "\n".join(lines[ls - 1 : le])
            primary_count = len(_IS_PRIMARY_KW.findall(exhibit_text))
            secondary_count = len(_IS_SECONDARY_KW.findall(exhibit_text))
            if primary_count >= 3 and secondary_count >= 2:
                if is_hints is None:
                    is_hints = []
                is_hints.append({
                    "exhibit": doc.get("exhibit", "unknown"),
                    "lines": [ls, le],
                    "primary_count": primary_count,
                    "secondary_count": secondary_count,
                })
        # Sort by primary keyword density so the best exhibit is first.
        if is_hints:
            is_hints.sort(key=lambda h: h["primary_count"], reverse=True)

    return scale, shares_scale, is_hints


# ── Income-statement section extractor ───────────────────────────────────────

_IS_START_PATTERNS = [
    re.compile(r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+operations", re.I),
    re.compile(r"(?:condensed\s+)?consolidated\s+statement\s+of\s+operations", re.I),
    re.compile(r"statements?\s+of\s+operations\s+and\s+comprehensive", re.I),
    re.compile(r"(?:condensed\s+)?(?:consolidated\s+)?income\s+statement", re.I),
    re.compile(r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+earnings", re.I),
    re.compile(r"statements?\s+of\s+income", re.I),
    re.compile(r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+comprehensive", re.I),
]

_IS_STOP_PATTERNS = [
    re.compile(r"(?:condensed\s+)?consolidated\s+(?:balance\s+sheets?|statements?\s+of\s+financial\s+position)", re.I),
    re.compile(r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+cash\s+flows", re.I),
    re.compile(r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+changes\s+in\s+(?:\w+\s+)*equity", re.I),
    re.compile(r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+stockholders", re.I),
    re.compile(r"(?:condensed\s+)?consolidated\s+statements?\s+of\s+shareholders", re.I),
    re.compile(r"notes\s+to\s+(?:the\s+)?(?:condensed\s+)?(?:consolidated\s+)?financial\s+statements", re.I),
    re.compile(r"(?:unaudited\s+)?notes\s+to\s+financial\s+statements", re.I),
    re.compile(r"independent\s+auditors?['']?\s+report", re.I),
]


def extract_is_section(exhibit_text: str, max_chars: int = 15_000) -> str | None:
    """Extract just the income-statement section from exhibit text.

    Scans for IS header patterns (Statements of Operations, Income
    Statement, etc.) and returns the text from that header to the next
    major section (Balance Sheet, Cash Flow, Notes).  Capped at *max_chars*
    to keep context small for the LLM.

    Returns None if no IS section is found (caller should fall back to
    the full exhibit text).
    """
    lines = exhibit_text.split("\n")
    is_start: int | None = None
    is_stop: int | None = None

    # Find the IS header
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in _IS_START_PATTERNS:
            if pattern.search(stripped):
                is_start = i
                break
        if is_start is not None:
            break

    if is_start is None:
        return None  # no IS header found — caller falls back to full text

    # Find the next major section after the IS header
    for i in range(is_start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        for pattern in _IS_STOP_PATTERNS:
            if pattern.search(stripped):
                is_stop = i
                break
        if is_stop is not None:
            break

    # If no stop pattern found, go 300 lines past the IS start (heuristic)
    if is_stop is None:
        is_stop = min(is_start + 300, len(lines))

    section = "\n".join(lines[is_start:is_stop])
    if len(section) > max_chars:
        section = section[:max_chars] + "\n... (section continues)"
    return section


def build_extraction_summary(
    concept_metrics: dict | None,
    currency: str | None,
    scale: str | None,
    company_name: str,
    period_label: str,
) -> str:
    """Build a compact extraction summary for memory calls.

    Instead of re-processing the full 38K exhibit text when calling
    remember_* tools, the agent gets this small summary (~500 chars)
    with just the key findings.  Drastically reduces context for memory
    LLM steps (30s → ~2s).
    """
    parts = [f"Extraction summary for {company_name} ({period_label}):"]
    if scale:
        parts.append(f"  Scale: {scale}")
    if currency:
        parts.append(f"  Currency: {currency}")
    if concept_metrics:
        sorted_items = sorted(concept_metrics.items(), key=lambda x: str(x[0]))
        parts.append(f"  Extracted {len(sorted_items)} metrics:")
        for cid, val in sorted_items[:15]:
            parts.append(f"    {cid}: {val:,.0f}" if isinstance(val, (int, float)) else f"    {cid}: {val}")
        if len(sorted_items) > 15:
            parts.append(f"    ... and {len(sorted_items) - 15} more")
    return "\n".join(parts)


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
        return _norm_label(s)

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
) -> tuple[dict[str, float], set[str], dict[str, str]]:
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

    Returns ``(concept_metrics, resolved_keys, semantic_reverse)`` where
    ``semantic_reverse`` maps concept_id → the filing metric key it was
    resolved from (used to learn filing-label aliases locally).
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
        return concept_metrics, set(), {}

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
        return concept_metrics, set(), {}

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
        return concept_metrics, set(), {}

    if not isinstance(mappings, list):
        return concept_metrics, set(), {}
    candidate_ids = {c["concept_id"] for c in candidates}
    used_ids = set(mapped_ids)
    resolved: set[str] = set()
    semantic_reverse: dict[str, str] = {}
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
        semantic_reverse[cid] = key
        logger.info(
            "semantic concept mapping: %r → %s (high confidence)", key, cid
        )
    return updated, resolved, semantic_reverse



# ── CALC (system:/calculated) concept derivation ─────────────────────────────

_SYSTEM_PREFIX_RX = re.compile(r"^system:", re.I)


def _build_hierarchy(
    target_concepts: list[dict],
) -> tuple[dict[str, list[str]], set[str]]:
    """Build parent-concept-id → list-of-child-concept-ids from the path hierarchy.

    Children are DIRECT only — exactly one path segment deeper.  Matching all
    descendants would double-count nested parents in higher-level sums (e.g.
    Operating Expenses must list "Restructure and Other" as one child, not
    also its grandchildren Restructuring Charges / Acquisition related and
    other, which roll up into Restructure and Other first).

    Returns ``(parent_children, ambiguous_paths)``.  When several sibling rows
    share the same path (common for dimensional members), the path alone cannot
    attribute direct children to a specific sibling, so the path is recorded as
    ambiguous and no parent under it receives children — deriving such a parent
    would otherwise cross-sum a child into every sibling.
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
    ambiguous_paths: set[str] = set()
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
        if not child_ids:
            continue
        if len(parent_nodes) > 1:
            ambiguous_paths.add(parent_path)
            continue
        parent_children[parent_nodes[0]["_id"]] = list(child_ids)

    return parent_children, ambiguous_paths


def _build_id_label_map(target_concepts: list[dict]) -> dict[str, str]:
    """Build concept_id → label."""
    return {c["_id"]: c.get("label", "?") for c in target_concepts}


def build_calc_derivation_block(
    target_concepts: list[dict],
) -> tuple[str, set[str]]:
    """Render CALC (system:/calculated) concepts as compute-only instructions.

    The extraction agent never reads these rows from the filing (they are
    rollup targets, usually not printed).  The block tells the agent to
    COMPUTE each one with compute() after extracting the verbatim rows, and
    returns ``(block_text, ambiguous_paths)`` — *ambiguous_paths* are
    hierarchy paths with multiple same-path parent rows whose children cannot
    be attributed deterministically (surfaced as observability).

    Returns an empty block string when there are no CALC concepts.
    """
    parent_children, ambiguous_paths = _build_hierarchy(target_concepts)
    id_label = _build_id_label_map(target_concepts)
    lines: list[str] = []
    for c in target_concepts:
        concept = (c.get("concept") or c.get("taxonomy_key") or "").strip()
        if not (_SYSTEM_PREFIX_RX.match(concept) or c.get("calculated")):
            continue
        key = (c.get("taxonomy_key") or concept).strip()
        label = c.get("label") or "?"
        key_str = f"[{key}]" if key else f'"{label}"'
        label_lower = label.lower()
        is_margin_or_ratio = (
            "margin" in label_lower or "ratio" in label_lower or "%" in label_lower
        )
        if "gross" in label_lower and "profit" in label_lower and not is_margin_or_ratio:
            lines.append(
                f'  • {key_str} — "{label}"  ← COMPUTE: Gross Profit = '
                "Revenue − |Cost of Revenue|.  Cost of Revenue is usually "
                "stored NEGATIVE, so NEVER subtract a negative — with Revenue "
                "15,400 and CoR -6,798 → 8,602, NEVER 22,198."
            )
            continue
        child_ids = parent_children.get(c["_id"], [])
        if not child_ids:
            lines.append(
                f'  • {key_str} — "{label}"  ← COMPUTE from the related rows you extracted.'
            )
            continue
        child_labels = ", ".join(
            f'"{id_label.get(cid, cid)}"' for cid in child_ids
        )
        lines.append(
            f'  • {key_str} — "{label}"  ← COMPUTE = sum of: {child_labels} '
            "(using each child's SIGN exactly as extracted)"
        )
    return "\n".join(lines), ambiguous_paths




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
