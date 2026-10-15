"""Validation for agent-proposed, previously unknown filing concepts.

The extraction agent may propose a new concept only after ``map_concept`` has
no match. This module accepts only numeric keys that remain unmapped after the
normal deterministic and semantic mapping passes, have cited line evidence,
and have a valid statement/parent reference. Mongo persistence is deferred to
the save node so a rejected filing cannot create concept rows.
"""
from __future__ import annotations

from typing import Any


def prepare_new_concept_candidates(
    proposals: list[dict] | None,
    *,
    metrics: dict[str, Any],
    target_concepts: list[dict],
    mapped_keys: set[str],
    derived_keys: set[str],
    evidence_by_key: dict[str, Any],
    document_line_count: int,
    target_statements: list[str],
) -> tuple[list[dict], list[str]]:
    """Filter tool proposals down to grounded, still-unmapped value candidates."""
    if not isinstance(proposals, list) or not proposals:
        return [], []

    allowed_statements = {str(s).strip().lower() for s in target_statements}
    mapped = set(mapped_keys)
    derived = {str(k).strip().strip("[]") for k in derived_keys}
    seen_keys: set[str] = set()
    accepted: list[dict] = []
    issues: list[str] = []

    for proposal in proposals[:30]:
        if not isinstance(proposal, dict):
            issues.append("ignored malformed new-concept proposal")
            continue

        metric_key = str(proposal.get("metric_key") or "").strip()
        label = str(proposal.get("label") or "").strip()
        statement_type = str(proposal.get("statement_type") or "").strip().lower()
        if not metric_key or not label:
            issues.append("ignored proposal missing metric_key or printed label")
            continue
        if metric_key in seen_keys:
            continue
        seen_keys.add(metric_key)

        value = metrics.get(metric_key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or metric_key.startswith("__")
        ):
            issues.append(f"{label}: no numeric value under proposed metric key")
            continue
        if metric_key in mapped:
            # Existing concept mapping always wins over concept creation.
            continue
        if metric_key.strip().strip("[]") in derived:
            issues.append(f"{label}: computed values cannot create verbatim concepts")
            continue
        if statement_type not in allowed_statements:
            issues.append(f"{label}: proposal statement is not targeted for this filing")
            continue

        evidence = evidence_by_key.get(metric_key)
        if not isinstance(evidence, dict):
            issues.append(f"{label}: missing per-value evidence")
            continue
        lines = evidence.get("lines")
        if (
            not isinstance(lines, list)
            or len(lines) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in lines)
            or lines[0] < 1
            or lines[1] < lines[0]
            or lines[1] > document_line_count
        ):
            issues.append(f"{label}: invalid or out-of-document evidence line range")
            continue

        parent_key = str(proposal.get("parent_key") or "").strip()
        is_root = parent_key.upper() == "ROOT"
        is_dimension = bool(proposal.get("dimension_concept"))
        parent: dict | None = None
        if is_root:
            if is_dimension:
                issues.append(f"{label}: dimensional member cannot be a statement root")
                continue
        else:
            parent_norm = parent_key.strip("[]").lower()
            parents = [
                concept for concept in target_concepts
                if str(concept.get("statement_type") or "").strip().lower() == statement_type
                and parent_norm in {
                    str(concept.get("taxonomy_key") or "").strip().lower(),
                    str(concept.get("concept") or "").strip().lower(),
                }
                and concept.get("_id")
            ]
            if len(parents) != 1:
                issues.append(f"{label}: parent key is missing or ambiguous")
                continue
            parent = parents[0]
            if not parent.get("path"):
                issues.append(f"{label}: selected parent has no stored hierarchy path")
                continue
            if parent.get("calculated") or str(parent.get("concept") or "").lower().startswith("system:"):
                issues.append(f"{label}: a calculated concept cannot parent a new concept")
                continue

        if is_dimension and not proposal.get("dimension_member_label"):
            issues.append(f"{label}: dimensional member label is required")
            continue

        accepted.append({
            "metric_key": metric_key,
            "value": float(value),
            "label": label,
            "statement_type": statement_type,
            "parent_concept": (parent or {}).get("concept"),
            "parent_concept_id": (parent or {}).get("_id"),
            "parent_path": (parent or {}).get("path"),
            "dimension_concept": is_dimension,
            "dimension_member_label": (
                str(proposal.get("dimension_member_label") or label).strip()
                if is_dimension else ""
            ),
            "dimension_axis": str(proposal.get("dimension_axis") or "").strip(),
            "source_lines": [lines[0], lines[1]],
            "scale": str(evidence.get("scale") or "").strip().lower(),
            "currency": str(evidence.get("currency") or "").strip().upper(),
        })

    if len(proposals) > 30:
        issues.append(f"ignored {len(proposals) - 30} proposal(s) above the per-filing cap")
    return accepted, issues
