"""Deterministic scale detection and multiplier guardrails.

The extraction pipeline ingests many document sources — SEC 8-K HTML press
releases, EDGAR exhibits, IR-hosted PDFs, shareholder letters, and other
website PDF URLs.  Each table may declare its own unit (``in thousands`` /
``in millions`` / ``in billions``), and exhibits in one bundle may differ.
This module provides deterministic, source-agnostic scale detection used by
the agent's ``detect_scale`` tool and as a safety fallback for the parser.

The *decision* of which scale applies to a value is the extraction agent's
(per table/section); this module only detects and validates the declaration.
"""
from __future__ import annotations

import re
from typing import Any

_SCALE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bin\s+millions?\b", re.I), "millions"),
    (re.compile(r"\bin\s+thousands?\b", re.I), "thousands"),
    (re.compile(r"\bin\s+billions?\b", re.I), "billions"),
]

_SCALE_MULTIPLIERS: dict[str, int] = {
    "millions": 1_000_000,
    "thousands": 1_000,
    "billions": 1_000_000_000,
}

# Declarations that are NOT a scale statement but contain "in ..." wording
# that would otherwise match (e.g. "in millions" inside a sentence about a
# prior-year comparison).  The detector stays conservative: it only reports
# a scale when a declaration appears inside parentheses or as a standalone
# heading line (the same shapes the legacy prescan accepted).
_HEADING_PREFIX = re.compile(
    r"^[^\S\n]*\$?[^\S\n]*"
    r"(?:(?:u\.?[^\S\n]?s\.?[^\S\n]+)?(?:dollars|amounts|all[^\S\n]+figures|figures)[^\S\n]+)?"
    r"in[^\S\n]+",
    re.I | re.M,
)


def detect_scale(text: str) -> dict[str, Any]:
    """Detect explicitly declared scale(s) in *text*.

    Returns ``{"scale", "confidence", "detected_scales", "evidence",
    "requires_review"}``:

    * ``scale`` is a single unit when exactly one is declared, ``"mixed"``
      when several are present, or ``None`` when nothing is declared.
    * ``confidence`` is ``"high"`` for an explicit declaration and
      ``"low"`` when only a partial/ambiguous hint was found.
    * ``requires_review`` is True when the result cannot be applied blindly
      (mixed declarations, or a dollar heading without an explicit unit).
    """
    # Standalone headings: "In millions", "(in millions)", "Dollars in
    # millions", "Amounts in thousands (except per share data)".
    heading_hits: list[tuple[int, str]] = []
    for pattern, scale_name in _SCALE_PATTERNS:
        for m in pattern.finditer(text):
            # A heading-line match must be on a line that reads like a
            # declaration (line starts with the scale phrase, possibly inside
            # parens) — not a mid-sentence mention.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end].strip()
            if _HEADING_PREFIX.search(line) or line.startswith("(") and "in" in line:
                heading_hits.append((m.start(), scale_name))

    # Parenthesized declarations anywhere on a line, e.g. "(in millions)".
    paren_hits: list[tuple[int, str]] = []
    for pattern, scale_name in _SCALE_PATTERNS:
        for m in pattern.finditer(text):
            start = text.rfind("(", 0, m.start())
            end = text.find(")", m.end())
            if start != -1 and end != -1 and end < start + 80:
                paren_hits.append((m.start(), scale_name))

    detected: dict[str, int] = {}
    first_pos: dict[str, int] = {}
    for pos, scale_name in heading_hits + paren_hits:
        detected[scale_name] = detected.get(scale_name, 0) + 1
        first_pos.setdefault(scale_name, pos)

    evidence = [f"{scale_name} (declared at char {pos})" for scale_name, pos in
                sorted(first_pos.items(), key=lambda kv: kv[1])]

    if not detected:
        return {
            "scale": None,
            "confidence": "low",
            "detected_scales": [],
            "evidence": [],
            "requires_review": True,
        }

    max_count = max(detected.values())
    leaders = [s for s, c in detected.items() if c == max_count]
    if len(leaders) > 1:
        return {
            "scale": "mixed",
            "confidence": "high",
            "detected_scales": sorted(detected),
            "evidence": evidence,
            "requires_review": True,
        }

    scale_name = leaders[0]
    return {
        "scale": scale_name,
        "confidence": "high",
        "detected_scales": sorted(detected),
        "evidence": evidence,
        "requires_review": len(detected) > 1,
    }


def scale_multiplier(scale: str | None) -> int:
    """Return the numeric multiplier for *scale* (1 when unknown)."""
    return _SCALE_MULTIPLIERS.get(scale or "", 1)


def validate_scale_metadata(metadata: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Validate a scale metadata record for persistence safety.

    Returns ``(ok, error_message)``.  ``error_message`` is None when the
    record carries a single unambiguous scale.
    """
    if not metadata:
        return False, "missing scale metadata"
    scale = metadata.get("scale")
    if scale in (None, "mixed"):
        return False, f"scale {scale!r} cannot be applied without resolution"
    if scale not in _SCALE_MULTIPLIERS:
        return False, f"unknown scale {scale!r}"
    return True, None
