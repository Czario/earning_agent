"""Deterministic currency detection and USD-only guardrails.

The extraction pipeline ingests many document sources — SEC 8-K HTML press
releases, EDGAR exhibits, IR-hosted PDFs, shareholder letters, and other
website PDF URLs.  Regardless of source, consolidated monetary figures are
normalized to USD before persistence.  When a document or table explicitly
declares another currency, the pipeline must not silently treat those numbers
as USD.  This module provides deterministic, source-agnostic currency detection
plus a USD-only validation gate.

It deliberately performs **no foreign-exchange conversion**: the extraction
agent must never invent an FX rate.  Foreign-currency figures are either taken
from an explicitly company-reported USD translation (which the agent returns
directly as a USD value) or marked unresolved and blocked at the save gate.
"""
from __future__ import annotations

import re
from typing import Any

# (pattern, iso_code, human name).  Word-bounded so "CAD" does not match
# "arcade".  Multi-word forms precede their single-token abbreviations.
_CURRENCY_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b(?:U\.?S\.?\s+dollars?|US\s*dollars?|United States dollars?|USD)\b", re.I), "USD", "US dollars"),
    (re.compile(r"\b(?:euros?|EUR)\b", re.I), "EUR", "euros"),
    (re.compile(r"\b(?:pounds?\s+sterling|British pounds?|GBP)\b", re.I), "GBP", "pounds sterling"),
    (re.compile(r"\b(?:Canadian dollars?|CAD)\b", re.I), "CAD", "Canadian dollars"),
    (re.compile(r"\b(?:Australian dollars?|AUD)\b", re.I), "AUD", "Australian dollars"),
    (re.compile(r"\b(?:Swiss francs?|CHF)\b", re.I), "CHF", "Swiss francs"),
    (re.compile(r"\b(?:Japanese yen|JPY)\b", re.I), "JPY", "Japanese yen"),
    (re.compile(r"\b(?:Indian rupees?|INR)\b", re.I), "INR", "Indian rupees"),
    (re.compile(r"\b(?:Chinese yuan|renminbi|RMB|CNY)\b", re.I), "CNY", "Chinese yuan"),
]

_SYMBOL_AMOUNT_RX: dict[str, re.Pattern[str]] = {
    "USD": re.compile(r"\$\s*\d"),
    "EUR": re.compile(r"€\s*\d"),
    "GBP": re.compile(r"£\s*\d"),
    "JPY": re.compile(r"¥\s*\d"),
}


def detect_currency(text: str) -> dict[str, Any]:
    """Detect explicitly declared currency(ies) in *text*.

    Returns ``{"currency", "confidence", "detected_codes", "evidence",
    "requires_review"}``.

    * ``currency`` is a single ISO code when exactly one currency is detected,
      ``"USD"`` when USD is present, ``"mixed"`` when multiple foreign codes
      are present, or ``"unknown"`` when nothing is declared.
    * ``confidence`` is ``"high"`` for an explicit declaration and ``"low"``
      when only a ``$`` amount symbol is present.
    * ``requires_review`` is True when the result cannot safely be persisted
      as USD (no declaration, foreign-only, or mixed foreign).
    """
    explicit_codes: dict[str, str] = {}  # code -> first evidence snippet
    for pattern, code, _name in _CURRENCY_PATTERNS:
        for m in pattern.finditer(text):
            explicit_codes.setdefault(code, m.group(0))
            break  # first match per code is enough evidence

    symbol_codes: set[str] = set()
    for code, pattern in _SYMBOL_AMOUNT_RX.items():
        if pattern.search(text):
            symbol_codes.add(code)

    detected_codes = sorted(set(explicit_codes) | symbol_codes)
    evidence = [f"{code}: {snippet}" for code, snippet in explicit_codes.items()]

    if "USD" in explicit_codes:
        currency = "USD"
        confidence = "high"
        requires_review = len(explicit_codes) > 1
    elif len(explicit_codes) == 1:
        currency = next(iter(explicit_codes))
        confidence = "high"
        requires_review = True  # foreign-only: not USD, must be blocked/resolved
    elif len(explicit_codes) > 1:
        currency = "mixed"
        confidence = "high"
        requires_review = True
    elif "USD" in symbol_codes:
        currency = "USD"
        confidence = "low"  # "$" alone is not proof of USD
        requires_review = False
    elif symbol_codes:
        currency = next(iter(symbol_codes))
        confidence = "low"
        requires_review = True
    else:
        currency = "unknown"
        confidence = "low"
        requires_review = True

    return {
        "currency": currency,
        "confidence": confidence,
        "detected_codes": detected_codes,
        "evidence": evidence,
        "requires_review": requires_review,
    }


def usd_metadata(text: str) -> dict[str, Any]:
    """Return the canonical currency metadata record for state.

    Thin wrapper around :func:`detect_currency` that names fields the pipeline
    stores on ``state["currency_metadata"]``.
    """
    detected = detect_currency(text)
    return {
        "currency": detected["currency"],
        "confidence": detected["confidence"],
        "detected_codes": detected["detected_codes"],
        "evidence": detected["evidence"],
        "requires_review": detected["requires_review"],
    }


def is_usd_safe(currency: str | None) -> bool:
    """Return True when *currency* can be persisted as USD.

    Only an explicit or symbol-implied ``USD`` is safe.  ``unknown``,
    foreign codes, and ``mixed`` are never safe without resolution.
    """
    return currency == "USD"


def validate_currency_metadata(metadata: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Validate a currency metadata record against the USD-only invariant.

    Returns ``(ok, error_message)``.  ``error_message`` is None when the record
    is safe to persist as USD.
    """
    if not metadata:
        return False, "missing currency metadata"
    currency = metadata.get("currency")
    if currency == "USD":
        return True, None
    if currency in (None, "unknown"):
        return False, "currency could not be determined — refusing to assume USD"
    if currency == "mixed":
        return False, "multiple currencies detected — refusing to assume USD"
    return False, f"non-USD currency {currency!r} — refusing to save as USD without conversion"
