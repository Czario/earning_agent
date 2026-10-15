"""Currency detection and the USD-preferred / native-currency policy.

The extraction pipeline ingests many document sources — SEC 8-K/6-K HTML press
releases, EDGAR exhibits, IR-hosted PDFs, shareholder letters, and other
website PDF URLs.  Some issuers report in USD; foreign issuers (ASML, TSM, ...)
often report only in their own currency (EUR, TWD, JPY, ...).

Policy — **USD preferred, native currency fallback, never invented**:

* Prefer the company-reported **USD** presentation when the filing provides one
  (a USD column/table/translation).
* Otherwise persist the filing's explicitly reported currency.
* **Never** perform a foreign-exchange conversion and **never** assume USD: a
  monetary value whose currency cannot be confirmed is not persisted.
* Use **one currency per statement** so a statement's values stay comparable;
  a statement is never stored as a mix of currencies.

This module provides deterministic, source-agnostic currency detection plus the
statement-level currency resolution used by the extraction pipeline, the save
gate, Q4 derivation, and guidance persistence.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# (pattern, iso_code, human name).  Word-bounded so "CAD" does not match
# "arcade".  Multi-word forms precede their single-token abbreviations.
_CURRENCY_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b(?:U\.?S\.?\s+dollars?|US\s*dollars?|United States dollars?|USD)\b", re.I), "USD", "US dollars"),
    (re.compile(r"\b(?:euros?|EUR)\b", re.I), "EUR", "euros"),
    (re.compile(r"\b(?:pounds?\s+sterling|British pounds?|GBP)\b", re.I), "GBP", "pounds sterling"),
    (re.compile(r"\b(?:Canadian dollars?|CAD)\b", re.I), "CAD", "Canadian dollars"),
    (re.compile(r"\b(?:Australian dollars?|AUD)\b", re.I), "AUD", "Australian dollars"),
    (re.compile(r"\b(?:New Zealand dollars?|NZD)\b", re.I), "NZD", "New Zealand dollars"),
    (re.compile(r"\b(?:Swiss francs?|CHF)\b", re.I), "CHF", "Swiss francs"),
    (re.compile(r"\b(?:Japanese yen|JPY)\b", re.I), "JPY", "Japanese yen"),
    (re.compile(r"\b(?:New Taiwan dollars?|Taiwan dollars?|NT\$|TWD)\b", re.I), "TWD", "New Taiwan dollars"),
    (re.compile(r"\b(?:Hong Kong dollars?|HKD)\b", re.I), "HKD", "Hong Kong dollars"),
    (re.compile(r"\b(?:Singapore dollars?|SGD)\b", re.I), "SGD", "Singapore dollars"),
    (re.compile(r"\b(?:Korean won|KRW)\b", re.I), "KRW", "Korean won"),
    (re.compile(r"\b(?:Indian rupees?|INR)\b", re.I), "INR", "Indian rupees"),
    (re.compile(r"\b(?:Chinese yuan|renminbi|RMB|CNY)\b", re.I), "CNY", "Chinese yuan"),
    (re.compile(r"\b(?:Brazilian reais|Brazilian real|BRL)\b", re.I), "BRL", "Brazilian reais"),
    (re.compile(r"\b(?:Mexican pesos?|MXN)\b", re.I), "MXN", "Mexican pesos"),
    (re.compile(r"\b(?:Swedish kron(?:a|or)|SEK)\b", re.I), "SEK", "Swedish kronor"),
    (re.compile(r"\b(?:Norwegian kron(?:e|er)|NOK)\b", re.I), "NOK", "Norwegian kroner"),
    (re.compile(r"\b(?:Danish kron(?:e|er)|DKK)\b", re.I), "DKK", "Danish kroner"),
    (re.compile(r"\b(?:South African rand|ZAR)\b", re.I), "ZAR", "South African rand"),
    (re.compile(r"\b(?:Polish zlot(?:y|ych)|PLN)\b", re.I), "PLN", "Polish zloty"),
    (re.compile(r"\b(?:Israeli new shekels?|ILS)\b", re.I), "ILS", "Israeli new shekels"),
    (re.compile(r"\b(?:Turkish lira|TRY)\b", re.I), "TRY", "Turkish lira"),
]

_SYMBOL_AMOUNT_RX: dict[str, re.Pattern[str]] = {
    # Negative lookbehind keeps "NT$ 5" (New Taiwan dollars) from also being
    # counted as a bare "$" USD amount.
    "USD": re.compile(r"(?<![A-Za-z])\$\s*\d"),
    "EUR": re.compile(r"€\s*\d"),
    "GBP": re.compile(r"£\s*\d"),
    "JPY": re.compile(r"¥\s*\d"),
    "TWD": re.compile(r"NT\$\s*\d"),
}

# A confirmed currency is a 3-letter ISO 4217 code that actually denotes a
# currency.  The special/non-currency ISO codes (no currency, testing, metals)
# and the pipeline's own sentinels are rejected.
_ISO_CODE_RX = re.compile(r"^[A-Z]{3}$")
_NON_CURRENCY_CODES = frozenset({
    "XXX", "XTS", "XAU", "XAG", "XPT", "XPD", "XBA", "XBB", "XBC", "XBD",
    "UNK", "MIX",
})


def is_currency_code(currency: str | None) -> bool:
    """True when *currency* is a confirmed ISO currency code.

    ``None``, ``""``, ``"unknown"``, ``"mixed"``, and the special non-currency
    ISO codes are never confirmed — so they can never be persisted as USD or as
    a real currency.
    """
    code = str(currency or "").strip().upper()
    return bool(_ISO_CODE_RX.match(code)) and code not in _NON_CURRENCY_CODES


def is_usd_safe(currency: str | None) -> bool:
    """Return True when *currency* is explicitly USD (legacy helper)."""
    return str(currency or "").strip().upper() == "USD"


def detect_currency(text: str) -> dict[str, Any]:
    """Detect explicitly declared currency(ies) in *text*.

    Returns ``{"currency", "confidence", "detected_codes", "evidence",
    "requires_review"}``.

    * ``currency`` is ``"USD"`` when USD is declared (USD is preferred), a
      single foreign ISO code when exactly one foreign currency is declared,
      ``"mixed"`` when several codes are present, or ``"unknown"`` when nothing
      is declared.
    * ``confidence`` is ``"high"`` for an explicit declaration and ``"low"``
      when only an amount symbol (e.g. ``$``) is present.
    * ``requires_review`` is True when the document does not resolve to a
      single confirmed currency.
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
        requires_review = False  # a single explicit native currency is usable
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
    """Return the canonical document-level currency metadata record for state.

    Thin wrapper around :func:`detect_currency` that names the fields the
    pipeline stores on ``state["currency_metadata"]``.  (Name kept for
    compatibility; the value is no longer USD-only.)
    """
    detected = detect_currency(text)
    return {
        "currency": detected["currency"],
        "confidence": detected["confidence"],
        "detected_codes": detected["detected_codes"],
        "evidence": detected["evidence"],
        "requires_review": detected["requires_review"],
    }


# ── Statement-level currency resolution (USD preferred) ──────────────────────

def choose_statement_currency(candidate_codes: Iterable[str | None]) -> tuple[str | None, str]:
    """Choose ONE confirmed currency for a statement's values.

    USD wins when the filing presents USD for that statement; otherwise the
    single reported currency is used.  Returns ``(currency, reason)`` where
    ``currency`` is ``None`` when nothing can be confirmed.
    """
    confirmed = [str(c).strip().upper() for c in candidate_codes if is_currency_code(c)]
    if not confirmed:
        return None, "no confirmed currency for this statement"
    unique = sorted(set(confirmed))
    if "USD" in unique:
        return "USD", "USD presentation preferred"
    if len(unique) == 1:
        return unique[0], "single reported currency"
    return None, f"mixed statement currencies ({', '.join(unique)}) — no conversion performed"


def resolve_statement_currencies(
    per_concept_statement: dict[str, str],
    per_concept_candidate: dict[str, str | None],
) -> dict[str, Any]:
    """Resolve one currency per statement and classify each concept value.

    *per_concept_statement* maps ``concept_id → statement_type`` and
    *per_concept_candidate* maps ``concept_id → candidate currency`` (the
    per-value evidence currency, or the document-level fallback) — both
    already uppercased.

    Returns ``{"chosen_by_statement", "reasons", "currency_by_concept",
    "dropped"}``.  ``currency_by_concept`` contains only values that match the
    statement's chosen currency; ``dropped`` groups the omitted concept ids by
    statement.
    """
    grouped: dict[str, list[tuple[str, str | None]]] = {}
    for cid, stmt in per_concept_statement.items():
        grouped.setdefault((stmt or "income").strip().lower(), []).append(
            (cid, per_concept_candidate.get(cid))
        )

    chosen: dict[str, str | None] = {}
    reasons: dict[str, str] = {}
    for stmt, items in grouped.items():
        code, reason = choose_statement_currency([c for _, c in items])
        chosen[stmt] = code
        reasons[stmt] = reason

    currency_by_concept: dict[str, str] = {}
    dropped: dict[str, list[str]] = {}
    for stmt, items in grouped.items():
        code = chosen[stmt]
        for cid, candidate in items:
            candidate_code = str(candidate or "").strip().upper()
            if code and candidate_code == code:
                currency_by_concept[cid] = code
            else:
                dropped.setdefault(stmt, []).append(cid)

    return {
        "chosen_by_statement": chosen,
        "reasons": reasons,
        "currency_by_concept": currency_by_concept,
        "dropped": dropped,
    }


def document_currency(chosen_by_statement: dict[str, str | None]) -> str:
    """Collapse per-statement currencies into a document-level label."""
    codes = sorted({c for c in (chosen_by_statement or {}).values() if is_currency_code(c)})
    if not codes:
        return "unknown"
    if len(codes) == 1:
        return codes[0]
    return "mixed"


def validate_currency_metadata(metadata: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Validate a document-level currency record.

    Returns ``(ok, error_message)``.  ``ok`` is True when the record names a
    single confirmed currency (USD or a foreign ISO code) — the only states in
    which monetary values may be persisted.
    """
    if not metadata:
        return False, "missing currency metadata"
    currency = metadata.get("currency")
    if is_currency_code(currency):
        return True, None
    if currency in (None, "unknown", ""):
        return False, "currency could not be determined — refusing to assume USD"
    if currency == "mixed":
        return False, "multiple currencies detected — refusing to save unlabeled values"
    return False, f"currency {currency!r} is not a confirmed ISO code"
