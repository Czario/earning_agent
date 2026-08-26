"""Learned memory — label aliases + layout hints, written by the agent via tools.

GOAL: make future extraction FASTER and MORE ACCURATE.

Two kinds of learnings, each recorded by a dedicated tool (fast Mongo upsert,
no extra LLM pass, no post-save node):

* ``label_alias`` — the filing labels a known concept differently.  The agent
  calls ``remember_alias(filing_label, canonical)``; the tool builds the text
  in a fixed format and REJECTS invalid input (identical labels, dimension
  member/taxonomy metadata).
* ``layout`` — a stable NAVIGATIONAL fact: which exhibit/section the income
  statement lives in.  The agent calls ``remember_layout(note)``; the tool
  REJECTS per-filing line numbers (they change every filing).

These reject rules are FORMAT guards: they refuse invalid input, never alter
a valid learning.  Learnings are ADVISORY ONLY and local-only (per CIK) —
they never supply values, never add concepts outside the extraction target,
never skip/exclude a concept (a metric absent today may be disclosed in a
future filing), and never override the filing.

Store:
* ``agent_memory_company`` — one doc per CIK: ``{cik, learnings, updated_at}``,
  each learning ``{id, type, text}``.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY_DB = "normalize_data"
_COMPANY_COL = "agent_memory_company"
_MAX_LEARNINGS = 30

_NOSKIP_WARNING = (
    "  ⚠ Use memory only to locate/map concepts FASTER and MORE ACCURATELY.  "
    "NEVER skip, exclude, or stop searching for a concept because of memory — "
    "a metric absent today may be disclosed in a future filing."
)

# Format guards — reject invalid input, never alter valid learnings.
# Dimension member / taxonomy metadata is NOT a label alias (Tier 0 keys
# already handle it deterministically).
_ALIAS_NOISE_RX = re.compile(
    r"\b[a-z][\w-]*:[A-Za-z0-9_]+"   # taxonomy keys: us-gaap:…, country:…
    r"|\[[^\]]*Member\]"             # [Member] dimension tags
    r"|→"                            # arrow annotations
    r"|\(segment\b",                 # (segment …
    re.IGNORECASE,
)
# Per-filing line numbers are unstable — they will be wrong next filing.
_LAYOUT_LINE_RX = re.compile(
    r"\blines?\s*[~≈-]?\s*\d+"
    r"|\bline\s*\d+"
    r"|\bat\s+line\s*\d+",
    re.IGNORECASE,
)
# Trailing date in a period label ("Three Months Ended June 30, 2026" →
# stable prefix "Three Months Ended").
_PERIOD_DATE_RX = re.compile(
    r"\s+(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s*(?:\d{4})?\s*$",
    re.IGNORECASE,
)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_db():
    from earnings_agents.integrations.normalize import _get_client
    return _get_client()[_MEMORY_DB]


# ── Recall (read path) ──────────────────────────────────────────────────────

def _learning_to_line(l: dict) -> str | None:
    """Render one learning as a single advisory bullet."""
    text = str(l.get("text") or "").strip()
    if not text:
        return None
    return f"  • {text}"


def recall_memory(
    cik: str,
    types: set[str] | None = None,
) -> dict[str, Any]:
    """Fetch advisory learnings for *cik* (optionally filtered by type).

    Returns ``{"local_block", "local_doc"}`` — ``local_block`` is
    ready-to-inject prompt text (empty string when there is nothing).
    """
    db = _get_db()
    local_doc = db[_COMPANY_COL].find_one({"cik": str(cik)}) or {}
    learnings = [
        l for l in (local_doc.get("learnings") or [])
        if (types is None or l.get("type") in types)
    ]
    local_lines = [line for line in (_learning_to_line(l) for l in learnings) if line]
    local_block = ""
    if local_lines:
        local_block = (
            "COMPANY MEMORY (advisory — the filing is authoritative):\n"
            + _NOSKIP_WARNING + "\n"
            + "\n".join(local_lines[:12])
        )
    return {"local_block": local_block, "local_doc": local_doc}


# ── Write path — tools the agent calls mid-run ─────────────────────────────

def _learning_key(l: dict) -> str:
    """Short, stable id: ``type`` + 10 hex chars of a sha1 of the text."""
    raw = f"{l['type']}:{_norm(l['text'])}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{l['type']}:{digest}"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(s)))


def _jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _containment(a: str, b: str) -> float:
    """How much of the SHORTER text's tokens appear in the longer one.

    Catches near-duplicates where one note is a rephrase + extension of the
    other (longer notes dilute Jaccard; containment does not)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _layout_is_duplicate(text: str, learnings: list) -> bool:
    """True when *text* is a near-duplicate of an existing layout learning
    (Jaccard or containment >= threshold — rephrased + extended repeats are
    caught even when one note is much longer)."""
    for existing in learnings:
        if existing.get("type") != "layout":
            continue
        other = str(existing.get("text") or "")
        sim = max(_jaccard(text, other), _containment(text, other))
        if sim >= _LAYOUT_SIMILARITY:
            return True
    return False


# Near-duplicate threshold for layout notes: the agent rephrases the same
# navigational fact slightly every run ("US/Rest of World" vs "United States
# / Rest of the World"), so exact-hash dedup alone lets near-duplicates pile
# up.  A same-type learning within this similarity is considered a repeat.
_LAYOUT_SIMILARITY = 0.5
# One short navigational fact per layout note — long multi-detail notes both
# dilute dedup similarity and accumulate contradictory details across runs.
_LAYOUT_MAX_CHARS = 220


def _persist(cik: str, kind: str, text: str) -> str:
    """Dedupe (exact hash + near-duplicate) + upsert one learning."""
    learning = {"type": kind, "text": text}
    key = _learning_key(learning)

    cik = str(cik)
    db = _get_db()
    current = db[_COMPANY_COL].find_one({"cik": cik}) or {}
    learnings = list(current.get("learnings") or [])
    if any(x.get("id") == key for x in learnings):
        return "Already remembered"
    # Layout notes are free-form: reject near-duplicates of an existing same-
    # type learning (label_alias is structured, so its exact hash is enough).
    if kind == "layout" and _layout_is_duplicate(text, learnings):
        return "Already remembered (similar)"
    learnings.append({"id": key, "type": kind, "text": text})
    db[_COMPANY_COL].update_one(
        {"cik": cik},
        {"$set": {
            "cik": cik,
            "updated_at": _now(),
            "learnings": learnings[:_MAX_LEARNINGS],
        }},
        upsert=True,
    )
    return "Remembered"


def _remember_alias(cik: str, filing_label: str, canonical: str) -> str:
    """Validate + persist one filing-label → canonical mapping."""
    filing_label = str(filing_label or "").strip()
    canonical = str(canonical or "").strip()
    if not filing_label or not canonical:
        return "Error: filing_label and canonical must be non-empty"
    if _norm(filing_label) == _norm(canonical):
        return "Skipped: filing_label and canonical are identical"
    if _ALIAS_NOISE_RX.search(filing_label) or _ALIAS_NOISE_RX.search(canonical):
        return "Skipped: dimension member / taxonomy metadata is not a label alias"
    return _persist(cik, "label_alias", f'"{canonical}" is labeled "{filing_label}"')


def _remember_layout(cik: str, note: str) -> str:
    """Validate + persist one stable navigational layout fact."""
    note = str(note or "").strip()
    if not note:
        return "Error: note must not be empty"
    if len(note) > _LAYOUT_MAX_CHARS:
        return (
            f"Skipped: note too long ({len(note)} chars) — record ONE short "
            "navigational fact (which exhibit/section the income statement "
            "is in), not a multi-detail paragraph"
        )
    if _LAYOUT_LINE_RX.search(note):
        return (
            "Skipped: line numbers are filing-specific and will be wrong next "
            "time; record stable structure instead (exhibit / section / "
            "relative position)"
        )
    return _persist(cik, "layout", note)


def build_remember_alias_tool(cik: str):
    """Return the ``remember_alias`` tool the extraction agent calls."""
    from langchain_core.tools import tool as _lc_tool

    @_lc_tool
    def remember_alias(filing_label: str, canonical: str) -> str:
        """Record that this company's filing labels a known concept differently.

        GOAL: make future extraction FASTER and MORE ACCURATE — a future run
        can map this label instantly and correctly instead of re-resolving or
        mis-mapping it.

        Args:
            filing_label: the label as printed in the filing
                (e.g. "Fair value adjustment of liabilities").
            canonical: the known concept label it maps to
                (e.g. "Change in fair value of liabilities").

        Only for genuine, stable label mappings you observed.  Do NOT record
        identical labels, dimension members/segments (handled by taxonomy
        keys), or absences.
        """
        return _remember_alias(cik, filing_label, canonical)

    return remember_alias


def build_remember_layout_tool(cik: str):
    """Return the ``remember_layout`` tool the extraction agent calls."""
    from langchain_core.tools import tool as _lc_tool

    @_lc_tool
    def remember_layout(note: str) -> str:
        """Record where the income statement / financial tables live in this company's filings.

        GOAL: make future extraction FASTER — future runs navigate straight
        there instead of searching the whole document.

        Args:
            note: ONE short STABLE navigational fact, e.g. "Income statement
                is in the second half of the release" or "Income statement is
                in Exhibit 99.2 (supplemental)".

        Do NOT include per-filing line numbers (they change every filing), do
        NOT describe filing content (e.g. how revenue is disaggregated), and
        do NOT record absences.
        """
        return _remember_layout(cik, note)

    return remember_layout


# ── Currency / period facts — recorded by the agents that detect them ─────

_CODE_RX = re.compile(r"^[A-Z]{3}$")


def _remember_currency(cik: str, currency: Any, other_codes: Any = "") -> str:
    """Validate + persist the reporting currency situation (called by the
    extraction agent after ``detect_currency`` confirms it).

    A filing may contain several currencies (e.g. a USD income statement plus
    a EUR table).  Only the extraction currency is stored plus the OTHER codes
    detected, so future runs expect them and still extract the right column.
    """
    currency = str(currency or "").strip().upper()
    if not _CODE_RX.match(currency):
        return f"Error: currency must be a 3-letter code (e.g. USD); got {currency!r}"
    others: list[str] = []
    for c in str(other_codes or "").split(","):
        c = c.strip().upper()
        if c and c != currency and _CODE_RX.match(c) and c not in others:
            others.append(c)
    if others:
        return _persist(
            cik, "currency",
            f"Reports in {currency}; filing may also contain {', '.join(others)}.",
        )
    return _persist(cik, "currency", f"Reports in {currency}.")


def _remember_period(cik: str, period_type: Any, period_label: Any) -> str:
    """Validate + persist filing cadence + period-label format (called by the
    period agent after it determines the period).  The date varies every
    filing, so only the stable parts are remembered; the period agent still
    reads the actual dates from the filing."""
    period_type = str(period_type or "").strip().lower()
    if period_type not in ("quarterly", "annual"):
        return "Skipped: period_type must be 'quarterly' or 'annual'"
    cadence = "quarterly" if period_type == "quarterly" else "annually"
    label = str(period_label or "").strip()
    prefix = _PERIOD_DATE_RX.sub("", label).strip() if label else ""
    if not prefix:
        prefix = "Three Months Ended" if period_type == "quarterly" else "Fiscal Year Ended"
    return _persist(cik, "period", f"Files {cadence}; period labels like '{prefix}'.")


def build_remember_currency_tool(cik: str):
    """Return the ``remember_currency`` tool the extraction agent calls."""
    from langchain_core.tools import tool as _lc_tool

    @_lc_tool
    def remember_currency(currency: str, other_codes: str = "") -> str:
        """Record the reporting currency situation for future runs.

        GOAL: make future extraction FASTER and MORE ACCURATE — future runs
        know which currency to extract and that other currencies may appear.

        Args:
            currency: the currency you extract (the USD column), e.g. "USD".
            other_codes: comma-separated OTHER currency codes present in the
                filing (e.g. "EUR,GBP"), or "" if the filing has only one
                currency.  Use the "Codes detected:" list from
                detect_currency() and exclude the one you extract.

        Call this only after you have detected the currency(s) via
        detect_currency().
        """
        return _remember_currency(cik, currency, other_codes)

    return remember_currency


def build_remember_period_tool(cik: str):
    """Return the ``remember_period`` tool the period agent calls."""
    from langchain_core.tools import tool as _lc_tool

    @_lc_tool
    def remember_period(period_type: str, period_label: str) -> str:
        """Record this company's filing cadence + period-label format.

        GOAL: make future period detection FASTER — future runs know the
        cadence and label format up front (they still read the actual dates
        from the filing).

        Args:
            period_type: "quarterly" or "annual" (what you determined).
            period_label: the exact period header, e.g. "Three Months Ended
                June 30, 2026" (only the stable prefix is stored).
        """
        return _remember_period(cik, period_type, period_label)

    return remember_period
