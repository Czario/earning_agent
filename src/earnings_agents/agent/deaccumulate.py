"""Year-to-date → quarterly de-accumulation for cash-flow statements.

Some filers present their cash-flow statement on a **year-to-date** basis only
("Six Months Ended" for a Q2 filing, "Nine Months Ended" for a Q3 filing) — the
printed flow figures are cumulative YTD, not the quarter.  This module
classifies each cash-flow row and de-accumulates the flows:

    Q2 = 6-month YTD − Q1
    Q3 = 9-month YTD − (Q1 + Q2)

Prior quarters are read from ``concept_values_quarterly`` for the SAME fiscal
year (the period agent's ``fiscal_year`` is authoritative).

Row classification:
  • ``flow``      — a flow over the period → de-accumulate.
  • ``ending``    — an end-of-period snapshot ("…ending balances") → use the
                    printed value as-is (it IS the quarter-end balance).
  • ``beginning`` — a beginning-of-period snapshot ("…beginning balances") →
                    SKIP.  A YTD statement shows the FISCAL-YEAR-START balance,
                    not the quarter-start balance, so it is never a quarterly
                    value.

This mirrors (and complements) the post-save Q4 derivation
(``integrations/q4.py``: ``Q4 = Annual − (Q1+Q2+Q3)``), but it runs IN-LOOP via
the extraction agent's ``deaccumulate_cashflow`` tool.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from bson import ObjectId

from earnings_agents.integrations.normalize import _get_client, _NORMALIZE_DB

logger = logging.getLogger(__name__)

# ── Row classification ───────────────────────────────────────────────────────

# Beginning-of-period snapshots (fiscal-year-start balance in a YTD statement).
_BEGIN_KW = re.compile(
    r"\b(?:beginning|opening)\b(?:\s+of\s+(?:the\s+)?(?:period|year))?"
    r"|\bstart\s+of\s+(?:the\s+)?(?:period|year)\b",
    re.I,
)
# End-of-period snapshots (period-end balance — correct as-is).
_END_KW = re.compile(
    r"\b(?:ending|closing)\b(?:\s+of\s+(?:the\s+)?(?:period|year))?"
    r"|\bend\s+of\s+(?:the\s+)?(?:period|year)\b",
    re.I,
)

# CamelCase concept-name markers (mirrors q4._POINT_IN_TIME_PATTERNS, split into
# beginning vs ending so the two are handled differently).
_BEGIN_CONCEPT_RX = re.compile(
    r"Beginning(?:Of|Balance)|Opening(?:Balance)?|AtBeginningOf",
    re.I,
)
_END_CONCEPT_RX = re.compile(
    r"Ending(?:Balance)?|Closing(?:Balance)?|EndOf(?:Year|Period)|AtEndOf|EndOfThe"
    r"|(?:^|:)(?:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents|CashAndCashEquivalentsAtCarryingValue|RestrictedCash(?:AndCashEquivalents)?AtCarryingValue|CashAndCashEquivalents|RestrictedCash)$",
    re.I,
)

# Specific beginning cash concept patterns (Rule 1: taken from prior quarter ending cash)
_BEGINNING_CASH_CONCEPT_RX = re.compile(
    r"(?:^|:)(?:Cash.*BeginningBalances?|RestrictedCash.*BeginningBalances?)$",
    re.I,
)

# Specific ending cash concept patterns (Rule 2: kept unchanged as extracted, inserted as-is)
_ENDING_CASH_CONCEPT_RX = re.compile(
    r"(?:^|:)(?:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents|CashAndCashEquivalentsAtCarryingValue|RestrictedCash(?:AndCashEquivalents)?AtCarryingValue|CashAndCashEquivalents|RestrictedCash)$",
    re.I,
)


def is_beginning_cash_concept(
    concept_name: str = "",
    taxonomy_key: str = "",
) -> bool:
    """True when *concept_name* or *taxonomy_key* represents the beginning cash balance."""
    for s in (concept_name, taxonomy_key):
        base = (s or "").split("|")[0].strip()
        if _BEGINNING_CASH_CONCEPT_RX.search(base):
            return True
    return False


def is_ending_cash_concept(
    concept_name: str = "",
    taxonomy_key: str = "",
) -> bool:
    """True when *concept_name* or *taxonomy_key* represents ending cash and cash equivalents."""
    for s in (concept_name, taxonomy_key):
        base = (s or "").split("|")[0].strip()
        if _ENDING_CASH_CONCEPT_RX.search(base):
            return True
    return False


def classify_cashflow_kind(
    concept_name: str,
    label: str = "",
    filing_label: str = "",
) -> str:
    """Return ``"beginning" | "ending" | "flow"`` for a cash-flow row.

    *filing_label* is the row label as printed in the filing (e.g. "Cash,
    cash equivalents and restricted cash, ending balances") — it is the
    authoritative beginning/ending signal, since the normalized XBRL label
    often lacks it.  *concept_name* is the XBRL concept (CamelCase) and
    *label* is the normalized concept label (fallbacks).
    """
    name = concept_name or ""
    lab = label or ""
    fl = filing_label or ""
    if _BEGIN_KW.search(fl) or _BEGIN_CONCEPT_RX.search(name) or _BEGIN_KW.search(lab):
        return "beginning"
    if _END_KW.search(fl) or _END_CONCEPT_RX.search(name) or _END_KW.search(lab):
        return "ending"
    return "flow"


# ── Period-basis detection (advisory) ───────────────────────────────────────

_PERIOD_BASIS_PATTERNS: list[tuple[re.Pattern, str, int]] = [
    (re.compile(r"three\s+months\s+ended", re.I), "quarterly", 3),
    (re.compile(r"six\s+months\s+ended", re.I), "ytd_6m", 6),
    (re.compile(r"nine\s+months\s+ended", re.I), "ytd_9m", 9),
    (
        re.compile(r"(?:twelve\s+months\s+ended|fiscal\s+year\s+ended|year\s+ended)", re.I),
        "annual", 12,
    ),
]


def detect_cashflow_period_basis(snippet: str) -> list[dict]:
    """Detect the period basis(es) declared in a text snippet.

    Returns a list of ``{"basis", "months", "evidence"}`` in document order.
    Empty when no "X Months Ended" / "Year Ended" header is present.
    """
    found: list[dict] = []
    seen: set[str] = set()
    for pat, basis, months in _PERIOD_BASIS_PATTERNS:
        m = pat.search(snippet or "")
        if m and basis not in seen:
            seen.add(basis)
            found.append({"basis": basis, "months": months, "evidence": m.group(0)})
    return found


_CF_HEADER_RX = re.compile(
    r"(?:condensed\s+)?(?:consolidated\s+)?statements?\s+of\s+cash\s+flows?",
    re.I,
)


def detect_cashflow_statement_basis(document_text: str) -> dict | None:
    """Detect the cash-flow statement's OWN column basis, in document order.

    Finds the "Statement(s) of Cash Flows" header and returns the FIRST
    period phrase after it (the cash-flow column header).  This is the
    deterministic signal the pipeline injects into the prompt: the income
    statement's "Three/Nine Months Ended" columns are ignored, so a YTD
    cash-flow statement is distinguished from a quarterly one.

    Returns ``{"basis", "months", "evidence"}`` or ``None`` when no cash-flow
    header is found.
    """
    m = _CF_HEADER_RX.search(document_text or "")
    if not m:
        return None
    window = document_text[m.end(): m.end() + 600]
    best: dict | None = None
    best_pos: int | None = None
    for pat, basis, months in _PERIOD_BASIS_PATTERNS:
        mm = pat.search(window)
        if mm and (best_pos is None or mm.start() < best_pos):
            best_pos = mm.start()
            best = {"basis": basis, "months": months, "evidence": mm.group(0)}
    return best


# ── Concept resolution ───────────────────────────────────────────────────────


def resolve_cashflow_concept(
    concept_key: str,
    target_concepts: list[dict] | None,
) -> dict | None:
    """Resolve a bracketed taxonomy key / concept name / label to a cash-flow
    concept dict from ``target_concepts`` (statement_type == "cashflow")."""
    from earnings_agents.agent.derive import _norm_label

    key = (concept_key or "").strip().strip("[]")
    key_lower = key.lower()
    cf_concepts = [
        c for c in (target_concepts or [])
        if (c.get("statement_type") or "").strip().lower() == "cashflow"
    ]
    if not cf_concepts:
        return None
    for c in cf_concepts:
        k = (c.get("taxonomy_key") or c.get("concept") or "").strip()
        if k.lower() == key_lower:
            return c
    # Also match base concept name or base tkey prefix before '|'
    for c in cf_concepts:
        concept_name = (c.get("concept") or "").strip().lower()
        base_tkey = (c.get("taxonomy_key") or "").split("|")[0].strip().lower()
        if key_lower in (concept_name, base_tkey, f"{concept_name}|cf", f"{base_tkey}|cf"):
            return c
    for c in cf_concepts:
        if _norm_label(c.get("label") or "") == _norm_label(concept_key):
            return c
    return None


# ── Prior-quarter load + de-accumulation ─────────────────────────────────────


def load_prior_quarter_values(
    cik: str,
    concept_id: Any,
    statement_type: str,
    fiscal_year: int,
    quarter: int,
) -> dict[int, float]:
    """Load Q1..Q(quarter-1) values for a concept from ``concept_values_quarterly``.

    Returns ``{quarter: value}`` (only quarters < *quarter* in the same fiscal
    year).  Missing quarters are simply absent from the returned dict.
    """
    if not concept_id or not cik:
        return {}
    try:
        oid = concept_id if isinstance(concept_id, ObjectId) else ObjectId(str(concept_id))
    except Exception:
        oid = concept_id
    db = _get_client()[_NORMALIZE_DB]
    rows = db["concept_values_quarterly"].find(
        {
            "concept_id": oid,
            "cik": cik,
            "statement_type": statement_type,
            "reporting_period.fiscal_year": fiscal_year,
            "reporting_period.quarter": {"$lt": quarter},
        }
    )
    out: dict[int, float] = {}
    for r in rows:
        q = (r.get("reporting_period") or {}).get("quarter")
        v = r.get("value")
        if isinstance(q, int) and isinstance(v, (int, float)):
            out[q] = float(v)
    return out


def load_prior_quarter_ending_cash(
    cik: str | None,
    fiscal_year: int | None,
    prior_quarter: int | None,
    target_concepts: list[dict] | None = None,
) -> float | None:
    """Load the ending cash and cash equivalents balance for the prior quarter.

    Queries concept_values_quarterly for the company's ending cash concept
    (us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents or
    CashAndCashEquivalentsAtCarryingValue) for fiscal_year and prior_quarter.
    """
    if not cik or fiscal_year is None or prior_quarter is None:
        return None

    ending_cids: list[Any] = []
    if target_concepts:
        # Check cashflow concepts first
        for c in target_concepts:
            if (c.get("statement_type") or "").strip().lower() == "cashflow":
                name = c.get("concept") or ""
                tkey = c.get("taxonomy_key") or ""
                if is_ending_cash_concept(name, tkey) and c.get("_id"):
                    ending_cids.append(c["_id"])
        # Fall back to other statements (e.g. balancesheet)
        if not ending_cids:
            for c in target_concepts:
                name = c.get("concept") or ""
                tkey = c.get("taxonomy_key") or ""
                if is_ending_cash_concept(name, tkey) and c.get("_id"):
                    ending_cids.append(c["_id"])

    if not ending_cids:
        try:
            db = _get_client()[_NORMALIZE_DB]
            docs = list(
                db["normalized_concepts_quarterly"].find(
                    {
                        "cik": cik,
                        "$or": [
                            {"concept": re.compile(r"CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", re.I)},
                            {"taxonomy_key": re.compile(r"CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", re.I)},
                            {"concept": re.compile(r"CashAndCashEquivalentsAtCarryingValue", re.I)},
                        ],
                    },
                    {"_id": 1, "statement_type": 1},
                )
            )
            docs.sort(key=lambda d: 0 if (d.get("statement_type") or "").lower() == "cashflow" else 1)
            ending_cids = [d["_id"] for d in docs if d.get("_id")]
        except Exception as exc:
            logger.warning("load_prior_quarter_ending_cash: concept lookup failed (%s)", exc)

    if ending_cids:
        try:
            db = _get_client()[_NORMALIZE_DB]
            oids = []
            for cid in ending_cids:
                try:
                    oids.append(cid if isinstance(cid, ObjectId) else ObjectId(str(cid)))
                except Exception:
                    oids.append(cid)
            rows = list(
                db["concept_values_quarterly"].find(
                    {
                        "cik": cik,
                        "concept_id": {"$in": oids},
                        "reporting_period.fiscal_year": fiscal_year,
                        "reporting_period.quarter": prior_quarter,
                    }
                )
            )
            if rows:
                rows.sort(key=lambda r: 0 if (r.get("statement_type") or "").lower() == "cashflow" else 1)
                val = rows[0].get("value")
                if isinstance(val, (int, float)):
                    return float(val)
        except Exception as exc:
            logger.warning("load_prior_quarter_ending_cash: value query failed (%s)", exc)

    return None


def _infer_prior_scale_divisor(value: float, prior_values: list[float]) -> int:
    """Detect when DB prior-quarter values are stored in full raw units
    (1,000x / 1,000,000x / 1,000,000,000x) while *value* was passed in the
    statement's printed table scale.

    Returns the divisor (1,000 / 1,000,000 / 1,000,000,000) to convert the
    DB priors into *value*'s table scale, or 1 when they are already in the
    same scale.
    """
    non_zero = [abs(p) for p in prior_values if p != 0]
    if value == 0 or not non_zero:
        return 1
    max_prior = max(non_zero)
    ratio = max_prior / abs(value)
    for mult in (1_000_000_000, 1_000_000, 1_000):
        step = max(mult // 100, 1)
        if (
            mult * 0.02 <= ratio <= mult * 50
            and max_prior >= mult
            and all(round(p) % step == 0 for p in non_zero)
        ):
            return mult
    return 1


def deaccumulate_cashflow_value(
    *,
    cik: str | None,
    value: float,
    concept_key: str,
    fiscal_year: int | None,
    quarter: int | None,
    ytd_months: int,
    target_concepts: list[dict] | None,
    filing_label: str = "",
) -> str:
    """De-accumulate one YTD cash-flow value into its quarterly value.

    Returns a human-readable instruction for the extraction agent.  Never
    mutates state — the agent reports the returned quarterly value (and lists
    the key in ``__derived__``).
    """
    from earnings_agents.agent.derive import format_value_for_llm

    if cik is None or fiscal_year is None or quarter is None:
        return (
            "Cannot de-accumulate: fiscal year/quarter context is unavailable. "
            "Report the printed value as-is."
        )
    if quarter not in (2, 3):
        return (
            f"No de-accumulation needed for quarter={quarter} (3-month = the "
            f"quarter). Report the printed value as-is."
        )
    expected = {2: 6, 3: 9}[quarter]
    if ytd_months != expected:
        return (
            f"ytd_months={ytd_months} does not match quarter={quarter} "
            f"(expected {expected} months). Re-read the cash-flow column header "
            f"and pass 6 (Q2) or 9 (Q3)."
        )

    concept = resolve_cashflow_concept(concept_key, target_concepts)
    if concept is None:
        return (
            f"No cash-flow concept matched '{concept_key}'. Use map_concept() "
            f"first to resolve the filing row to a bracketed key."
        )

    name = concept.get("concept") or concept.get("taxonomy_key") or ""
    label = concept.get("label") or ""
    stmt = (concept.get("statement_type") or "cashflow").strip().lower()
    key = (concept.get("taxonomy_key") or concept.get("concept") or "").strip()

    # Rule 1: Beginning cash balance snapshot in Q2/Q3 takes value from
    # prior quarter ending cash (us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents)
    if is_beginning_cash_concept(name, key):
        prior_q = quarter - 1
        prior_cash = load_prior_quarter_ending_cash(
            cik=cik,
            fiscal_year=fiscal_year,
            prior_quarter=prior_q,
            target_concepts=target_concepts,
        )
        if prior_cash is not None:
            divisor = _infer_prior_scale_divisor(value, [prior_cash])
            aligned_cash = prior_cash / divisor if divisor > 1 else prior_cash
            cash_str = format_value_for_llm(aligned_cash)
            return (
                f"De-accumulated Q{quarter} = {cash_str} (taken as-is from prior quarter Q{prior_q} "
                f"ending cash [us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents]).\n"
                f"Report {cash_str} under [{key}] and list [{key}] in __derived__ "
                f"(computed, not verbatim)."
            )
        return (
            f"Cannot de-accumulate [{key}]: missing prior quarter Q{prior_q} ending cash "
            f"([us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents]) for fiscal_year={fiscal_year}. "
            f"Leave the key out of finalize (or report it missing). Do NOT invent a prior-quarter value."
        )

    kind = classify_cashflow_kind(name, label, filing_label)
    if kind == "beginning":
        return (
            f"SKIP — '{label}' is a beginning-of-period snapshot: the YTD "
            f"statement shows the fiscal-year-start balance, not the "
            f"quarter-start balance. Do NOT report a quarterly value for "
            f"[{key}]."
        )
    if kind == "ending":
        val_str = format_value_for_llm(value)
        return (
            f"AS-IS — '{label}' is an end-of-period snapshot (the quarter-end "
            f"balance). Report the printed value {val_str} under [{key}] "
            f"without de-accumulating."
        )

    cid = concept.get("_id")
    priors = load_prior_quarter_values(cik, cid, stmt, fiscal_year, quarter) if cid is not None else {}
    missing = [q for q in range(1, quarter) if q not in priors]
    if missing:
        if value == 0:
            return (
                f"De-accumulated Q{quarter} = 0 (YTD is 0, prior quarters treated as 0)\n"
                f"Report 0 under [{key}] and list [{key}] in __derived__ (computed, not verbatim)."
            )
        missing_s = ", ".join(f"Q{q}" for q in missing)
        return (
            f"Cannot de-accumulate [{key}]: missing prior quarter(s) {missing_s} "
            f"for fiscal_year={fiscal_year}. Leave the key out of finalize (or "
            f"report it missing). Do NOT invent a prior-quarter value."
        )

    divisor = _infer_prior_scale_divisor(value, [priors[q] for q in range(1, quarter)])
    aligned_priors = {
        q: (priors[q] / divisor if divisor > 1 else priors[q])
        for q in range(1, quarter)
    }
    prior_sum = sum(aligned_priors[q] for q in range(1, quarter))
    quarterly = round(value - prior_sum, 6)
    breakdown = " + ".join(
        f"Q{q} {format_value_for_llm(aligned_priors[q])}" for q in range(1, quarter)
    )
    val_str = format_value_for_llm(value)
    q_str = format_value_for_llm(quarterly)
    return (
        f"De-accumulated Q{quarter} = {val_str} − ({breakdown}) = {q_str}\n"
        f"Report {q_str} under [{key}] and list [{key}] in __derived__ "
        f"(computed, not verbatim)."
    )
