"""Client for reading from and writing to the normalize_data MongoDB database.

Used when EARNINGS_SAVE_TARGET=normalize_data.  The module keeps a single
shared MongoClient instance (same pattern as mongodb_client.py) and registers
an atexit handler for clean shutdown.
"""
from __future__ import annotations

import atexit
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from bson import ObjectId
from pymongo import MongoClient, UpdateOne

from earnings_agents.config import MONGODB_URI

logger = logging.getLogger(__name__)

_NORMALIZE_DB = "normalize_data"
_client: Optional[MongoClient] = None  # type: ignore[type-arg]


def _get_client() -> MongoClient:  # type: ignore[type-arg]
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
        atexit.register(lambda: _client.close() if _client else None)  # type: ignore[union-attr]
    return _client


# ── Company lookup ───────────────────────────────────────────────────────────

def get_company_by_ticker(ticker: str) -> dict[str, Any] | None:
    """Return ``{cik, name, fiscal_year_end_month}`` for *ticker*, or ``None``.

    Queries normalize_data.companies by ``ticker_symbol`` (case-insensitive).
    ``fiscal_year_end_month`` is derived from the ``corporate_info.fiscal_year_end``
    field, which is stored as an "MMDD" string (e.g. "0630" for June 30).
    """
    db = _get_client()[_NORMALIZE_DB]
    doc = db["companies"].find_one(
        {"ticker_symbol": ticker.upper()},
        {"cik": 1, "name": 1, "corporate_info.fiscal_year_end": 1},
    )
    if doc is None:
        return None
    fy_code: str = (doc.get("corporate_info") or {}).get("fiscal_year_end", "1231") or "1231"
    try:
        fy_end_month = int(fy_code[:2])
        if not (1 <= fy_end_month <= 12):
            fy_end_month = 12
    except (ValueError, TypeError):
        fy_end_month = 12
    return {
        "cik": str(doc["cik"]),
        "name": doc.get("name", ticker),
        "fiscal_year_end_month": fy_end_month,
        "fiscal_year_end_code": fy_code,
    }


# ── Concept lookup ───────────────────────────────────────────────────────────

_CONCEPT_PREFIX_RX = re.compile(r"(?:us-gaap|system|ifrs-full|dei|srt):", re.IGNORECASE)
_MEMBER_RX = re.compile(
    r"(?:us-gaap|system|ifrs-full|dei|srt):([A-Za-z0-9_]+)", re.IGNORECASE
)
_FULL_MEMBER_RX = re.compile(
    r"((?:us-gaap|ifrs-full|dei|srt):[A-Za-z0-9_]+Member)", re.IGNORECASE
)
_CAMEL_SPLIT_RX = re.compile(r"(?<!^)(?=[A-Z])")


def _extract_member_tag(raw: str) -> str:
    """Return the full XBRL member concept tag from a raw label string.

    Raw dimensional labels look like::

        "Net sales\\n\\n\\nus-gaap:ProductMember"

    Returns the full tag (e.g. ``"us-gaap:ProductMember"``) or ``""``.
    """
    m = _FULL_MEMBER_RX.search(raw)
    return m.group(1) if m else ""


def _clean_label(raw: str) -> tuple[str, str]:
    """Split a raw concept label into ``(base_label, member_qualifier)``.

    Some upstream rows store ``label`` as a multi-line string with an XBRL
    axis member appended after blank lines, e.g.::

        "Net sales\\n\\n\\n\\nus-gaap:ProductMember"

    Splitting it lets us:
      * use ``base_label`` (``"Net sales"``) so the LLM can match the document
        text directly when the breakdown labels are already unique;
      * fall back to ``"Net sales (Product)"`` only when another row in the
        same statement collapses to the same ``base_label``.

    Returns ``("", "")`` when the raw string is empty or contains nothing but
    a concept reference.
    """
    if not raw:
        return "", ""
    parts = _CONCEPT_PREFIX_RX.split(raw, maxsplit=1)
    head = re.sub(r"\s+", " ", parts[0]).strip()
    if not head:
        return "", ""
    member = ""
    m = _MEMBER_RX.search(raw)
    if m:
        token = re.sub(r"Member$", "", m.group(1))
        member = _CAMEL_SPLIT_RX.sub(" ", token).strip()
    return head, member


def _ancestor_labels(path: str, path_to_head: dict[str, str]) -> list[str]:
    """Return ancestor labels for *path* from top-level down to immediate parent.

    e.g. ``"001.001.001"`` → ``["Total revenue", "Original segmentation"]``.
    """
    if not path:
        return []
    parts = path.split(".")
    chain: list[str] = []
    for i in range(1, len(parts)):
        lbl = path_to_head.get(".".join(parts[:i]))
        if lbl:
            chain.append(lbl)
    return chain


def _distinguishing_parent_label(
    path: str,
    sibling_paths: list[str],
    path_to_head: dict[str, str],
) -> str:
    """Return the shallowest ancestor label that distinguishes *path* from its
    same-labelled siblings, or ``""`` when no ancestor label does.

    Two rows can share a label (e.g. "Hardware" under both Revenue and Cost of
    Revenue).  The path encodes which parent each row belongs to, so we walk
    the ancestor chain (top-level first) and return the first level at which
    this row's ancestor differs from all siblings.
    """
    mine = _ancestor_labels(path, path_to_head)
    max_depth = max(
        (len(_ancestor_labels(p, path_to_head)) for p in sibling_paths),
        default=0,
    )
    for depth in range(max_depth):
        my_lbl = mine[depth] if depth < len(mine) else None
        if not my_lbl:
            continue
        others = {
            _ancestor_labels(p, path_to_head)[depth]
            if depth < len(_ancestor_labels(p, path_to_head))
            else None
            for p in sibling_paths
            if p != path
        }
        if my_lbl not in others:
            return my_lbl
    return ""


def get_statement_concepts(
    cik: str,
    statement_types: list[str] | None = None,
    period_type: str = "quarterly",
    concept_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return sorted concept dicts for *cik* and *statement_types*.

    *period_type* selects the source collection:
      - ``"quarterly"`` (default) → ``normalized_concepts_quarterly``
      - ``"annual"``              → ``normalized_concepts_annual``

    *concept_ids* (optional) restricts the load to exactly those concept
    ``_id`` strings — the recent-value window filter is applied HERE, at query
    time, so non-recent concepts are never even loaded.  EXCEPTION: system:
    (``concept`` matching ``^system:``) and ``calculated`` concepts are ALWAYS
    loaded regardless of the window — they are the CALC derivation targets,
    and excluding them would make them permanently underivable.

    Filters out only abstract (``abstract: true``) and hidden (``hide: true``)
    rows.  All other rows — including calculated/system concepts, dimensional
    breakdown rows, and XBRL structural labels — are included.

    Results are sorted by ``path`` so the prompt lists concepts in statement
    order.

    Each returned dict has keys: ``_id`` (str), ``concept`` (GAAP name),
    ``label`` (cleaned, disambiguated only when needed), ``path``,
    ``statement_type``, ``taxonomy_key`` (stable XBRL identity used as the
    JSON key in the extraction prompt and as the mapping key back to
    ``concept_id``), plus ``dimension`` / ``dimension_concept`` (bool) as
    the authoritative dimensional-row signal.

    Rows whose ``label`` is empty after cleanup are dropped with a debug log.
    When two rows in the same statement collapse to the same base label, the
    axis member qualifier (e.g. ``"(Product)"``) is appended when available;
    otherwise the row is disambiguated by its parent section (derived from
    ``path``, e.g. ``"Hardware (Cost of Revenue)"``).  Taxonomy keys are
    path-qualified when the same GAAP concept appears under multiple parents,
    so extraction JSON keys never collide.
    """
    if statement_types is None:
        statement_types = ["income"]
    collection_name = (
        "normalized_concepts_annual"
        if period_type == "annual"
        else "normalized_concepts_quarterly"
    )
    db = _get_client()[_NORMALIZE_DB]
    from earnings_agents.hooks import report_call as _report_call
    _report_call(f"  [db]  query {collection_name}  concepts for CIK {cik}")

    query: dict[str, Any] = {
        "cik": cik,
        "statement_type": {"$in": statement_types},
        "active": {"$ne": False},
        "$or": [
            # Regular concepts: not abstract, not hidden
            {"abstract": {"$ne": True}, "hide": {"$ne": True}},
            # Calculated/system: include regardless of abstract/hide flag
            {"concept": {"$regex": "^system:", "$options": "i"}},
            {"calculated": {"$in": [True, "True", "true"]}},
        ],
    }
    if concept_ids is not None:
        oids: list[Any] = []
        for cid in concept_ids:
            try:
                oids.append(ObjectId(cid))
            except Exception:  # noqa: BLE001 — skip malformed ids
                continue
        if not oids:
            return []
        # Recent-window filter with the system:/calculated exemption — those
        # concepts are always loaded (derivation targets), everything else
        # must be inside the recent-value window.
        query["$and"] = [{
            "$or": [
                {"_id": {"$in": oids}},
                {"concept": {"$regex": "^system:", "$options": "i"}},
                {"calculated": {"$in": [True, "True", "true"]}},
            ],
        }]

    cursor = db[collection_name].find(
        query,
        {
            "_id": 1,
            "concept": 1,
            "label": 1,
            "path": 1,
            "order_key": 1,
            "statement_type": 1,
            "dimension": 1,
            "dimension_concept": 1,
        },
    # path alone is not unique (sibling rows can share a path, disambiguated
    # by order_key) — sort by both so row order is deterministic across runs.
    ).sort([("path", 1), ("order_key", 1)])

    # First pass: collect rows with cleaned labels.
    parsed: list[tuple[dict[str, Any], str, str, str, str]] = []  # (doc, head, member, member_tag, path)
    base_counts: dict[tuple[str, str], int] = {}
    base_key_counts: dict[tuple[str, str], int] = {}
    label_groups: dict[tuple[str, str], list[str]] = {}
    path_to_head: dict[str, str] = {}
    for d in cursor:
        concept = d.get("concept", "") or ""
        raw_label = d.get("label", "")
        head, member = _clean_label(raw_label)
        if not head:
            logger.debug(
                "get_statement_concepts: dropping concept with empty label "
                "(cik=%s concept=%s raw_label=%r)",
                cik, concept, raw_label,
            )
            continue
        member_tag = _extract_member_tag(raw_label)
        path = d.get("path", "") or ""
        st = d.get("statement_type", "")
        parsed.append((d, head, member, member_tag, path))
        key = (st, head.lower())
        base_counts[key] = base_counts.get(key, 0) + 1
        label_groups.setdefault(key, []).append(path)
        tkey = (f"{concept}|{member_tag}" if member_tag else concept).lower()
        base_key_counts[(st, tkey)] = base_key_counts.get((st, tkey), 0) + 1
        # First row at a path wins: duplicate paths exist (sibling rows
        # disambiguated by order_key), so last-write would be arbitrary and
        # order-dependent.
        if path and path not in path_to_head:
            path_to_head[path] = head

    # Second pass: disambiguate labels and taxonomy keys only when collisions
    # actually occur; dedup exact duplicates (same statement + concept + path).
    out: list[dict[str, Any]] = []
    seen_final: set[tuple[str, str, str]] = set()
    for d, head, member, member_tag, path in parsed:
        concept = d.get("concept", "") or ""
        st = d.get("statement_type", "")
        base_key = (st, head.lower())

        # Label disambiguation: prefer the XBRL member qualifier; otherwise
        # use the parent (ancestor) label from the path so the LLM can tell
        # "Hardware" under Revenue apart from "Hardware" under Cost of Revenue.
        if base_counts[base_key] > 1:
            if member:
                final_label = f"{head} ({member})"
            else:
                parent = _distinguishing_parent_label(
                    path, label_groups.get(base_key, []), path_to_head
                )
                if parent and parent.lower() != head.lower():
                    final_label = f"{head} ({parent})"
                else:
                    final_label = head
        else:
            final_label = head

        # Taxonomy key must be unique per row or the LLM's JSON keys collide
        # (e.g. us-gaap:ServiceMember under both Revenue and Cost of Revenue).
        taxonomy_key = f"{concept}|{member_tag}" if member_tag else concept
        tkey = taxonomy_key.lower()
        if base_key_counts[(st, tkey)] > 1 and path:
            taxonomy_key = f"{taxonomy_key}|{path}"

        # Drop only exact duplicates (same concept + path in the same statement).
        dedup_key = (st, concept.lower(), path)
        if dedup_key in seen_final:
            logger.debug(
                "get_statement_concepts: dropping duplicate concept/path %r "
                "(cik=%s concept=%s path=%r)",
                final_label, cik, concept, path,
            )
            continue
        seen_final.add(dedup_key)

        out.append(
            {
                "_id": str(d["_id"]),
                "concept": concept,
                "label": final_label,
                "path": path,
                "statement_type": st,
                "taxonomy_key": taxonomy_key,
                "dimension": bool(d.get("dimension")),
                "dimension_concept": bool(d.get("dimension_concept")),
            }
        )
    return out


def get_calculated_concepts(
    cik: str,
    statement_types: list[str] | None = None,
    period_type: str = "quarterly",
) -> list[dict[str, Any]]:
    """Return calculated/system concept dicts for *cik*.

    Mirrors ``get_statement_concepts`` but returns ONLY the rows that function
    excludes — i.e. rows with a ``system:``-prefixed concept name or
    ``calculated: True``.  These represent metrics that the downstream
    normaliser derives; they are not present verbatim in earnings press releases
    but can be computed from extracted values by the derivation engine in
    ``analysis/calculators.py``.

    Each returned dict has the same shape as ``get_statement_concepts`` output
    (``_id``, ``concept``, ``label``, ``path``, ``statement_type``) so it can
    be passed alongside ``target_concepts`` to ``derive_missing_concept_metrics``.
    """
    if statement_types is None:
        statement_types = ["income"]
    collection_name = (
        "normalized_concepts_annual"
        if period_type == "annual"
        else "normalized_concepts_quarterly"
    )
    db = _get_client()[_NORMALIZE_DB]
    from earnings_agents.hooks import report_call as _report_call
    _report_call(f"  [db]  query {collection_name}  calculated concepts for CIK {cik}")
    cursor = db[collection_name].find(
        {
            "cik": cik,
            "statement_type": {"$in": statement_types},
            "abstract": {"$ne": True},
            "hide": {"$ne": True},
            "active": {"$ne": False},
            "$or": [
                {"concept": {"$regex": "^system:", "$options": "i"}},
                {"calculated": {"$in": [True, "True", "true"]}},
            ],
        },
        {
            "_id": 1,
            "concept": 1,
            "label": 1,
            "path": 1,
            "statement_type": 1,
        },
    ).sort("path", 1)

    out: list[dict[str, Any]] = []
    for d in cursor:
        raw_label = d.get("label", "")
        head, _ = _clean_label(raw_label)
        if not head:
            logger.debug(
                "get_calculated_concepts: dropping concept with empty label "
                "(cik=%s concept=%s raw_label=%r)",
                cik, d.get("concept", ""), raw_label,
            )
            continue
        out.append(
            {
                "_id": str(d["_id"]),
                "concept": d.get("concept", ""),
                "label": head,
                "path": d.get("path", ""),
                "statement_type": d.get("statement_type", ""),
            }
        )

    logger.debug(
        "get_calculated_concepts: found %d calculated concept(s) for cik=%s (%s)",
        len(out), cik, period_type,
    )
    return out


# ── Period helpers ───────────────────────────────────────────────────────────

_MONTH_NAME_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[.]?\s+(\d{1,2}),?\s+(\d{4})",
    re.IGNORECASE,
)

# Keywords that indicate a full-year / annual period.
_ANNUAL_PERIOD_RE = re.compile(
    r"\b(year|twelve\s+months?|52\s+weeks?|53\s+weeks?|annual|full[- ]year)\b",
    re.IGNORECASE,
)

# Number words that appear in US earnings period strings.
_PERIOD_WORD_NUMS: dict[str, int] = {
    "three": 3,
    "six": 6,
    "nine": 9,
    "twelve": 12,
    "thirteen": 13,
    "twenty-six": 26,
    "thirty-nine": 39,
    "fifty-two": 52,
    "fifty-three": 53,
}

# Matches "Thirteen Weeks", "26 Weeks", "Six Months", "9 Months", etc.
# Longer word forms must precede their sub-strings in the alternation.
_DURATION_RE = re.compile(
    r"\b(thirteen|twenty-six|thirty-nine|fifty-(?:two|three)"
    r"|three|six|nine|twelve|\d+)"
    r"\s+(weeks?|months?)\b",
    re.IGNORECASE,
)

# Matches "First Quarter", "Second Quarter", etc. or bare "Q1"–"Q4".
_ORDINAL_QUARTER_RE = re.compile(
    r"\b(first|second|third|fourth)\s+quarter\b|\bQ([1-4])\b",
    re.IGNORECASE,
)
_ORDINAL_TO_NUM: dict[str, int] = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _extract_duration(period_str: str) -> tuple[int, str] | None:
    """Return ``(count, unit)`` parsed from *period_str*, or ``None``.

    *unit* is either ``"weeks"`` or ``"months"``.
    """
    m = _DURATION_RE.search(period_str)
    if not m:
        return None
    raw = m.group(1).lower()
    unit = "months" if m.group(2).lower().startswith("m") else "weeks"
    try:
        count = int(raw)
    except ValueError:
        count = _PERIOD_WORD_NUMS.get(raw)
        if count is None:
            return None
    return count, unit


def _quarter_from_period_str(period_str: str) -> int | None:
    """Return the fiscal quarter (1–3) inferred from *period_str*, or ``None``.

    Uses cumulative duration to determine the quarter:

    * Week-based unambiguous values:
      26 w → Q2, 39 w → Q3.  (13 weeks is ambiguous — falls back to date math.)
    * Month-based unambiguous values:
      6 m → Q2, 9 m → Q3.  (3 months is ambiguous — falls back to date math.)
    * Ordinal words: "First Quarter" → Q1, etc.

    ``"Thirteen Weeks"`` / ``"3 Months"`` is **never** inferred as Q1 because
    many companies report non-cumulatively — they print the same duration
    label for every quarter.  The caller must fall back to calendar-month
    math using the company's ``fiscal_year_end_month`` and the period-end date.

    Annual periods (52/53 weeks, 12 months) are handled by
    ``detect_period_type`` upstream and never reach here.
    """
    if not period_str:
        return None

    # Explicit ordinal label takes priority.
    m = _ORDINAL_QUARTER_RE.search(period_str)
    if m:
        if m.group(1):
            return _ORDINAL_TO_NUM.get(m.group(1).lower())
        return int(m.group(2))

    duration = _extract_duration(period_str)
    if duration is None:
        return None
    count, unit = duration

    if unit == "weeks":
        # 13 w is NOT inferred as Q1 — just like "3 months", it is
        # ambiguous (non-cumulative reporters print "Thirteen Weeks"
        # for every quarter).  Only cumulative durations are unambiguous.
        if 25 <= count <= 27:
            return 2
        if 38 <= count <= 40:
            return 3
    elif unit == "months":
        # 3 months is ambiguous — not inferred.  6/9 months are cumulative.
        if count == 6:
            return 2
        if count == 9:
            return 3
    return None


def detect_period_type(period_str: str) -> str:
    """Return ``'annual'`` or ``'quarterly'`` based on *period_str*.

    Annual indicators: "Year Ended", "Twelve Months Ended", "52/53 Weeks Ended".
    Everything else (including "Three Months", "Thirteen Weeks") → quarterly.
    """
    return "annual" if _ANNUAL_PERIOD_RE.search(period_str) else "quarterly"


def parse_period_end_date(period_str: str) -> date | None:
    """Extract a ``date`` from a period string like "Three Months Ended March 31, 2026".

    Handles both full month names ("March") and abbreviated forms ("Mar").
    Returns ``None`` when no recognisable date pattern is found.
    """
    m = _MONTH_NAME_RE.search(period_str)
    if m:
        month_name = m.group(1)
        date_str = f"{month_name} {m.group(2)} {m.group(3)}"
        # Try full month name first ("March"), then abbreviated ("Mar").
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
    return None


def compute_fiscal_period(
    period_end_date: date,
    fiscal_year_end_month: int,
    period_str: str = "",
) -> tuple[int, int]:
    """Return ``(fiscal_year, quarter)`` for a period end date.

    ``fiscal_year_end_month``: 1-12 (e.g. 6 for June, 12 for December).

    *Fiscal year* is determined by comparing the period-end month against
    ``fiscal_year_end_month``.  *Quarter* uses the company's
    ``fiscal_year_end_month`` to divide the fiscal year into 3-month
    blocks — no prediction from duration text (see below).

    When *period_str* carries an unambiguous cumulative duration
    (e.g. "Twenty-Six Weeks" → Q2, "Six Months" → Q2, "Nine Months" → Q3,
    or an ordinal like "Second Quarter"), that value is used.
    "Thirteen Weeks" and "Three Months" are **never** treated as Q1
    because many companies report non-cumulatively — the same label
    appears for every quarter.  The function falls back to calendar-month
    math using *period_end_date* (or the date in *period_str*, which is
    preferred when available).

    The period_str **date** (e.g. "May 25, 2026" from "Three Months Ended
    May 25, 2026") overrides *period_end_date* when both are present and
    differ — the LLM reads the actual column header from the document,
    which is more trustworthy than SEC EDGAR's company-set reportDate on 8-Ks.

    Examples (MSFT, fy_end_month=6):
      - 2026-03-31 → FY2026 Q3
      - 2026-06-30 → FY2026 Q4
      - 2025-09-30 → FY2026 Q1

    Examples (SMPL, fy_end_month=8):
      - "Three Months Ended May 30, 2026"   → FY2026 Q3
      - "Thirteen Weeks Ended May 30, 2026"  → FY2026 Q3
      - "Thirteen Weeks Ended Nov 30, 2025"  → FY2026 Q1
    """
    m = period_end_date.month
    y = period_end_date.year

    # Quarter is always derived from the period end date — no text-based
    # extraction needed (avoids matching comparison quarters in the document
    # body).  The period end date is unambiguous and compute_fiscal_period
    # handles all fiscal year-end conventions correctly.

    # The LLM's period_str date (e.g. "Three Months Ended May 25, 2026"
    # read from the column header) may differ from SEC's reportDate when
    # the LLM reads a comparison column by mistake.  SEC's reportDate
    # (the *period_end_date* argument) is authoritative — only fall back
    # to the LLM's date when no SEC date is available.
    str_date = parse_period_end_date(period_str)
    _has_sec_date = period_end_date is not None
    if str_date is not None and str_date != period_end_date and not _has_sec_date:
        logger.debug(
            "compute_fiscal_period: no SEC date — using period_str date %s",
            str_date,
        )
        m = str_date.month
        y = str_date.year

    # Fiscal year: if period month falls within the FY ending in fy_end_month
    # of year y (i.e. m <= fy_end_month) → fiscal_year = y; otherwise y + 1.
    fiscal_year = y if m <= fiscal_year_end_month else y + 1

    # Quarter: prefer unambiguous period-string-derived value; fall back to
    # calendar-month math (no guessing — "13 weeks"/"3 months" are ambiguous).
    quarter = _quarter_from_period_str(period_str)
    if quarter is None:
        # Calendar-month fallback — exact for companies whose quarter ends
        # align with calendar month boundaries (the vast majority).
        fy_start_month = fiscal_year_end_month % 12 + 1
        fy_month_offset = (m - fy_start_month) % 12
        quarter = fy_month_offset // 3 + 1

        # Boundary tolerance (Dec FY only): dates within 3 days after a
        # standard quarter boundary may actually belong to the PREVIOUS
        # quarter (e.g. July 3, 3 days after June 30 → should be Q2).
        # Only for December fiscal year because non-Dec FY boundaries
        # are naturally offset and standard math is correct.
        _BOUNDARY_TOLERANCE = 3
        if fiscal_year_end_month == 12:
            for _q in range(1, 5):
                _q_end_month = (fy_start_month + _q * 3 - 1) % 12 or 12
                _boundary_years = [y]
                if _q == 4 and m <= 3:
                    _boundary_years.append(y - 1)
                for _by in _boundary_years:
                    try:
                        if _q_end_month == 12:
                            _boundary = date(_by, 12, 31)
                        else:
                            _boundary = date(_by, _q_end_month + 1, 1) - timedelta(days=1)
                    except (ValueError, OverflowError):
                        continue
                    _diff_days = (period_end_date - _boundary).days
                    if 0 <= _diff_days <= _BOUNDARY_TOLERANCE and _q != quarter:
                        logger.debug(
                            "compute_fiscal_period: end_date %s is %d days after "
                            "Q%d boundary %s — snapping quarter %d → %d",
                            period_end_date, _diff_days, _q, _boundary, quarter, _q,
                        )
                        quarter = _q
                        if _q == 4 and _by == y - 1:
                            fiscal_year = _by
                        break

    return fiscal_year, quarter


def parse_period_start_date(period_str: str, end_date: date) -> date | None:
    """Return the first calendar day of the reporting period, or ``None``.

    Uses the cumulative duration encoded in *period_str* to count backwards
    from *end_date*:

    * Week-based: ``end_date - (weeks * 7) + 1 day``
      e.g. "Thirteen Weeks Ended May 2, 2026" → Feb 1, 2026
    * Month-based: first day of the month that is *n* months before the
      end month (inclusive of the period).
      e.g. "Six Months Ended June 30, 2026" → Jan 1, 2026

    Returns ``None`` when no duration can be parsed from *period_str*.
    """
    duration = _extract_duration(period_str)
    if duration is None:
        return None
    count, unit = duration
    if unit == "weeks":
        return end_date - timedelta(weeks=count) + timedelta(days=1)
    # months
    start_m = end_date.month - count + 1
    start_y = end_date.year
    while start_m <= 0:
        start_m += 12
        start_y -= 1
    return date(start_y, start_m, 1)


# ── Latest period lookup ─────────────────────────────────────────────────────

def get_latest_period(cik: str) -> dict[str, Any] | None:
    """Return the most recently stored period for *cik* across both collections.

    Queries ``concept_values_annual`` and ``concept_values_quarterly`` and
    returns the record with the latest ``reporting_period.end_date``.

    Returned dict has keys:
      ``period_type``  — ``"annual"`` or ``"quarterly"``
      ``fiscal_year``  — int
      ``quarter``      — int | None  (None for annual)
      ``end_date``     — ``datetime`` (UTC)

    Returns ``None`` when no data exists for *cik* in either collection.
    """
    db = _get_client()[_NORMALIZE_DB]
    best: dict[str, Any] | None = None
    best_end: datetime | None = None

    for period_type in ("quarterly", "annual"):
        col = db[f"concept_values_{period_type}"]
        doc = col.find_one(
            {"cik": cik, "statement_type": "income"},
            {"reporting_period.end_date": 1, "reporting_period.fiscal_year": 1,
             "reporting_period.quarter": 1},
            sort=[("reporting_period.end_date", -1)],
        )
        if doc is None:
            continue
        rp = doc.get("reporting_period", {})
        end_dt: datetime | None = rp.get("end_date")
        if end_dt is None:
            continue
        if best_end is None or end_dt > best_end:
            best_end = end_dt
            best = {
                "period_type": period_type,
                "fiscal_year": rp.get("fiscal_year"),
                "quarter": rp.get("quarter"),  # None for annual
                "end_date": end_dt,
            }

    return best


def fiscal_period_exists(
    cik: str,
    fiscal_year: int,
    quarter: int | None = None,
) -> bool:
    """Return True when concept values already exist for *cik* + *fiscal_year* (+ *quarter*).

    When *quarter* is None the check is against ``concept_values_annual``
    (annual / full-year filings); otherwise ``concept_values_quarterly``.
    """
    db = _get_client()[_NORMALIZE_DB]
    collection_name = (
        "concept_values_annual" if quarter is None else "concept_values_quarterly"
    )
    filt: dict[str, Any] = {
        "cik": cik,
        "statement_type": "income",
        "reporting_period.fiscal_year": fiscal_year,
    }
    if quarter is not None:
        filt["reporting_period.quarter"] = quarter
    return db[collection_name].count_documents(filt, limit=1) > 0


def count_fiscal_period_values(
    cik: str,
    fiscal_year: int,
    quarter: int | None = None,
) -> int:
    """Return the number of concept values stored for *cik* + *fiscal_year* (+ *quarter*).

    When *quarter* is None the count is against ``concept_values_annual``
    (annual / full-year filings); otherwise ``concept_values_quarterly``.
    """
    db = _get_client()[_NORMALIZE_DB]
    collection_name = (
        "concept_values_annual" if quarter is None else "concept_values_quarterly"
    )
    filt: dict[str, Any] = {
        "cik": cik,
        "statement_type": "income",
        "reporting_period.fiscal_year": fiscal_year,
    }
    if quarter is not None:
        filt["reporting_period.quarter"] = quarter
    return db[collection_name].count_documents(filt)


def delete_fiscal_period(
    cik: str,
    fiscal_year: int,
    quarter: int | None = None,
) -> int:
    """Delete all concept values for *cik* + *fiscal_year* (+ *quarter*).

    When *quarter* is None the delete is against ``concept_values_annual``
    (annual / full-year filings); otherwise ``concept_values_quarterly``.

    Returns the number of documents deleted.
    """
    db = _get_client()[_NORMALIZE_DB]
    collection_name = (
        "concept_values_annual" if quarter is None else "concept_values_quarterly"
    )
    filt: dict[str, Any] = {
        "cik": cik,
        "statement_type": "income",
        "reporting_period.fiscal_year": fiscal_year,
    }
    if quarter is not None:
        filt["reporting_period.quarter"] = quarter
    result = db[collection_name].delete_many(filt)
    if result.deleted_count:
        period_label = f"FY{fiscal_year} Q{quarter}" if quarter is not None else f"FY{fiscal_year} (annual)"
        logger.info(
            "delete_fiscal_period: removed %d document(s) from %s for CIK %s %s",
            result.deleted_count, collection_name, cik, period_label,
        )
    return result.deleted_count


def get_recently_valued_concept_ids(
    cik: str,
    period_type: str = "quarterly",
    n_periods: int = 3,
) -> set[str]:
    """Return concept_id strings that had a value in the last *n_periods* periods.

    Queries the ``concept_values_{annual|quarterly}`` collection for *cik*,
    finds the *n_periods* most recent distinct ``reporting_period.end_date``
    values, and returns the set of ``concept_id`` values (as strings) that had
    at least one stored value in any of those periods.

    Purpose: the HARD extraction-target filter.  A concept that has not been
    reported in any of the recent periods is very unlikely to appear in the
    current filing, so it is dropped from the extraction target entirely — it
    never reaches the agent prompt, mapping, derivation, or save.

    Returns an **empty set** when no history exists; the caller skips the run
    (nothing to extract).
    """
    col_name = (
        "concept_values_annual" if period_type == "annual"
        else "concept_values_quarterly"
    )
    db = _get_client()[_NORMALIZE_DB]
    col = db[col_name]
    periods = col.distinct(
        "reporting_period.end_date",
        {"cik": cik, "statement_type": "income"},
    )
    periods = sorted([p for p in periods if p is not None], reverse=True)[:n_periods]
    if not periods:
        return set()
    ids = col.distinct(
        "concept_id",
        {
            "cik": cik,
            "statement_type": "income",
            "reporting_period.end_date": {"$in": periods},
        },
    )
    return {str(i) for i in ids if i is not None}


def upsert_concept_values(
    cik: str,
    company_name: str,
    concept_metrics: dict[str, float],
    period_str: str,
    fiscal_year_end_month: int,
    fiscal_year_end_code: str = "1231",
    statement_type: str = "income",
    report_date: date | None = None,
    period_type_override: str | None = None,
    derived_concept_ids: set[str] | None = None,
    accession_number: str | None = None,
) -> int:
    """Bulk-upsert concept values into the appropriate collection.

    Routes to ``concept_values_quarterly`` or ``concept_values_annual``.
    Routing precedence (highest first):

    1. *period_type_override* — when the caller already resolved the period
       type upstream (``state["detected_period_type"]`` from
       ``load_company_concepts_node``), it is authoritative.  This keeps the
       prompt's period selection and the save collection in lock-step from a
       single source of truth.
    2. **Fiscal year-end month** — when the resolved period-end month equals
       the company's *fiscal_year_end_month*, the filing is annual.
    3. **Duration keywords** in *period_str* (``detect_period_type``):
      - "Three Months Ended …" / "Thirteen Weeks Ended …" → quarterly
      - "Year Ended …" / "Twelve Months Ended …" / "52/53 Weeks Ended …" → annual

    *report_date*: when provided (sourced from the SEC submissions API
    ``reportDate`` field), it overrides the end date that would otherwise be
    parsed from *period_str* via ``parse_period_end_date``.  This ensures the
    stored ``end_date`` is the exact period-end date declared to the SEC rather
    than an LLM-extracted approximation.  *period_str* is still used for
    duration detection (quarterly vs annual, quarter number, start date).

    Documents are written to match the existing schema used by the SEC-based
    pipeline, with ``concept_id`` stored as ``ObjectId`` and ``end_date`` as
    a native ``datetime`` so the upsert filter correctly de-duplicates
    re-runs of the same earnings release.

    Returns the number of operations submitted (0 on early-exit failures).
    """
    if not concept_metrics:
        logger.debug("upsert_concept_values: empty concept_metrics — nothing to do")
        return 0

    # ── Resolve the period end date ──────────────────────────────────────
    # LLM __period__ is the sole source — it reads the actual document header
    # and identifies the correct column.  No SEC fallback.
    end_date = parse_period_end_date(period_str)
    if end_date is None:
        logger.warning(
            "upsert_concept_values: LLM __period__ not parseable from %r — refusing to save",
            period_str,
        )
        return 0

    # ── Annual vs quarterly routing ─────────────────────────────────────
    # Uses LLM's __period__ duration ("Three Months Ended" vs "Year Ended")
    # with fiscal-year-end-month as fallback.
    if period_type_override in ("annual", "quarterly"):
        period_type = period_type_override
    elif period_str:
        period_type = detect_period_type(period_str)
    elif end_date.month == fiscal_year_end_month:
        period_type = "annual"
    else:
        period_type = "quarterly"
    collection_name = f"concept_values_{period_type}"
    form_type = "10-K" if period_type == "annual" else "10-Q"

    fiscal_year, quarter = compute_fiscal_period(end_date, fiscal_year_end_month, period_str)
    start_date = parse_period_start_date(period_str, end_date) if period_str else None
    # Store end_date as a native UTC datetime to match the existing collection schema.
    end_datetime = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0,
                            tzinfo=timezone.utc)
    period_date_str = end_date.strftime("%Y-%m-%d")
    now = datetime.now(tz=timezone.utc)

    db = _get_client()[_NORMALIZE_DB]
    collection = db[collection_name]

    ops: list[UpdateOne] = []
    for concept_id_str, value in concept_metrics.items():
        try:
            concept_oid = ObjectId(concept_id_str)
        except Exception:  # noqa: BLE001 — invalid ObjectId string, skip
            logger.warning(
                "upsert_concept_values: invalid ObjectId %r — skipping", concept_id_str
            )
            continue

        period_doc: dict[str, Any] = {
            "end_date": end_datetime,
            "period_date": period_date_str,
            "fiscal_year": fiscal_year,
        }
        # Annual periods have no quarter dimension and no start_date; quarterly do.
        if period_type == "quarterly":
            period_doc["quarter"] = quarter
            if start_date is not None:
                period_doc["start_date"] = datetime(
                    start_date.year, start_date.month, start_date.day,
                    tzinfo=timezone.utc,
                )

        doc: dict[str, Any] = {
            "concept_id": concept_oid,
            "cik": cik,
            "statement_type": statement_type,
            "form_type": form_type,
            "reporting_period": period_doc,
            "value": value,
            "earning_data": True,
            "created_at": now,
            "dimension_value": False,
            "calculated": concept_id_str in (derived_concept_ids or set()),
        }
        if accession_number:
            doc["accession_number"] = accession_number
        filter_doc: dict[str, Any] = {
            "cik": cik,
            "concept_id": concept_oid,
            "reporting_period.fiscal_year": fiscal_year,
        }
        if period_type == "quarterly":
            filter_doc["reporting_period.quarter"] = quarter

        ops.append(
            UpdateOne(
                filter_doc,
                {"$set": doc},
                upsert=True,
            )
        )

    if not ops:
        return 0

    # ── Delete existing data for this period before inserting fresh ───────
    # Avoids duplicate-key errors when the same fiscal period was previously
    # stored with a slightly different end_date (e.g. Q3 vs Q2 reclassification
    # of the same June 27 period).  The unique index is on {cik,
    # concept_id, fiscal_year, quarter}, so deleting by these keys first
    # guarantees clean insertion.
    _del_filt: dict[str, Any] = {
        "cik": cik,
        "statement_type": statement_type,
        "reporting_period.fiscal_year": fiscal_year,
    }
    if period_type == "quarterly":
        _del_filt["reporting_period.quarter"] = quarter
    _del_count = collection.delete_many(_del_filt).deleted_count
    if _del_count:
        logger.info(
            "upsert_concept_values: deleted %d stale doc(s) for CIK %s FY%d %s",
            _del_count, cik, fiscal_year,
            f"Q{quarter}" if period_type == "quarterly" else "(annual)",
        )

    from earnings_agents.hooks import report_call
    period_label = f"FY{fiscal_year} Q{quarter}" if period_type == "quarterly" else f"FY{fiscal_year}"
    report_call(f"  [db]  upsert {len(ops)} concept(s) → {collection_name}  {period_label}")
    collection.bulk_write(ops, ordered=False)
    report_call(f"  [db]  ✓ upserted {len(ops)} concept(s)")
    if period_type == "quarterly":
        logger.info(
            "upsert_concept_values: %d concept(s) → %s  CIK %s FY%d Q%d",
            len(ops), collection_name, cik, fiscal_year, quarter,
        )
    else:
        logger.info(
            "upsert_concept_values: %d concept(s) → %s  CIK %s FY%d",
            len(ops), collection_name, cik, fiscal_year,
        )
    return len(ops)
