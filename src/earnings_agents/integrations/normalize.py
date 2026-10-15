"""Client for reading from and writing to the normalize_data MongoDB database.

Used when EARNINGS_SAVE_TARGET=normalize_data.  The module keeps a single
shared MongoClient instance (same pattern as mongodb_client.py) and registers
an atexit handler for clean shutdown.
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from bson import ObjectId
from pymongo import MongoClient, UpdateOne

from earnings_agents.agent.currency import is_currency_code
from earnings_agents.agent.period import DetectedPeriod, format_period_label
from earnings_agents.config import MONGODB_URI, NORMALIZE_DB_NAME

logger = logging.getLogger(__name__)

_NORMALIZE_DB = NORMALIZE_DB_NAME
_client: Optional[MongoClient] = None  # type: ignore[type-arg]


def _get_client() -> MongoClient:  # type: ignore[type-arg]
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
        atexit.register(lambda: _client.close() if _client else None)  # type: ignore[union-attr]
    return _client


# ── Company lookup ───────────────────────────────────────────────────────────

def get_company_by_ticker(ticker: str) -> dict[str, Any] | None:
    """Return ``{cik, name, fiscal_year_end_month, industry}`` for *ticker*.

    Queries normalize_data.companies by ``ticker_symbol`` (case-insensitive).
    ``fiscal_year_end_month`` is derived from the ``corporate_info.fiscal_year_end``
    field, which is stored as an "MMDD" string (e.g. "0630" for June 30).
    ``industry`` is the raw ``{sic_code, sic_description}`` subdocument (``{}``
    when absent) — advisory context for the extraction agent.
    """
    db = _get_client()[_NORMALIZE_DB]
    doc = db["companies"].find_one(
        {"ticker_symbol": ticker.upper()},
        {"cik": 1, "name": 1, "corporate_info.fiscal_year_end": 1, "industry": 1},
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
        "industry": doc.get("industry") or {},
    }


# ── Collection routing — single source of truth ─────────────────────────────

# SEC form identifiers we accept as provenance: "8-K", "6-K", "10-Q", "20-F",
# "8-K/A", "40-F", "S-1", ...  Deliberately permissive (any short dashed
# alphanumeric token) but strict enough to reject junk/labels.
# Fields an older version of this writer stored but the current contract does
# not: they must never linger on a value document.  `save_token` was the
# per-save marker (now replaced by a concept-id-scoped stale sweep),
# `source_lines` is evidence kept in pipeline state only, and
# `source_form_type` is folded into `form_type` (the real filed SEC form), and
# `extraction_status` was per-row bookkeeping nothing reads.
_RETIRED_VALUE_FIELDS = (
    "save_token", "source_lines", "source_form_type", "extraction_status",
)

_SEC_FORM_RX = re.compile(r"^[A-Z0-9]{1,5}(?:-[A-Z0-9]{1,3})?(?:/A)?$")


def normalize_form_type(form_type: str | None) -> str | None:
    """Return a normalized SEC form identifier, or None when not usable.

    Accepts the real filed forms (``8-K``, ``6-K``, ``10-Q``, ``20-F``,
    ``8-K/A``, ...) so a value records the form it actually came from.  The
    synthetic period-family labels (``10-Q``/``10-K``) are also valid inputs,
    which is what callers fall back to when provenance is unknown.
    """
    code = str(form_type or "").strip().upper()
    # A real SEC form identifier always carries a number ("8-K", "10-Q",
    # "20-F", "S-1"); this rejects sentinels like "MIXED"/"UNKNOWN".
    if not code or not _SEC_FORM_RX.match(code) or not any(c.isdigit() for c in code):
        return None
    return code


def _concepts_collection(period: DetectedPeriod) -> str:
    """Return the normalized-concepts collection for the agent period."""
    if period.period_type == "annual":
        return "normalized_concepts_annual"
    if period.period_type == "quarterly":
        return "normalized_concepts_quarterly"
    raise ValueError(f"Unsupported period_type: {period.period_type!r}")


def _values_collection(period: DetectedPeriod) -> str:
    """Return the concept-values collection for the agent period."""
    if period.period_type == "annual":
        return "concept_values_annual"
    if period.period_type == "quarterly":
        return "concept_values_quarterly"
    raise ValueError(f"Unsupported period_type: {period.period_type!r}")


def bootstrap_quarterly_concepts_from_annual(cik: str) -> int:
    """Seed an empty company's quarterly concept collection from annual rows.

    Company-level and all-or-nothing: when the company has ANY row in
    ``normalized_concepts_quarterly`` the quarterly universe is already in use
    and nothing is copied.  Only when that collection is empty for the company
    are all annual concept rows cloned — WITHOUT their annual ``_id``, so
    MongoDB assigns a fresh, unique ``_id`` in the quarterly collection.  The
    two hierarchies must never share document identities.  Only ``form_type``
    is changed (to the quarterly ``10-Q``).  No annual VALUES are copied.

    Intra-hierarchy references (``parent_concept_id`` and the dimensional
    ``concept_id``) are re-pointed at the NEW quarterly rows so a copied child
    never links back into ``normalized_concepts_annual``.  A reference that did
    not resolve to a copied annual row is left as-is.

    No bookkeeping collection is used: the quarterly concept collection itself
    is the state.  The company-level emptiness check makes a retry a no-op, so
    an already-bootstrapped company is never copied twice.

    Returns the number of annual concept rows copied (zero when no bootstrap
    was needed or no annual rows exist).
    """
    db = _get_client()[_NORMALIZE_DB]
    quarterly = db["normalized_concepts_quarterly"]
    annual = db["normalized_concepts_annual"]

    if quarterly.find_one({"cik": cik}, {"_id": 1}) is not None:
        return 0

    annual_docs = list(annual.find({"cik": cik}))
    if not annual_docs:
        logger.info(
            "quarterly concept bootstrap: no annual concept rows for CIK %s", cik
        )
        return 0

    # Clone every annual row WITHOUT its `_id`: MongoDB assigns a new unique
    # quarterly `_id`, so annual and quarterly never share document ids.
    source_ids: list[Any] = []
    docs_to_insert: list[dict[str, Any]] = []
    for source_doc in annual_docs:
        quarterly_doc = dict(source_doc)
        source_id = quarterly_doc.pop("_id", None)
        if source_id is None:
            raise ValueError(
                f"Annual concept row for CIK {cik} has no _id; cannot clone safely"
            )
        quarterly_doc["form_type"] = "10-Q"
        source_ids.append(source_id)
        docs_to_insert.append(quarterly_doc)

    insert_result = quarterly.insert_many(docs_to_insert, ordered=False)
    new_ids = list(insert_result.inserted_ids)

    # Re-link intra-hierarchy references to the NEW quarterly identities so a
    # copied child/parent/dimensional link never points back at annual rows.
    annual_to_quarterly_id = dict(zip(source_ids, new_ids))
    ref_updates: list[UpdateOne] = []
    for new_id, quarterly_doc in zip(new_ids, docs_to_insert):
        set_fields: dict[str, Any] = {}
        for ref_field in ("parent_concept_id", "concept_id"):
            referenced = quarterly_doc.get(ref_field)
            if referenced is None:
                continue
            remapped = annual_to_quarterly_id.get(referenced)
            if remapped is not None:
                set_fields[ref_field] = remapped
        if set_fields:
            ref_updates.append(UpdateOne({"_id": new_id}, {"$set": set_fields}))
    if ref_updates:
        quarterly.bulk_write(ref_updates, ordered=False)

    from earnings_agents.hooks import report_call
    report_call(
        f"  [db]  bootstrapped {len(new_ids)} quarterly concept(s) from annual "
        f"concepts for CIK {cik}"
    )
    logger.info(
        "quarterly concept bootstrap: copied %d annual concept row(s) for CIK %s",
        len(new_ids), cik,
    )
    return len(new_ids)


def _concept_order_rank(value: str) -> int:
    """Convert a Gotham-style alpha order key (a..z, aa...) to a rank."""
    text = str(value or "").strip().lower()
    if not text or any(ch < "a" or ch > "z" for ch in text):
        return 0
    rank = 0
    for ch in text:
        rank = rank * 26 + (ord(ch) - ord("a") + 1)
    return rank


def _concept_order_key(rank: int) -> str:
    """Convert a positive sibling rank to a stable alpha order key."""
    rank = max(1, int(rank))
    chars: list[str] = []
    while rank:
        rank, remainder = divmod(rank - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return "".join(reversed(chars))


def _new_concept_name(label: str, statement_type: str, parent: str | None,
                      dimension_concept: bool, dimension_axis: str) -> str:
    """Build a stable internal custom key; the filing label remains the display label."""
    words = re.findall(r"[A-Za-z0-9]+", label)
    slug = "".join(word[:1].upper() + word[1:] for word in words)
    if not slug or not slug[0].isalpha():
        raise ValueError(f"cannot create a stable custom concept key from label {label!r}")
    identity = "|".join((
        statement_type.lower(), label.casefold(), parent or "ROOT",
        "dimension" if dimension_concept else "line_item", dimension_axis.casefold(),
    ))
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
    member_suffix = "Member" if dimension_concept else ""
    return f"custom:{slug}{member_suffix}_{suffix}"


def persist_new_concept_candidates(
    cik: str,
    period: DetectedPeriod,
    candidates: list[dict],
) -> tuple[dict[str, dict], list[str]]:
    """Create/reuse validated company concepts for unmapped filing values.

    Concept names are stable company-specific ``custom:`` keys generated from
    the source label and parent identity. Parent placement is agent-selected;
    this function only materializes the next available child/root path and
    order key, mirroring Gotham's agent-proposed hierarchy + deterministic
    materialization split. New concepts are never created for candidates with
    an unresolved parent path.

    Returns ``(by_metric_key, issues)``. Individual concept failures are
    non-fatal so valid previously mapped metrics can still be saved.
    """
    if not candidates:
        return {}, []
    collection_name = _concepts_collection(period)
    collection = _get_client()[_NORMALIZE_DB][collection_name]
    by_metric_key: dict[str, dict] = {}
    issues: list[str] = []

    for candidate in candidates:
        metric_key = str(candidate.get("metric_key") or "").strip()
        label = str(candidate.get("label") or "").strip()
        statement_type = str(candidate.get("statement_type") or "").strip().lower()
        parent_concept = candidate.get("parent_concept")
        parent_id_raw = candidate.get("parent_concept_id")
        dimension_concept = bool(candidate.get("dimension_concept"))
        dimension_axis = str(candidate.get("dimension_axis") or "").strip()
        if not metric_key or not label or not statement_type:
            issues.append("new concept skipped: missing metric key, label, or statement")
            continue

        try:
            parent_id = ObjectId(str(parent_id_raw)) if parent_id_raw else None
        except Exception:  # noqa: BLE001
            issues.append(f"{label}: parent concept ID is invalid")
            continue

        parent_doc: dict | None = None
        if parent_id is not None:
            parent_doc = collection.find_one({
                "_id": parent_id,
                "cik": cik,
                "statement_type": statement_type,
            })
            if not parent_doc or not parent_doc.get("path"):
                issues.append(f"{label}: parent concept is not stored in the target period hierarchy")
                continue
        elif dimension_concept:
            issues.append(f"{label}: dimensional concept has no parent")
            continue

        parent_name = str(parent_doc.get("concept")) if parent_doc else None
        try:
            concept_name = _new_concept_name(
                label, statement_type, parent_name, dimension_concept, dimension_axis,
            )
        except ValueError as exc:
            issues.append(f"{label}: {exc}")
            continue

        identity_query: dict[str, Any] = {
            "cik": cik,
            "statement_type": statement_type,
            "concept": concept_name,
            "dimension_concept": dimension_concept,
            "parent_concept": parent_name,
        }
        existing = collection.find_one(identity_query)
        if existing:
            by_metric_key[metric_key] = {
                "concept_id": str(existing["_id"]),
                "concept": existing,
                "created": False,
            }
            continue

        # Read all rows in this company+statement so new paths are allocated
        # against the complete stored hierarchy, not just the prompt's recent
        # concept subset.
        rows = list(collection.find(
            {"cik": cik, "statement_type": statement_type},
            {"path": 1, "order_key": 1, "parent_concept": 1,
             "parent_concept_id": 1, "_id": 1},
        ))
        parent_path = str(parent_doc.get("path")) if parent_doc else None
        if parent_path:
            direct_prefix = parent_path + "."
            parent_depth = len(parent_path.split("."))
            siblings = [
                row for row in rows
                if isinstance(row.get("path"), str)
                and row["path"].startswith(direct_prefix)
                and len(row["path"].split(".")) == parent_depth + 1
            ]
        else:
            siblings = [
                row for row in rows
                if isinstance(row.get("path"), str)
                and len(row["path"].split(".")) == 1
            ]

        numeric_slots: list[int] = []
        for sibling in siblings:
            try:
                slot = int(str(sibling["path"]).split(".")[-1])
            except (KeyError, ValueError):
                continue
            if slot not in (404, 555):
                numeric_slots.append(slot)
        next_slot = max(numeric_slots, default=0) + 1
        segment = f"{next_slot:03d}"
        path = f"{parent_path}.{segment}" if parent_path else segment
        next_order = max((_concept_order_rank(row.get("order_key")) for row in siblings), default=0) + 1
        order_key = _concept_order_key(next_order)

        identity = "|".join((cik, statement_type, concept_name, parent_name or "ROOT"))
        concept_id = ObjectId(hashlib.sha256(identity.encode("utf-8")).digest()[:12])
        doc: dict[str, Any] = {
            "_id": concept_id,
            "cik": cik,
            "statement_type": statement_type,
            "concept": concept_name,
            "form_type": "10-K" if period.period_type == "annual" else "10-Q",
            "label": label,
            "canonical_concept": concept_name,
            "path": path,
            "order_key": order_key,
            "abstract": False,
            "hide": False,
            "dimension": False,
            "dimension_concept": dimension_concept,
        }
        if parent_doc:
            doc.update({
                "parent_concept": parent_name,
                "parent_concept_id": parent_id,
                "parent_path": parent_path,
            })
        if dimension_concept:
            doc["concept_id"] = parent_id
            doc["dimension_member"] = concept_name
            doc["dimension_member_label"] = str(
                candidate.get("dimension_member_label") or label
            ).strip()
            if dimension_axis:
                doc["dimension_axis"] = dimension_axis

        try:
            # A deterministic ID plus $setOnInsert makes retries/concurrent
            # reprocessing idempotent and never rewrites an existing concept.
            insert_doc = {k: v for k, v in doc.items() if k != "_id"}
            collection.update_one(
                {"_id": concept_id},
                {"$setOnInsert": insert_doc},
                upsert=True,
            )
            persisted = collection.find_one({"_id": concept_id, "cik": cik})
            if not persisted:
                issues.append(f"{label}: concept row was not materialized")
                continue
            by_metric_key[metric_key] = {
                "concept_id": str(concept_id),
                "concept": persisted,
                "created": True,
            }
            rows.append({"_id": concept_id, "path": path, "order_key": order_key})
            siblings.append({"_id": concept_id, "path": path, "order_key": order_key})
        except Exception as exc:  # noqa: BLE001 — one new row must not lose saved rows
            logger.warning("new concept persistence failed for %s: %s", label, exc)
            issues.append(f"{label}: concept persistence failed ({exc})")

    return by_metric_key, issues


# ── Concept lookup ───────────────────────────────────────────────────────────

_CONCEPT_PREFIX_RX = re.compile(r"(?:us-gaap|system|ifrs-full|dei|srt):", re.IGNORECASE)
_MEMBER_RX = re.compile(
    r"(?:us-gaap|system|ifrs-full|dei|srt):([A-Za-z0-9_]+)", re.IGNORECASE
)
_FULL_MEMBER_RX = re.compile(
    r"((?:us-gaap|ifrs-full|dei|srt):[A-Za-z0-9_]+Member)", re.IGNORECASE
)
_CAMEL_SPLIT_RX = re.compile(r"(?<!^)(?=[A-Z])")

# Page-number pollution in the hierarchy path (observed live: a ``custom:404``
# row on PDD, OCI rows at ``555`` on INTU): a path segment that is a bare page
# or HTTP status number (404/555) is not a real income-statement row.
# Such rows are excluded at fetch time (and re-checked in the load loop) so
# they never reach the extraction target.
_POLLUTED_PATH_SEGMENTS = ("404", "555")
_POLLUTED_PATH_RX = re.compile(r"(^|\.)(404|555)(\.|$)")


def _is_polluted_path(path: str) -> bool:
    """True when *path* carries a page-number segment (``404``/``555``).

    Upstream normalizer pollution: page numbers leak into the row's hierarchy
    path (observed live on PDD).  No real income-statement row has a path
    segment that is a bare page/status number, so such rows are skipped at
    fetch time and never enter the extraction target.
    """
    if not path:
        return False
    return any(seg in _POLLUTED_PATH_SEGMENTS for seg in path.split("."))


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
    *,
    period: DetectedPeriod,
    concept_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return sorted concept dicts for *cik* and *statement_types*.

    *period* is the canonical period-agent result and selects the source
    collection from ``period.period_type``:
      - ``"quarterly"`` → ``normalized_concepts_quarterly``
      - ``"annual"``    → ``normalized_concepts_annual``

    *concept_ids* (optional) restricts the load to exactly those concept
    ``_id`` strings — the recent-value window filter is applied HERE, at query
    time, so non-recent concepts are never even loaded.  EXCEPTION: system:
    (``concept`` matching ``^system:``) and ``calculated`` concepts are ALWAYS
    loaded regardless of the window — they are the CALC derivation targets,
    and excluding them would make them permanently underivable.

    Filters out abstract (``abstract: true``) and hidden (``hide: true``) rows,
    and rows whose ``path`` carries a page-number segment (``404``/``555`` —
    upstream normalizer pollution).  All other rows — including calculated/
    system concepts, dimensional breakdown rows, and XBRL structural labels —
    are included.

    Results are sorted by ``path`` then ``order_key`` so the prompt lists
    concepts in statement order deterministically, including same-path rows.

    Each returned dict has keys: ``_id`` (str), ``concept`` (GAAP name),
    ``label`` (cleaned, disambiguated only when needed), ``path``,
    ``order_key``, ``statement_type``, ``taxonomy_key`` (stable XBRL identity
    used as the JSON key in the extraction prompt and as the mapping key back
    to ``concept_id``), plus ``dimension`` / ``dimension_concept`` (bool) as
    the authoritative dimensional-row signal.  ``(path, order_key)`` is the
    row identity; path alone is not unique in normalized XBRL data.

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
    collection_name = _concepts_collection(period)
    db = _get_client()[_NORMALIZE_DB]
    from earnings_agents.hooks import report_call as _report_call
    _report_call(f"  [db]  query {collection_name}  concepts for CIK {cik}")

    query: dict[str, Any] = {
        "cik": cik,
        "statement_type": {"$in": statement_types},
        "active": {"$ne": False},
        # Never fetch page-number path rows (404/555 pollution) — excluded at
        # query time so they never enter the target list.
        "path": {"$not": _POLLUTED_PATH_RX},
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
            "dimension_member": 1,
            "dimension_member_label": 1,
            "dimension_axis": 1,
            "calculated": 1,
        },
    # path alone is not unique (sibling rows can share a path, disambiguated
    # by order_key) — sort by both so row order is deterministic across runs.
    ).sort([("path", 1), ("order_key", 1)])

    # First pass: collect rows with cleaned labels.
    parsed: list[tuple[dict[str, Any], str, str, str, str]] = []  # (doc, head, member, member_tag, path)
    base_counts: dict[tuple[str, str], int] = {}
    base_key_counts: dict[tuple[str, str], int] = {}
    stmt_types_per_key: dict[str, set[str]] = {}
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
        member_tag = d.get("dimension_member") or _extract_member_tag(raw_label)
        member = d.get("dimension_member_label") or member
        path = d.get("path", "") or ""
        if _is_polluted_path(path):
            logger.info(
                "get_statement_concepts: skipping page-number path row "
                "(cik=%s concept=%s path=%r)",
                cik, concept, path,
            )
            continue
        st = d.get("statement_type", "")
        parsed.append((d, head, member, member_tag, path))
        key = (st, head.lower())
        base_counts[key] = base_counts.get(key, 0) + 1
        label_groups.setdefault(key, []).append(path)
        tkey = (
            f"{concept}|{member_tag}"
            if member_tag and member_tag != concept else concept
        ).lower()
        base_key_counts[(st, tkey)] = base_key_counts.get((st, tkey), 0) + 1
        stmt_types_per_key.setdefault(tkey, set()).add(st)
        # The first sorted row at a path is used only as the fallback parent
        # label for label disambiguation.  The hierarchy builder retains every
        # (path, order_key) row; this map must not be used for hierarchy edges.
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
        # (e.g. us-gaap:ServiceMember under both Revenue and Cost of Revenue,
        # or us-gaap:NetIncomeLoss across income and cashflow).
        taxonomy_key = (
            f"{concept}|{member_tag}"
            if member_tag and member_tag != concept else concept
        )
        tkey = taxonomy_key.lower()
        has_cross_stmt = len(stmt_types_per_key.get(tkey, set())) > 1
        has_intra_stmt = base_key_counts.get((st, tkey), 0) > 1

        if has_cross_stmt:
            taxonomy_key = f"{taxonomy_key}|{st}"
        if has_intra_stmt and path:
            taxonomy_key = f"{taxonomy_key}|{path}"

        # Drop only exact duplicates (same concept + path in the same statement).
        order_key = d.get("order_key")
        order_key_identity = "" if order_key is None else str(order_key)
        dedup_key = (st, concept.lower(), path, order_key_identity)
        if dedup_key in seen_final:
            logger.debug(
                "get_statement_concepts: dropping exact duplicate concept/path/order "
                "%r (cik=%s concept=%s path=%r order_key=%r)",
                final_label, cik, concept, path, order_key,
            )
            continue
        seen_final.add(dedup_key)

        out.append(
            {
                "_id": str(d["_id"]),
                "concept": concept,
                "label": final_label,
                "path": path,
                "order_key": order_key,
                "statement_type": st,
                "taxonomy_key": taxonomy_key,
                "dimension": bool(d.get("dimension")),
                "dimension_concept": bool(d.get("dimension_concept")),
                "calculated": bool(d.get("calculated")),
                "dimension_member": member_tag or "",
                "dimension_member_label": member or "",
                "dimension_axis": d.get("dimension_axis") or member_tag or "",
            }
        )
    return out


def get_calculated_concepts(
    cik: str,
    statement_types: list[str] | None = None,
    *,
    period: DetectedPeriod,
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
    collection_name = _concepts_collection(period)
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
            "path": {"$not": _POLLUTED_PATH_RX},
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
            "order_key": 1,
            "statement_type": 1,
            "dimension": 1,
            "dimension_concept": 1,
            "calculated": 1,
        },
    ).sort([("path", 1), ("order_key", 1)])

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
        if _is_polluted_path(d.get("path", "") or ""):
            logger.info(
                "get_calculated_concepts: skipping page-number path row "
                "(cik=%s concept=%s path=%r)",
                cik, d.get("concept", ""), d.get("path", ""),
            )
            continue
        out.append(
            {
                "_id": str(d["_id"]),
                "concept": d.get("concept", ""),
                "label": head,
                "path": d.get("path", ""),
                "order_key": d.get("order_key"),
                "statement_type": d.get("statement_type", ""),
                "dimension": bool(d.get("dimension")),
                "dimension_concept": bool(d.get("dimension_concept")),
                "calculated": bool(d.get("calculated")),
            }
        )

    logger.debug(
        "get_calculated_concepts: found %d calculated concept(s) for cik=%s (%s)",
        len(out), cik, period.period_type,
    )
    return out


# ── Period helpers ───────────────────────────────────────────────────────────

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


def parse_period_start_date(period: DetectedPeriod) -> date | None:
    """Return the first calendar day of the canonical quarterly period.

    Uses the cumulative duration encoded in the agent's period label to count
    backwards from the agent's period end:

    * Week-based: ``end_date - (weeks * 7) + 1 day``
      e.g. "Thirteen Weeks Ended May 2, 2026" → Feb 1, 2026
    * Month-based: first day of the month that is *n* months before the
      end month (inclusive of the period).
      e.g. "Six Months Ended June 30, 2026" → Jan 1, 2026

    Returns ``None`` when no duration can be parsed from the period label.
    """
    if period.period_type != "quarterly":
        return None
    duration = _extract_duration(period.period_label or "")
    end_date = period.period_end
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


# ── Exact-period operations ──────────────────────────────────────────────────

def fiscal_period_exists(cik: str, period: DetectedPeriod) -> bool:
    """Return whether the canonical agent period already has values."""
    db = _get_client()[_NORMALIZE_DB]
    filt: dict[str, Any] = {
        "cik": cik,
        "statement_type": "income",
        "reporting_period.fiscal_year": period.fiscal_year,
    }
    if period.quarter is not None:
        filt["reporting_period.quarter"] = period.quarter
    return db[_values_collection(period)].count_documents(
        filt, limit=1
    ) > 0


def get_recently_valued_concept_ids(
    cik: str,
    period: DetectedPeriod,
    n_periods: int = 3,
    statement_types: list[str] | None = None,
) -> set[str]:
    """Return concept_id strings that had a value in the last *n_periods* periods.

    Queries the ``concept_values_{annual|quarterly}`` collection for *cik*,
    finds the *n_periods* most recent distinct ``reporting_period.end_date``
    values, and returns the set of ``concept_id`` values (as strings) that had
    at least one stored value in any of those periods.

    Purpose: the extraction-target PRIORITIZATION signal — NOT an eligibility
    filter.  `load_company_concepts` builds its target as recent concepts ∪
    all dimensional rows ∪ `system:`/`calculated` concepts, so new segments
    and newly disclosed rows stay extractable even without history; this
    function only marks which concepts are known to be recently valued.

    Returns an **empty set** when no history exists (the caller bootstraps
    from the full universe).
    """
    col_name = _values_collection(period)
    db = _get_client()[_NORMALIZE_DB]
    col = db[col_name]
    query: dict[str, Any] = {"cik": cik}
    if statement_types:
        query["statement_type"] = {"$in": list(statement_types)}
    else:
        query["statement_type"] = "income"

    periods = col.distinct(
        "reporting_period.end_date",
        query,
    )
    periods = sorted([p for p in periods if p is not None], reverse=True)[:n_periods]
    if not periods:
        return set()
    id_query = dict(query)
    id_query["reporting_period.end_date"] = {"$in": periods}
    ids = col.distinct(
        "concept_id",
        id_query,
    )
    return {str(i) for i in ids if i is not None}


def upsert_concept_values(
    cik: str,
    company_name: str,
    concept_metrics: dict[str, float],
    period: DetectedPeriod,
    statement_type: str = "income",
    target_statements: list[str] | None = None,
    derived_concept_ids: set[str] | None = None,
    value_metadata_by_id: dict[str, dict] | None = None,
    accession_number: str | None = None,
    source_form_type: str | None = None,
) -> int:
    """Bulk-upsert concept values into the appropriate collection.

    Routes to ``concept_values_quarterly`` or ``concept_values_annual`` using
    the supplied canonical period-agent result.  No duration, filename,
    cadence, EDGAR metadata, or month-based period decision is made here.
    The agent's period label is used only for the stored label and quarterly
    ``start_date`` calculation.

    Documents are written to match the existing schema used by the SEC-based
    pipeline, with ``concept_id`` stored as ``ObjectId`` and ``end_date`` as
    a native ``datetime`` so the upsert filter correctly de-duplicates
    re-runs of the same earnings release.

    **Atomicity**: the period is replaced write-first, clean-stale-after.
    All values are upserted BEFORE any stale document is deleted, so a failed/interrupted write never leaves the
    period empty or partially overwritten — the previous run's documents stay
    in place until the new ones are fully written, and the stale sweep only
    removes documents that do not carry the current save token.

    Returns the number of operations submitted (0 on early-exit failures).
    """
    if not concept_metrics:
        logger.debug("upsert_concept_values: empty concept_metrics — nothing to do")
        return 0

    # The period agent owns every reporting-period dimension.  This function
    # only persists the canonical result; it never parses or infers a period.
    period_type = period.period_type
    collection_name = _values_collection(period)
    # The SOURCE SEC form actually filed (6-K / 8-K / 10-Q / 20-F ...) is the
    # real provenance of a value.  The period-family label (10-Q/10-K) is only
    # a fallback for when the source form could not be determined — it must not
    # be stored as if it were the filed form.
    form_type = normalize_form_type(source_form_type) or (
        "10-K" if period_type == "annual" else "10-Q"
    )
    fiscal_year = period.fiscal_year
    quarter = period.quarter
    end_date = period.period_end
    period_str = period.period_label or ""
    start_date = parse_period_start_date(period) if period_str else None
    # Store end_date as a native UTC datetime to match the existing collection schema.
    end_datetime = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0,
                            tzinfo=timezone.utc)
    period_date_str = end_date.strftime("%Y-%m-%d")
    now = datetime.now(tz=timezone.utc)

    db = _get_client()[_NORMALIZE_DB]
    collection = db[collection_name]
    # Concept ids upserted by THIS call.  The post-write stale sweep deletes
    # only docs whose concept_id is not in here, which gives the same
    # write-first atomic replace as a per-save marker field without storing one.
    written_ids: list[ObjectId] = []

    ops: list[UpdateOne] = []
    for concept_id_str, value in concept_metrics.items():
        try:
            concept_oid = ObjectId(concept_id_str)
        except Exception:  # noqa: BLE001 — invalid ObjectId string, skip
            logger.warning(
                "upsert_concept_values: invalid ObjectId %r — skipping", concept_id_str
            )
            continue

        meta: dict[str, Any] = (value_metadata_by_id or {}).get(concept_id_str) or {}
        # No FX conversion exists anywhere in this pipeline, and USD is never
        # assumed: a value whose currency cannot be confirmed is not written.
        value_currency = str(meta.get("currency") or "").strip().upper()
        if not is_currency_code(value_currency):
            logger.warning(
                "upsert_concept_values: skipping concept %s — currency %r is "
                "not a confirmed ISO code",
                concept_id_str, meta.get("currency"),
            )
            continue
        dimension_value = bool(meta.get("dimension") or meta.get("dimension_concept"))

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
            "statement_type": meta.get("statement_type") or statement_type,
            "form_type": form_type,
            "reporting_period": period_doc,
            "value": value,
            "earning_data": True,
            "created_at": now,
            "dimension_value": dimension_value,
            "calculated": bool(meta.get("calculated")) or concept_id_str in (derived_concept_ids or set()),
            "currency": value_currency,
        }
        if dimension_value:
            if meta.get("dimension_member"):
                doc["dimension_member"] = meta["dimension_member"]
            if meta.get("dimension_member_label"):
                doc["dimension_member_label"] = meta["dimension_member_label"]
            if meta.get("dimension_axis"):
                doc["dimension_axis"] = meta["dimension_axis"]
        if meta.get("original_currency"):
            doc["original_currency"] = meta["original_currency"]
            doc["original_value"] = meta.get("original_value")
            doc["original_scale"] = meta.get("original_scale")
        if meta.get("source_exhibit"):
            doc["source_exhibit"] = meta["source_exhibit"]
        if meta.get("scale"):
            doc["scale"] = str(meta["scale"]).strip().lower()
        if meta.get("source_snippet"):
            doc["source_snippet"] = str(meta["source_snippet"])[:500]
        if accession_number:
            doc["accession_number"] = accession_number
        filter_doc: dict[str, Any] = {
            "cik": cik,
            "concept_id": concept_oid,
            "reporting_period.fiscal_year": fiscal_year,
        }
        if period_type == "quarterly":
            filter_doc["reporting_period.quarter"] = quarter
        written_ids.append(concept_oid)
        # This filter is EXACTLY the collection's unique index key
        # (cik, concept_id, fiscal_year, quarter).  ``concept_id`` is the
        # concept row's unique ObjectId, so ``dimension_member`` /
        # ``dimension_axis`` are descriptive metadata on the stored doc — NOT
        # part of the row identity — and must stay OUT of the filter.  Adding
        # them made the filter stricter than the unique index and asymmetric
        # with the written doc (empty members are omitted from the doc but were
        # written as ``""`` into the filter), producing E11000 duplicate-key
        # collisions on re-runs of the same period.

        # `$set` never removes fields, so explicitly drop retired fields from
        # the document shape instead of leaving them behind on a row that
        # already exists.  Keeps the written contract exact on every run.
        ops.append(
            UpdateOne(
                filter_doc,
                {
                    "$set": doc,
                    "$unset": {field: "" for field in _RETIRED_VALUE_FIELDS},
                },
                upsert=True,
            )
        )

    if not ops:
        return 0

    # ── Write FIRST, then sweep stale ────────────────────────────────────
    # All values for this period are upserted (in place, via the identity
    # filter) BEFORE any deletion.  A failed/interrupted write therefore never
    # leaves the period empty: the previous run's documents remain until the
    # new ones are fully written.  The stale sweep afterwards removes every
    # document of this period that does not carry this save's token (old
    # runs, superseded concept rows, rows removed from the target list).
    from earnings_agents.hooks import report_call
    collection.bulk_write(ops, ordered=False)

    _stale_filt: dict[str, Any] = {
        "cik": cik,
        "reporting_period.fiscal_year": fiscal_year,
        # Everything this save did NOT touch for this period is stale.
        "concept_id": {"$nin": written_ids},
    }
    if target_statements:
        _stale_filt["statement_type"] = {"$in": list(target_statements)}
    elif statement_type:
        _stale_filt["statement_type"] = statement_type

    if period_type == "quarterly":
        _stale_filt["reporting_period.quarter"] = quarter
    _del_count = collection.delete_many(_stale_filt).deleted_count
    if _del_count:
        logger.info(
            "upsert_concept_values: swept %d stale doc(s) for CIK %s %s",
            _del_count, cik, format_period_label(period),
        )

    period_label = format_period_label(period)
    report_call(
        f"  [db]  ✓ upserted {len(ops)} concept(s) → {collection_name}  {period_label}"
    )
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
