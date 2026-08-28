"""Q4 derivation for annual filings — income statement only.

After an ANNUAL filing is saved to ``concept_values_annual`` (Q4 == annual in
this pipeline, so the quarterly collection has no Q4 for that fiscal year),
derive Q4 quarterly values from the saved annual values:

    Q4 = Annual − (Q1 + Q2 + Q3)

This is a faithful port of the ``calculations`` project's
``services/q4_calculation_service.py`` + ``repositories/financial_repository.py``
(``--calculate-q4 --statement is`` scope: income statement only), adapted to
this pipeline's schema and invariants:

  • **Flow concepts** (revenue, expenses): Q4 = Annual − (Q1 + Q2 + Q3).
    In strict mode (default) a missing Q1/Q2/Q3 or annual value SKIPS the
    concept — a fabricated Q4 must never be stored.  With
    ``Q4_ALLOW_INCOMPLETE=1`` missing values are treated as 0 (the source
    project's behavior).
  • **Point-in-time concepts** (cash balances, shares outstanding, period
    markers): Q4 = Annual (copied verbatim, never differenced).
  • **Concept matching** (annual → quarterly): same concept name with path
    proximity scoring for duplicate names; label + path / label + path-prefix
    for dimensional/segment rows whose names differ between collections
    (e.g. quarterly ``aapl:AmericasSegmentMember`` vs annual
    ``us-gaap:OperatingSegmentsMember``); XBRL dimension-member identity as
    the final fallback.
  • **Idempotent**: an existing Q4 for (concept_id, fiscal_year) is skipped
    unless the annual save REPLACED an existing period (``recalculate``), in
    which case the Q4 is upserted — write-first, never delete-before-write.
  • **Never fails a run**: every concept is processed independently; failures
    land in the returned summary (observability), never in the run status.
"""
from __future__ import annotations

import atexit
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from bson import ObjectId
from pymongo import MongoClient

from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.config import MONGODB_URI
from earnings_agents.integrations.normalize import _clean_label

logger = logging.getLogger(__name__)

_NORMALIZE_DB = "normalize_data"
_client: Optional[MongoClient] = None  # type: ignore[type-arg]


def _get_client() -> MongoClient:  # type: ignore[type-arg]
    """Shared MongoClient (same pattern as integrations/normalize.py)."""
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
        atexit.register(lambda: _client.close() if _client else None)  # type: ignore[union-attr]
    return _client


# Point-in-time concept patterns — ported verbatim from the calculations
# project (services/q4_calculation_service.py).  These concepts are snapshots
# at a specific date (ending balances, weighted-average shares, period
# markers) rather than flows over a period, so the Q4 = Annual − (Q1+Q2+Q3)
# differencing does NOT apply: Q4 = Annual (copied).
_POINT_IN_TIME_PATTERNS = [
    # Cash and equivalents (balance sheet items - ending balances)
    "CashAndCashEquivalents",
    "CashCashEquivalents",
    "RestrictedCash",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    # Period markers
    "EndOfYear",
    "EndOfPeriod",
    "EndOfTheYear",
    "EndOfThePeriod",
    "BeginningOfYear",
    "BeginningOfPeriod",
    "BeginningOfTheYear",
    "BeginningOfThePeriod",
    "AtEndOf",
    "AtBeginningOf",
    # Shares outstanding (point-in-time, not cumulative - these are AVERAGES not sums)
    "SharesOutstanding",
    "CommonStockSharesOutstanding",
    "StockSharesOutstanding",
    "WeightedAverageNumberOfShares",
    "WeightedAverageNumberOfDilutedShares",
    "WeightedAverageNumberOfSharesOutstanding",
    # Exchange rate effects and increases/decreases (these are reconciliation items)
    "PeriodIncreaseDecrease",
    "EffectOfExchangeRate",
    "EffectOfExchange",
    # Ending/beginning balances
    "EndingBalance",
    "BeginningBalance",
    "ClosingBalance",
    "OpeningBalance",
]


def _is_point_in_time_concept(concept_name: str, label: str = "") -> bool:
    """True when *concept_name*/*label* matches a point-in-time pattern.

    Point-in-time concepts represent snapshots at specific dates (like cash
    balances) rather than flows over a period, so Q4 = Annual − (Q1+Q2+Q3)
    doesn't apply — Q4 = Annual.
    """
    concept_lower = concept_name.lower()
    label_lower = label.lower()
    for pattern in _POINT_IN_TIME_PATTERNS:
        pattern_lower = pattern.lower()
        if pattern_lower in concept_lower or pattern_lower in label_lower:
            return True
    return False


def _path_proximity_score(a_path: str, b_path: str) -> int:
    """Number of leading equal path segments between two hierarchy paths.

    Ported from the calculations project's path proximity scoring: quarterly
    ``007.001`` vs annual ``007.003`` → score 1 (same first segment); exact
    match → 2.  Used to disambiguate multiple concepts sharing one name.
    """
    a = a_path.split(".") if a_path else []
    b = b_path.split(".") if b_path else []
    score = 0
    for x, y in zip(a, b):
        if x == y:
            score += 1
        else:
            break
    return score


def _find_quarterly_concept(
    db: Any,
    annual_concept: dict[str, Any],
    cik: str,
    statement_type: str,
) -> dict[str, Any] | None:
    """Find the quarterly concept doc matching an *annual_concept* doc.

    Ported from the calculations project's ``_find_matching_annual_concept``
    / ``_find_quarterly_concept`` (direction reversed: annual → quarterly),
    with the same fallback ladder:

    1. Same ``concept`` name (+ cik + statement_type).  When multiple rows
       share the name, the row whose path shares the most leading segments
       with the annual concept's path wins (path proximity scoring).
    2. Dimensional/segment rows whose names differ between collections
       (e.g. quarterly ``aapl:AmericasSegmentMember`` vs annual
       ``us-gaap:OperatingSegmentsMember``): exact path + cleaned label,
       then label + same first path segment (never crossing into a different
       top-level hierarchy group).
    3. XBRL dimension-member identity (``dimensions.explicitMember``).

    Returns ``None`` when no trustworthy match exists — a wrong match would
    produce a Q4 value under the wrong concept, which must never happen.
    """
    concept = annual_concept.get("concept") or ""
    base: dict[str, Any] = {"cik": cik, "statement_type": statement_type}
    q_col = db["normalized_concepts_quarterly"]

    # ── 1. Same concept name ──────────────────────────────────────────────
    if concept:
        matches = list(q_col.find({**base, "concept": concept}))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            annual_path = annual_concept.get("path") or ""
            scored = [
                (m, _path_proximity_score(annual_path, m.get("path") or ""))
                for m in matches
            ]
            best_score = max(score for _, score in scored)
            if best_score > 0:
                closest = [m for m, s in scored if s == best_score]
                if len(closest) == 1:
                    return closest[0]
                matches = closest

    # ── 2. Dimensional/segment rows: label + path matching ────────────────
    annual_label = _clean_label(annual_concept.get("label") or "")[0]
    annual_path = annual_concept.get("path") or ""
    if annual_label:
        # Raw labels can carry an XBRL member suffix (e.g.
        # "Net sales\n\n\nus-gaap:ProductMember") — clean BOTH sides before
        # comparing, so a quarterly row at the same path with a member-qualified
        # label still matches the annual row's base label.  Comparison is
        # case-insensitive (observed live: annual "Interest Expense" vs
        # quarterly "Interest expense" — same row, different casing).
        def _label_matches(candidate: dict[str, Any]) -> bool:
            return (
                _clean_label(candidate.get("label") or "")[0].lower()
                == annual_label.lower()
            )

        if annual_path:
            by_path = list(q_col.find({**base, "path": annual_path}))
            for candidate in by_path:
                if _label_matches(candidate):
                    return candidate
            first = annual_path.split(".")[0]
            if first:
                by_prefix = list(
                    q_col.find(
                        {**base, "path": {"$regex": f"^{re.escape(first)}\\."}}
                    )
                )
                for candidate in by_prefix:
                    if _label_matches(candidate):
                        return candidate

    # ── 3. XBRL dimension-member identity ─────────────────────────────────
    dims = annual_concept.get("dimensions") or {}
    member = dims.get("explicitMember") or annual_concept.get("dimension_member")
    if member:
        candidates = list(
            q_col.find({**base, "dimensions.explicitMember": member})
        )
        if not candidates:
            candidates = list(q_col.find({**base, "dimension_member": member}))
        if candidates:
            return candidates[0]

    return None


def _load_q_values(
    db: Any,
    q_oid: ObjectId,
    cik: str,
    fiscal_year: int,
    statement_type: str,
) -> dict[int, dict[str, Any]]:
    """Load the quarterly Q1/Q2/Q3 value docs for a concept, keyed by quarter."""
    rows = list(
        db["concept_values_quarterly"].find(
            {
                "concept_id": q_oid,
                "cik": cik,
                "statement_type": statement_type,
                "reporting_period.fiscal_year": fiscal_year,
                "reporting_period.quarter": {"$in": [1, 2, 3]},
            }
        )
    )
    by_q: dict[int, dict[str, Any]] = {}
    for r in rows:
        q = (r.get("reporting_period") or {}).get("quarter")
        if q in (1, 2, 3):
            by_q[q] = r
    return by_q


def _q4_exists(
    db: Any,
    q_oid: ObjectId,
    cik: str,
    fiscal_year: int,
    statement_type: str,
) -> bool:
    """True when a Q4 (quarter=4) value already exists for this concept."""
    return (
        db["concept_values_quarterly"].find_one(
            {
                "concept_id": q_oid,
                "cik": cik,
                "statement_type": statement_type,
                "reporting_period.fiscal_year": fiscal_year,
                "reporting_period.quarter": 4,
            }
        )
        is not None
    )


def _build_q4_doc(
    *,
    cik: str,
    q_oid: ObjectId,
    statement_type: str,
    fiscal_year: int,
    q4_value: float,
    annual_value_row: dict[str, Any],
    q_rows: dict[int, dict[str, Any]],
    is_point_in_time: bool,
    fallback_end: Any,
) -> dict[str, Any]:
    """Build the Q4 concept-value doc matching the pipeline's save schema.

    Mirrors the document shape written by ``upsert_concept_values`` (same
    ``reporting_period`` layout, top-level dimension metadata, currency,
    ``calculated: True``) so downstream consumers can't tell a derived Q4
    from an extracted one except via ``calculated``/``note``.
    """
    rp = annual_value_row.get("reporting_period") or {}
    end = rp.get("end_date")
    if not isinstance(end, datetime):
        end = datetime(
            fallback_end.year, fallback_end.month, fallback_end.day,
            tzinfo=timezone.utc,
        )
    period_date = rp.get("period_date") or end.strftime("%Y-%m-%d")
    now = datetime.now(tz=timezone.utc)

    # Q4 starts the day after Q3 ends (when Q3 is available).
    start_date: datetime | None = None
    q3 = q_rows.get(3)
    if q3 is not None:
        q3_end = (q3.get("reporting_period") or {}).get("end_date")
        if isinstance(q3_end, datetime):
            start_date = q3_end + timedelta(days=1)

    period_doc: dict[str, Any] = {
        "end_date": end,
        "period_date": period_date,
        "fiscal_year": fiscal_year,
        "quarter": 4,
        "note": (
            "Q4 = annual value (point-in-time concept)"
            if is_point_in_time
            else "Q4 calculated from annual 10-K minus Q1-Q3"
        ),
    }
    if start_date is not None:
        period_doc["start_date"] = start_date

    doc: dict[str, Any] = {
        "concept_id": q_oid,
        "cik": cik,
        "statement_type": statement_type,
        "form_type": "10-Q",
        "reporting_period": period_doc,
        "value": q4_value,
        "earning_data": True,
        "created_at": now,
        "dimension_value": bool(annual_value_row.get("dimension_value")),
        "calculated": True,
        "currency": annual_value_row.get("currency") or "USD",
    }
    if annual_value_row.get("accession_number"):
        doc["accession_number"] = annual_value_row["accession_number"]
    # Dimension metadata is copied from the annual value record (same filing).
    for key in ("dimension_member", "dimension_member_label", "dimension_axis"):
        if annual_value_row.get(key):
            doc[key] = annual_value_row[key]
    return doc


def _bump_skip(summary: dict[str, Any], reason: str) -> None:
    reasons = summary["skipped_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def calculate_q4_for_period(
    cik: str,
    period: DetectedPeriod,
    annual_concept_ids: list[str],
    *,
    statement_type: str = "income",
    allow_incomplete: bool = False,
    recalculate: bool = False,
) -> dict[str, Any]:
    """Derive Q4 (quarterly) values from the just-saved annual filing values.

    For every annual concept id in *annual_concept_ids* (the concepts the
    pipeline just saved to ``concept_values_annual``):

    * find the matching quarterly concept (``_find_quarterly_concept``);
    * load Q1/Q2/Q3 from ``concept_values_quarterly`` and the annual value
      from ``concept_values_annual``;
    * point-in-time → Q4 = Annual; flow → Q4 = Annual − (Q1+Q2+Q3);
    * insert into ``concept_values_quarterly`` (skip if Q4 already exists,
      unless *recalculate* — then upsert, write-first).

    *allow_incomplete* (``Q4_ALLOW_INCOMPLETE``) adopts the calculations
    project's treat-missing-as-0 behavior; the default strict mode skips
    concepts with missing inputs so a fabricated Q4 is never stored.

    Never raises for per-concept failures — they are collected in the
    returned summary (observability only).
    """
    db = _get_client()[_NORMALIZE_DB]
    fiscal_year = period.fiscal_year
    a_col = db["normalized_concepts_annual"]
    av_col = db["concept_values_annual"]
    qv_col = db["concept_values_quarterly"]

    summary: dict[str, Any] = {
        "status": "completed",
        "statement_type": statement_type,
        "fiscal_year": fiscal_year,
        "recalculated": recalculate,
        "processed": 0,
        "calculated": 0,      # flow concepts — Q4 = Annual − (Q1+Q2+Q3)
        "point_in_time": 0,   # point-in-time concepts — Q4 = Annual
        "skipped": 0,
        "skipped_reasons": {},
        "errors": [],
    }

    for cid_str in annual_concept_ids:
        try:
            annual_oid = ObjectId(cid_str)
        except Exception:  # noqa: BLE001 — malformed id, skip
            summary["skipped"] += 1
            _bump_skip(summary, "invalid concept id")
            continue

        annual_concept = a_col.find_one({"_id": annual_oid})
        if not annual_concept:
            summary["skipped"] += 1
            _bump_skip(summary, "annual concept not found")
            continue

        q_concept = _find_quarterly_concept(db, annual_concept, cik, statement_type)
        if not q_concept:
            summary["skipped"] += 1
            _bump_skip(summary, "no matching quarterly concept")
            continue
        q_oid = q_concept["_id"]

        annual_value_row = av_col.find_one(
            {
                "concept_id": annual_oid,
                "cik": cik,
                "statement_type": statement_type,
                "reporting_period.fiscal_year": fiscal_year,
            }
        )
        if not annual_value_row:
            summary["skipped"] += 1
            _bump_skip(summary, "no annual value")
            continue

        if not recalculate and _q4_exists(db, q_oid, cik, fiscal_year, statement_type):
            summary["skipped"] += 1
            _bump_skip(summary, "Q4 already exists")
            continue

        q_rows = _load_q_values(db, q_oid, cik, fiscal_year, statement_type)
        annual_value = annual_value_row["value"]
        concept_name = annual_concept.get("concept") or ""
        label = _clean_label(annual_concept.get("label") or "")[0]
        is_point_in_time = _is_point_in_time_concept(concept_name, label)

        if is_point_in_time:
            q4_value = annual_value
        else:
            q1 = q_rows.get(1)
            q2 = q_rows.get(2)
            q3 = q_rows.get(3)
            if not allow_incomplete:
                missing = [
                    f"Q{q}"
                    for q, row in ((1, q1), (2, q2), (3, q3))
                    if row is None
                ]
                if missing:
                    summary["skipped"] += 1
                    _bump_skip(summary, f"missing {'/'.join(missing)}")
                    continue
            q4_value = annual_value - (
                (q1["value"] if q1 else 0.0)
                + (q2["value"] if q2 else 0.0)
                + (q3["value"] if q3 else 0.0)
            )

        try:
            doc = _build_q4_doc(
                cik=cik,
                q_oid=q_oid,
                statement_type=statement_type,
                fiscal_year=fiscal_year,
                q4_value=q4_value,
                annual_value_row=annual_value_row,
                q_rows=q_rows,
                is_point_in_time=is_point_in_time,
                fallback_end=period.period_end,
            )
            if recalculate:
                # Write-first upsert on the exact unique-index key
                # (cik, concept_id, fiscal_year, quarter) — replaces a stale
                # Q4 in place, no delete-before-write window.
                qv_col.update_one(
                    {
                        "cik": cik,
                        "concept_id": q_oid,
                        "reporting_period.fiscal_year": fiscal_year,
                        "reporting_period.quarter": 4,
                    },
                    {"$set": doc},
                    upsert=True,
                )
            else:
                qv_col.insert_one(doc)
        except Exception as exc:  # noqa: BLE001 — per-concept, never fails the run
            summary["errors"].append(
                f"{concept_name} FY{fiscal_year}: {exc}"
            )
            continue

        summary["processed"] += 1
        if is_point_in_time:
            summary["point_in_time"] += 1
        else:
            summary["calculated"] += 1

    logger.info(
        "calculate_q4_for_period: CIK %s FY%d (%s) — %d flow + %d point-in-time "
        "Q4 value(s), %d skipped, %d error(s)",
        cik, fiscal_year, statement_type,
        summary["calculated"], summary["point_in_time"],
        summary["skipped"], len(summary["errors"]),
    )
    return summary
