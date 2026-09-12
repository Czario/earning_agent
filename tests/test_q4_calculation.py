"""Tests for the post-save Q4 derivation (income statement only).

Covers the ported calculations-project logic (Q4 = Annual − (Q1+Q2+Q3),
point-in-time copy, annual→quarterly concept matching) plus the node guards
and graph routing.  Uses a tiny in-memory Mongo fake (same spirit as
``test_atomic_save.py``) so no live DB is needed.
"""
from __future__ import annotations

import re
import unittest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from bson import ObjectId

from earnings_agents import config as _config
from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.integrations.q4 import (
    _is_point_in_time_concept,
    calculate_q4_for_period,
)
from earnings_agents.nodes.q4 import calculate_q4_node

_MISSING = object()


def _get_path(doc, path):
    cur = doc
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return _MISSING
        cur = cur[p]
    return cur


def _eq(a, b):
    if a is _MISSING:
        return b is _MISSING
    if b is _MISSING:
        return False
    if isinstance(a, ObjectId) or isinstance(b, ObjectId):
        return str(a) == str(b)
    return a == b


def _matches(doc, filt):
    """Subset matcher supporting dot paths, $in, $ne, $regex (enough for the
    queries the Q4 engine issues)."""
    for k, v in filt.items():
        actual = _get_path(doc, k)
        if isinstance(v, dict) and any(
            op in v for op in ("$in", "$ne", "$regex", "$not")
        ):
            if "$in" in v and not any(_eq(actual, x) for x in v["$in"]):
                return False
            if "$ne" in v and _eq(actual, v["$ne"]):
                return False
            if "$regex" in v:
                if actual is _MISSING or not re.search(v["$regex"], str(actual)):
                    return False
            if "$not" in v and _eq(actual, v["$not"]):
                return False
        else:
            if not _eq(actual, v):
                return False
    return True


class _FakeCollection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.inserted = []
        self.upserted = []  # ("updated"|"upserted", doc)

    def _all(self):
        return self.docs + self.inserted

    def find_one(self, filt):
        for d in self._all():
            if _matches(d, filt):
                return d
        return None

    def find(self, filt):
        return [d for d in self._all() if _matches(d, filt)]

    def insert_one(self, doc):
        self.inserted.append(doc)
        return MagicMock(inserted_id=doc.get("_id"))

    def update_one(self, filt, update, upsert=False):
        for i, d in enumerate(self._all()):
            if _matches(d, filt):
                d.update(update["$set"])
                self.upserted.append(("updated", d))
                return MagicMock(matched_count=1)
        if upsert:
            new_doc = dict(update["$set"])
            self.upserted.append(("upserted", new_doc))
            return MagicMock(matched_count=0)
        return MagicMock(matched_count=0)


class _FakeDb:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = _FakeCollection()
        return self.collections[name]


class _FakeClient:
    def __init__(self, db):
        self.db = db

    def __getitem__(self, name):
        assert name == "normalize_data"
        return self.db


def _dt(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


def _period():
    return DetectedPeriod(
        period_type="annual",
        period_end=date(2026, 12, 31),
        quarter=None,
        fiscal_year=2026,
        period_label="Fiscal Year Ended December 31, 2026",
    )


def _concept(oid, concept, label, path, cik="000123", dimensions=None):
    doc = {
        "_id": ObjectId(oid),
        "cik": cik,
        "statement_type": "income",
        "concept": concept,
        "label": label,
        "path": path,
        "dimension": False,
        "dimension_concept": False,
    }
    if dimensions:
        doc["dimensions"] = dimensions
    return doc


def _value(concept_id, cik, fy, quarter, value, end, **extra):
    rp = {
        "end_date": _dt(*end),
        "period_date": f"{end[0]:04d}-{end[1]:02d}-{end[2]:02d}",
        "fiscal_year": fy,
    }
    if quarter is not None:
        rp["quarter"] = quarter
    doc = {
        "concept_id": ObjectId(concept_id),
        "cik": cik,
        "statement_type": "income",
        "form_type": "10-Q" if quarter else "10-K",
        "reporting_period": rp,
        "value": value,
        "earning_data": True,
        "dimension_value": False,
        "calculated": False,
        "currency": "USD",
    }
    doc.update(extra)
    return doc


class Q4EngineTestBase(unittest.TestCase):
    def setUp(self):
        self.db = _FakeDb()
        self.client = _FakeClient(self.db)
        patcher = patch("earnings_agents.integrations.q4._get_client")
        self.mock_get_client = patcher.start()
        self.mock_get_client.return_value = self.client
        self.addCleanup(patcher.stop)

    def _seed(self, annual_concepts, quarterly_concepts, annual_values, quarterly_values):
        self.db["normalized_concepts_annual"].docs = list(annual_concepts)
        self.db["normalized_concepts_quarterly"].docs = list(quarterly_concepts)
        self.db["concept_values_annual"].docs = list(annual_values)
        self.db["concept_values_quarterly"].docs = list(quarterly_values)


class TestQ4FlowMath(Q4EngineTestBase):
    def test_q4_equals_annual_minus_q1_q2_q3(self):
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[_concept(a_oid, "us-gaap:Revenues", "Revenue", "001")],
            quarterly_concepts=[_concept(q_oid, "us-gaap:Revenues", "Revenue", "001")],
            annual_values=[_value(a_oid, "000123", 2026, None, 500.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 100.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 110.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 120.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["calculated"], 1)
        self.assertEqual(result["point_in_time"], 0)
        self.assertEqual(result["skipped"], 0)
        inserted = self.db["concept_values_quarterly"].inserted[0]
        self.assertEqual(inserted["value"], 170.0)  # 500 - (100+110+120)
        self.assertEqual(inserted["concept_id"], ObjectId(q_oid))
        self.assertTrue(inserted["calculated"])
        self.assertEqual(inserted["form_type"], "10-Q")
        self.assertEqual(inserted["currency"], "USD")
        rp = inserted["reporting_period"]
        self.assertEqual(rp["quarter"], 4)
        self.assertEqual(rp["fiscal_year"], 2026)
        self.assertEqual(rp["end_date"], _dt(2026, 12, 31))  # annual end
        self.assertEqual(rp["start_date"], _dt(2026, 10, 1))  # Q3 end + 1 day
        self.assertEqual(
            rp["note"], "Q4 calculated from annual 10-K minus Q1-Q3"
        )


class TestPointInTime(Q4EngineTestBase):
    def test_point_in_time_copies_annual_value(self):
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self.assertTrue(
            _is_point_in_time_concept("us-gaap:CashAndCashEquivalentsAtCarryingValue")
        )
        self._seed(
            annual_concepts=[_concept(a_oid, "us-gaap:CashAndCashEquivalents", "Cash", "001")],
            quarterly_concepts=[_concept(q_oid, "us-gaap:CashAndCashEquivalents", "Cash", "001")],
            annual_values=[_value(a_oid, "000123", 2026, None, 100.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 10.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 20.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 30.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["point_in_time"], 1)
        inserted = self.db["concept_values_quarterly"].inserted[0]
        self.assertEqual(inserted["value"], 100.0)  # NOT 100-(10+20+30)=40
        self.assertEqual(
            inserted["reporting_period"]["note"],
            "Q4 = annual value (point-in-time concept)",
        )


class TestMissingValuesPolicy(Q4EngineTestBase):
    def _seed_missing_q3(self):
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[_concept(a_oid, "us-gaap:Revenues", "Revenue", "001")],
            quarterly_concepts=[_concept(q_oid, "us-gaap:Revenues", "Revenue", "001")],
            annual_values=[_value(a_oid, "000123", 2026, None, 500.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 100.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 110.0, (2026, 6, 30)),
            ],
        )
        return a_oid

    def test_strict_mode_skips_when_quarter_missing(self):
        a_oid = self._seed_missing_q3()
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["calculated"], 0)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["skipped_reasons"].get("missing Q3"), 1)
        self.assertEqual(len(self.db["concept_values_quarterly"].inserted), 0)

    def test_allow_incomplete_treats_missing_as_zero(self):
        a_oid = self._seed_missing_q3()
        result = calculate_q4_for_period(
            "000123", _period(), [a_oid], allow_incomplete=True
        )
        self.assertEqual(result["calculated"], 1)
        inserted = self.db["concept_values_quarterly"].inserted[0]
        self.assertEqual(inserted["value"], 290.0)  # 500 - (100+110+0)

    def test_skips_when_no_annual_value(self):
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[_concept(a_oid, "us-gaap:Revenues", "Revenue", "001")],
            quarterly_concepts=[_concept(q_oid, "us-gaap:Revenues", "Revenue", "001")],
            annual_values=[],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 100.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 110.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 120.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["skipped_reasons"].get("no annual value"), 1)


class TestConceptMatching(Q4EngineTestBase):
    def test_matches_by_exact_concept_name(self):
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[_concept(a_oid, "us-gaap:Revenues", "Revenue", "001")],
            quarterly_concepts=[_concept(q_oid, "us-gaap:Revenues", "Revenue", "001")],
            annual_values=[_value(a_oid, "000123", 2026, None, 500.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 100.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 110.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 120.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["calculated"], 1)
        self.assertEqual(
            self.db["concept_values_quarterly"].inserted[0]["concept_id"],
            ObjectId(q_oid),
        )

    def test_duplicate_names_pick_path_proximity(self):
        # Both quarterly rows are named us-gaap:Revenues; the one whose path
        # shares the most leading segments with the annual concept wins.
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_other = "bbbbbbbbbbbbbbbbbbbbbbbb"
        q_close = "cccccccccccccccccccccccc"
        self._seed(
            annual_concepts=[_concept(a_oid, "us-gaap:Revenues", "Revenue", "007")],
            quarterly_concepts=[
                _concept(q_other, "us-gaap:Revenues", "Revenue", "001"),
                _concept(q_close, "us-gaap:Revenues", "Revenue", "007"),
            ],
            annual_values=[_value(a_oid, "000123", 2026, None, 500.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_close, "000123", 2026, 1, 100.0, (2026, 3, 31)),
                _value(q_close, "000123", 2026, 2, 110.0, (2026, 6, 30)),
                _value(q_close, "000123", 2026, 3, 120.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["calculated"], 1)
        self.assertEqual(
            self.db["concept_values_quarterly"].inserted[0]["concept_id"],
            ObjectId(q_close),
        )

    def test_dimensional_label_path_prefix_match(self):
        # Annual names segments us-gaap:OperatingSegmentsMember, quarterly uses
        # a company-specific member — matched by label + path prefix.
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[
                _concept(a_oid, "us-gaap:OperatingSegmentsMember", "Americas", "001.003.001")
            ],
            quarterly_concepts=[
                _concept(q_oid, "aapl:AmericasSegmentMember", "Americas", "001.003.002")
            ],
            annual_values=[_value(a_oid, "000123", 2026, None, 200.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 50.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 60.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 70.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["calculated"], 1)
        self.assertEqual(
            self.db["concept_values_quarterly"].inserted[0]["concept_id"],
            ObjectId(q_oid),
        )

    def test_dimensional_label_member_suffix_cleaned_both_sides(self):
        # Raw quarterly labels carry an XBRL member suffix after blank lines;
        # the base label must still match the annual row's clean label.
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[
                _concept(a_oid, "us-gaap:Revenues", "Net sales", "001"),
            ],
            quarterly_concepts=[
                _concept(
                    q_oid, "us-gaap:Revenues",
                    "Net sales\\n\\n\\nus-gaap:ProductMember", "001",
                ),
            ],
            annual_values=[_value(a_oid, "000123", 2026, None, 300.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 100.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 90.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 80.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["calculated"], 1)
        self.assertEqual(
            self.db["concept_values_quarterly"].inserted[0]["concept_id"],
            ObjectId(q_oid),
        )

    def test_dimensional_label_matches_case_insensitively(self):
        # Observed live: annual "Interest Expense" vs quarterly
        # "Interest expense" — same row at the same path, different casing.
        # A case-only difference must still match (otherwise the Q4 is
        # skipped even though Q1–Q3 data exists).
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[
                _concept(a_oid, "us-gaap:InterestExpense", "Interest Expense", "006"),
            ],
            quarterly_concepts=[
                _concept(
                    q_oid, "us-gaap:InterestExpenseNonoperating",
                    "Interest expense", "006",
                ),
            ],
            annual_values=[_value(a_oid, "000123", 2026, None, 90.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 86.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 85.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 82.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["calculated"], 1)
        self.assertEqual(
            self.db["concept_values_quarterly"].inserted[0]["concept_id"],
            ObjectId(q_oid),
        )
        self.assertEqual(self.db["concept_values_quarterly"].inserted[0]["value"], -163.0)

    def test_dimension_member_fallback(self):
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[
                _concept(
                    a_oid, "us-gaap:Revenues", "Net sales", "001",
                    dimensions={"explicitMember": "us-gaap:ProductMember"},
                )
            ],
            quarterly_concepts=[
                _concept(
                    q_oid, "custom:ProductRevenue", "Product revenue", "001",
                    dimensions={"explicitMember": "us-gaap:ProductMember"},
                )
            ],
            annual_values=[_value(a_oid, "000123", 2026, None, 300.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 100.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 90.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 80.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["calculated"], 1)
        self.assertEqual(
            self.db["concept_values_quarterly"].inserted[0]["concept_id"],
            ObjectId(q_oid),
        )

    def test_no_matching_quarterly_concept_skips(self):
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        self._seed(
            annual_concepts=[_concept(a_oid, "us-gaap:Revenues", "Revenue", "001")],
            quarterly_concepts=[],
            annual_values=[_value(a_oid, "000123", 2026, None, 500.0, (2026, 12, 31))],
            quarterly_values=[],
        )
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["skipped_reasons"].get("no matching quarterly concept"), 1)


class TestIdempotency(Q4EngineTestBase):
    def _seed_with_q4(self, existing_q4_value=999.0):
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[_concept(a_oid, "us-gaap:Revenues", "Revenue", "001")],
            quarterly_concepts=[_concept(q_oid, "us-gaap:Revenues", "Revenue", "001")],
            annual_values=[_value(a_oid, "000123", 2026, None, 500.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 100.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 110.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 120.0, (2026, 9, 30)),
                _value(q_oid, "000123", 2026, 4, existing_q4_value, (2026, 12, 31)),
            ],
        )
        return a_oid, q_oid

    def test_existing_q4_is_skipped(self):
        a_oid, q_oid = self._seed_with_q4()
        result = calculate_q4_for_period("000123", _period(), [a_oid])
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["skipped_reasons"].get("Q4 already exists"), 1)
        self.assertEqual(len(self.db["concept_values_quarterly"].inserted), 0)
        self.assertEqual(
            self.db["concept_values_quarterly"].docs[-1]["value"], 999.0
        )

    def test_recalculate_upserts_replacing_stale_q4(self):
        a_oid, _ = self._seed_with_q4()
        result = calculate_q4_for_period(
            "000123", _period(), [a_oid], recalculate=True
        )
        self.assertEqual(result["calculated"], 1)
        # The stale Q4 was replaced in place (write-first upsert), no insert.
        self.assertEqual(len(self.db["concept_values_quarterly"].inserted), 0)
        updated = [d for kind, d in self.db["concept_values_quarterly"].upserted]
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["value"], 170.0)

    def test_recalculate_upserts_when_no_q4_exists(self):
        a_oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        q_oid = "bbbbbbbbbbbbbbbbbbbbbbbb"
        self._seed(
            annual_concepts=[_concept(a_oid, "us-gaap:Revenues", "Revenue", "001")],
            quarterly_concepts=[_concept(q_oid, "us-gaap:Revenues", "Revenue", "001")],
            annual_values=[_value(a_oid, "000123", 2026, None, 500.0, (2026, 12, 31))],
            quarterly_values=[
                _value(q_oid, "000123", 2026, 1, 100.0, (2026, 3, 31)),
                _value(q_oid, "000123", 2026, 2, 110.0, (2026, 6, 30)),
                _value(q_oid, "000123", 2026, 3, 120.0, (2026, 9, 30)),
            ],
        )
        result = calculate_q4_for_period(
            "000123", _period(), [a_oid], recalculate=True
        )
        self.assertEqual(result["calculated"], 1)
        upserted = [d for kind, d in self.db["concept_values_quarterly"].upserted]
        self.assertEqual(upserted[0]["value"], 170.0)


class TestNodeGuards(unittest.TestCase):
    def setUp(self):
        self._orig_calc = _config.CALCULATE_Q4_AFTER_ANNUAL
        self._orig_incomplete = _config.Q4_ALLOW_INCOMPLETE

    def tearDown(self):
        _config.CALCULATE_Q4_AFTER_ANNUAL = self._orig_calc
        _config.Q4_ALLOW_INCOMPLETE = self._orig_incomplete

    def _state(self, **overrides):
        state = {
            "ticker": "TEST",
            "company_name": "Test Co",
            "status": "saved",
            "cik": "000123",
            "concept_metrics": {"aaaaaaaaaaaaaaaaaaaaaaaa": 500.0},
            "detected_period": {
                "period_type": "annual",
                "period_end": "2026-12-31",
                "quarter": None,
                "fiscal_year": 2026,
                "period_label": "Fiscal Year Ended December 31, 2026",
            },
        }
        state.update(overrides)
        return state

    def test_disabled_config_skips(self):
        _config.CALCULATE_Q4_AFTER_ANNUAL = False
        with patch(
            "earnings_agents.integrations.q4.calculate_q4_for_period"
        ) as mock_calc:
            out = calculate_q4_node(self._state())
        mock_calc.assert_not_called()
        self.assertEqual(out["status"], "saved")
        self.assertNotIn("q4_calculation", out)

    def test_quarterly_period_skips(self):
        state = self._state()
        state["detected_period"] = {
            "period_type": "quarterly",
            "period_end": "2026-06-30",
            "quarter": 2,
            "fiscal_year": 2026,
            "period_label": "Three Months Ended June 30, 2026",
        }
        with patch(
            "earnings_agents.integrations.q4.calculate_q4_for_period"
        ) as mock_calc:
            out = calculate_q4_node(state)
        mock_calc.assert_not_called()
        self.assertEqual(out["status"], "saved")

    def test_not_saved_skips(self):
        with patch(
            "earnings_agents.integrations.q4.calculate_q4_for_period"
        ) as mock_calc:
            out = calculate_q4_node(self._state(status="failed"))
        mock_calc.assert_not_called()

    def test_no_cik_skips(self):
        with patch(
            "earnings_agents.integrations.q4.calculate_q4_for_period"
        ) as mock_calc:
            out = calculate_q4_node(self._state(cik=None))
        mock_calc.assert_not_called()

    def test_annual_saved_runs_and_records_summary(self):
        summary = {
            "status": "completed",
            "statement_type": "income",
            "fiscal_year": 2026,
            "recalculated": False,
            "processed": 2,
            "calculated": 2,
            "point_in_time": 0,
            "skipped": 1,
            "skipped_reasons": {"no matching quarterly concept": 1},
            "errors": [],
        }
        with patch(
            "earnings_agents.integrations.q4.calculate_q4_for_period",
            return_value=summary,
        ) as mock_calc:
            out = calculate_q4_node(self._state())
        mock_calc.assert_called_once()
        _, kwargs = mock_calc.call_args
        self.assertEqual(kwargs["statement_type"], "income")
        self.assertEqual(kwargs["cik"], "000123")
        self.assertEqual(out["q4_calculation"], summary)
        self.assertEqual(out["status"], "saved")

    def test_replace_sets_recalculate(self):
        summary = {
            "status": "completed", "statement_type": "income", "fiscal_year": 2026,
            "recalculated": True, "processed": 1, "calculated": 1,
            "point_in_time": 0, "skipped": 0, "skipped_reasons": {}, "errors": [],
        }
        with patch(
            "earnings_agents.integrations.q4.calculate_q4_for_period",
            return_value=summary,
        ) as mock_calc:
            out = calculate_q4_node(self._state(_pending_replace={"cik": "000123"}))
        self.assertTrue(mock_calc.call_args.kwargs["recalculate"])
        self.assertEqual(out["status"], "saved")

    def test_engine_exception_never_fails_run(self):
        with patch(
            "earnings_agents.integrations.q4.calculate_q4_for_period",
            side_effect=RuntimeError("mongo down"),
        ):
            out = calculate_q4_node(self._state())
        self.assertEqual(out["status"], "saved")
        self.assertEqual(out["q4_calculation"]["status"], "error")


class TestGraphRouting(unittest.TestCase):
    def test_route_after_save(self):
        from earnings_agents.graph import _route_after_save
        # Post-save: guidance persistence runs BEFORE the Q4 derivation.
        self.assertEqual(_route_after_save({"status": "saved"}), "save_guidance")
        self.assertEqual(_route_after_save({"status": "failed"}), "__end__")
        self.assertEqual(_route_after_save({"status": "skipped"}), "__end__")

    def test_graph_has_calculate_q4_node(self):
        from earnings_agents.graph import build_graph
        compiled = build_graph()
        nodes = set(compiled.get_graph().nodes.keys())
        self.assertIn("calculate_q4", nodes)
        self.assertIn("save_guidance", nodes)
        self.assertIn("mongodb_save", nodes)

    def test_graph_edges_save_guidance_before_calculate_q4(self):
        from earnings_agents.graph import build_graph
        compiled = build_graph()
        g = compiled.get_graph()
        # mongodb_save conditionally routes to save_guidance; save_guidance
        # has a straight edge into calculate_q4.
        edges = [(e.source, e.target) for e in g.edges]
        self.assertIn(("save_guidance", "calculate_q4"), edges)
        self.assertIn(("calculate_q4", "__end__"), edges)


if __name__ == "__main__":
    unittest.main()
