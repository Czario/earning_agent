"""Tests for Phase 4: Multi-Statement Post-Save Q4 Derivations & Semantics."""
from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from bson import ObjectId

from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.integrations.q4 import (
    _is_point_in_time_concept,
    calculate_q4_for_period,
)
from earnings_agents.nodes.q4 import calculate_q4_node
from earnings_agents.state import EarningsAgentState


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
    for k, v in filt.items():
        actual = _get_path(doc, k)
        if isinstance(v, dict) and any(
            op in v for op in ("$in", "$ne", "$regex", "$not")
        ):
            if "$in" in v and not any(_eq(actual, x) for x in v["$in"]):
                return False
            if "$ne" in v and _eq(actual, v["$ne"]):
                return False
        else:
            if not _eq(actual, v):
                return False
    return True


class _FakeCollection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.inserted = []
        self.upserted = []

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
        for d in self._all():
            match = True
            for k, v in filt.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                d.update(update.get("$set", {}))
                self.upserted.append(("updated", d))
                return MagicMock(matched_count=1)
        if upsert:
            new_doc = dict(update.get("$set", {}))
            self.upserted.append(("upserted", new_doc))
            self.inserted.append(new_doc)
            return MagicMock(matched_count=0, upserted_id=ObjectId())
        return MagicMock(matched_count=0)


class TestPhase4PointInTimeSemantics(unittest.TestCase):
    def test_balancesheet_is_always_point_in_time(self):
        # Even concepts named "Revenues" or "Cash" are point-in-time if statement_type is balancesheet
        self.assertTrue(
            _is_point_in_time_concept("TotalAssets", "Total Assets", statement_type="balancesheet")
        )
        self.assertTrue(
            _is_point_in_time_concept("AccountsPayable", "Accounts Payable", statement_type="balancesheet")
        )
        self.assertTrue(
            _is_point_in_time_concept("RetainedEarnings", "Retained Earnings", statement_type="balancesheet")
        )

    def test_cashflow_distinguishes_ending_balances_from_flow(self):
        # Ending cash balance is point-in-time
        self.assertTrue(
            _is_point_in_time_concept(
                "CashAndCashEquivalentsAtCarryingValue",
                "Cash and cash equivalents at end of period",
                statement_type="cashflow",
            )
        )
        # Periodic operating cash flow is flow/duration
        self.assertFalse(
            _is_point_in_time_concept(
                "NetCashProvidedByUsedInOperatingActivities",
                "Net cash provided by operating activities",
                statement_type="cashflow",
            )
        )

    def test_income_statement_flow_vs_averages(self):
        # Revenue is flow
        self.assertFalse(
            _is_point_in_time_concept("Revenues", "Total Revenues", statement_type="income")
        )
        # Weighted average shares is point-in-time
        self.assertTrue(
            _is_point_in_time_concept(
                "WeightedAverageNumberOfDilutedSharesOutstanding",
                "Diluted shares",
                statement_type="income",
            )
        )


class TestPhase4CalculateQ4MultiStatement(unittest.TestCase):
    def setUp(self):
        self.cik = "000123"
        self.fy = 2026
        self.period = DetectedPeriod(
            period_type="annual",
            period_end=date(2026, 12, 31),
            quarter=None,
            fiscal_year=self.fy,
            period_label="Fiscal Year Ended December 31, 2026",
        )

        self.cid_is_a = ObjectId("5072e8e0c9d3f4a1b2c3d4e1")
        self.cid_is_q = ObjectId("5072e8e0c9d3f4a1b2c3d4e2")

        self.cid_bs_a = ObjectId("5072e8e0c9d3f4a1b2c3d4e3")
        self.cid_bs_q = ObjectId("5072e8e0c9d3f4a1b2c3d4e4")

        self.cid_cf_a = ObjectId("5072e8e0c9d3f4a1b2c3d4e5")
        self.cid_cf_q = ObjectId("5072e8e0c9d3f4a1b2c3d4e6")

        self.a_concepts = [
            {"_id": self.cid_is_a, "concept": "us-gaap:Revenues", "label": "Revenues", "statement_type": "income", "path": "001"},
            {"_id": self.cid_bs_a, "concept": "us-gaap:Assets", "label": "Total Assets", "statement_type": "balancesheet", "path": "001"},
            {"_id": self.cid_cf_a, "concept": "us-gaap:CashAndCashEquivalents", "label": "Cash at End of Period", "statement_type": "cashflow", "path": "001"},
        ]
        self.q_concepts = [
            {"_id": self.cid_is_q, "concept": "us-gaap:Revenues", "label": "Revenues", "statement_type": "income", "path": "001", "cik": self.cik},
            {"_id": self.cid_bs_q, "concept": "us-gaap:Assets", "label": "Total Assets", "statement_type": "balancesheet", "path": "001", "cik": self.cik},
            {"_id": self.cid_cf_q, "concept": "us-gaap:CashAndCashEquivalents", "label": "Cash at End of Period", "statement_type": "cashflow", "path": "001", "cik": self.cik},
        ]

        # Annual saved values
        self.annual_values = [
            {"concept_id": self.cid_is_a, "cik": self.cik, "statement_type": "income", "value": 1000.0, "reporting_period": {"fiscal_year": self.fy}},
            {"concept_id": self.cid_bs_a, "cik": self.cik, "statement_type": "balancesheet", "value": 5000.0, "reporting_period": {"fiscal_year": self.fy}},
            {"concept_id": self.cid_cf_a, "cik": self.cik, "statement_type": "cashflow", "value": 1500.0, "reporting_period": {"fiscal_year": self.fy}},
        ]

        # Quarterly values (only for Income Statement flow differencing: Q1=200, Q2=250, Q3=300 -> Q4 expected 250)
        self.quarterly_values = [
            {"concept_id": self.cid_is_q, "cik": self.cik, "statement_type": "income", "value": 200.0, "reporting_period": {"fiscal_year": self.fy, "quarter": 1}},
            {"concept_id": self.cid_is_q, "cik": self.cik, "statement_type": "income", "value": 250.0, "reporting_period": {"fiscal_year": self.fy, "quarter": 2}},
            {"concept_id": self.cid_is_q, "cik": self.cik, "statement_type": "income", "value": 300.0, "reporting_period": {"fiscal_year": self.fy, "quarter": 3}},
            # Notice: No Q1-Q3 values for Balance Sheet or Ending Cash, because they are point-in-time!
        ]

        self.db = {
            "normalized_concepts_annual": _FakeCollection(self.a_concepts),
            "normalized_concepts_quarterly": _FakeCollection(self.q_concepts),
            "concept_values_annual": _FakeCollection(self.annual_values),
            "concept_values_quarterly": _FakeCollection(self.quarterly_values),
        }

    def test_multi_statement_q4_calculation(self):
        with patch("earnings_agents.integrations.q4._get_client") as mock_client:
            mock_client.return_value.__getitem__.side_effect = lambda k: self.db if k == "normalize_data" else None

            summary = calculate_q4_for_period(
                cik=self.cik,
                period=self.period,
                annual_concept_ids=[str(self.cid_is_a), str(self.cid_bs_a), str(self.cid_cf_a)],
                statement_type=None,
                target_statements=["income", "balancesheet", "cashflow"],
            )
            self.assertEqual(summary["processed"], 3)
            # 1 flow concept (Revenues: 1000 - (200+250+300) = 250)
            self.assertEqual(summary["calculated"], 1)
            # 2 point-in-time concepts (Assets: 5000, Cash End: 1500)
            self.assertEqual(summary["point_in_time"], 2)

            q_col = self.db["concept_values_quarterly"]
            inserted = q_col.inserted

            # Check Revenues Q4
            rev_doc = next(d for d in inserted if d["concept_id"] == self.cid_is_q)
            self.assertEqual(rev_doc["value"], 250.0)
            self.assertEqual(rev_doc["statement_type"], "income")
            self.assertIn("Q4 calculated from annual", rev_doc["reporting_period"]["note"])

            # Check Assets Q4
            assets_doc = next(d for d in inserted if d["concept_id"] == self.cid_bs_q)
            self.assertEqual(assets_doc["value"], 5000.0)
            self.assertEqual(assets_doc["statement_type"], "balancesheet")
            self.assertIn("point-in-time", assets_doc["reporting_period"]["note"])

            # Check Cash End Q4
            cash_doc = next(d for d in inserted if d["concept_id"] == self.cid_cf_q)
            self.assertEqual(cash_doc["value"], 1500.0)
            self.assertEqual(cash_doc["statement_type"], "cashflow")
            self.assertIn("point-in-time", cash_doc["reporting_period"]["note"])


if __name__ == "__main__":
    unittest.main()
