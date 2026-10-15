"""Tests for the company-level quarterly concept bootstrap from annual rows."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from bson import ObjectId

from earnings_agents.integrations.normalize import (
    bootstrap_quarterly_concepts_from_annual,
)
from earnings_agents.nodes.concepts import load_company_concepts_node


class _Collection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.find_queries = []
        self.bulk_calls = 0

    def find_one(self, query, projection=None):  # noqa: ARG002
        return next((doc for doc in self.docs if self._matches(doc, query)), None)

    def find(self, query):
        self.find_queries.append(query)
        return [doc for doc in self.docs if self._matches(doc, query)]

    def bulk_write(self, ops, ordered=False):  # noqa: ARG002
        self.bulk_calls += 1
        for op in ops:
            filt = op._filter
            existing = next((doc for doc in self.docs if self._matches(doc, filt)), None)
            if existing is None and op._upsert:
                self.docs.append({**filt, **op._doc.get("$setOnInsert", {})})

    @staticmethod
    def _matches(doc, query):
        return all(doc.get(key) == value for key, value in query.items())


class _Database:
    def __init__(self, collections):
        self.collections = collections

    def __getitem__(self, name):
        return self.collections.setdefault(name, _Collection())


class _Client:
    def __init__(self, database):
        self.database = database

    def __getitem__(self, name):  # noqa: ARG002
        return self.database


def _annual_rows(cik):
    parent_id = ObjectId()
    return [
        {
            "_id": parent_id,
            "cik": cik,
            "statement_type": "income",
            "concept": "us-gaap:Revenues",
            "label": "Revenue",
            "form_type": "20-F",
            "canonical_concept": "us-gaap:Revenues",
            "path": "001",
            "order_key": "a",
            "abstract": False,
            "dimension": False,
            "dimension_concept": False,
        },
        {
            "_id": ObjectId(),
            "cik": cik,
            "statement_type": "income",
            "concept": "us-gaap:CostOfRevenue",
            "label": "Cost of revenue",
            "form_type": "20-F",
            "canonical_concept": "us-gaap:CostOfRevenue",
            "path": "001.001",
            "order_key": "b",
            "abstract": False,
            "dimension": False,
            "dimension_concept": False,
            "parent_concept": "us-gaap:Revenues",
            "parent_concept_id": parent_id,
            "parent_path": "001",
        },
        {
            "_id": ObjectId(),
            "cik": cik,
            "statement_type": "income",
            "concept": "asml:NXEMember",
            "label": "NXE",
            "form_type": "20-F",
            "canonical_concept": "asml:NXEMember",
            "path": "001.001.001",
            "dimension": False,
            "dimension_concept": True,
            "concept_id": parent_id,
            "parent_concept": "us-gaap:Revenues",
        },
        {
            "_id": ObjectId(),
            "cik": cik,
            "statement_type": "balancesheet",
            "concept": "us-gaap:Assets",
            "label": "Assets",
            "form_type": "20-F/A",
            "path": "001",
            "dimension": False,
            "dimension_concept": False,
        },
    ]


class TestQuarterlyConceptBootstrap(unittest.TestCase):
    def _run(self, collections, cik="000123"):
        with patch(
            "earnings_agents.integrations.normalize._get_client",
            return_value=_Client(_Database(collections)),
        ):
            return bootstrap_quarterly_concepts_from_annual(cik)

    def test_empty_quarterly_collection_copies_all_annual_rows_and_only_changes_form_type(self):
        cik = "000123"
        annual_docs = _annual_rows(cik)
        annual = _Collection(annual_docs)
        quarterly = _Collection()

        copied = self._run({
            "normalized_concepts_annual": annual,
            "normalized_concepts_quarterly": quarterly,
        }, cik)

        self.assertEqual(copied, len(annual_docs))
        self.assertEqual(quarterly.bulk_calls, 1)
        self.assertEqual(len(quarterly.docs), len(annual_docs))
        by_id = {doc["_id"]: doc for doc in quarterly.docs}
        for annual_doc in annual_docs:
            quarterly_doc = by_id[annual_doc["_id"]]
            expected = {**annual_doc, "form_type": "10-Q"}
            self.assertEqual(quarterly_doc, expected)
        # References stay valid because IDs are preserved in the same collection copy.
        child = next(d for d in quarterly.docs if d["concept"] == "us-gaap:CostOfRevenue")
        self.assertIn(child["parent_concept_id"], by_id)

    def test_any_existing_quarterly_concept_prevents_company_bootstrap(self):
        cik = "000123"
        # Even a quarterly concept in a different statement suppresses the
        # company-level annual bootstrap; this is intentionally not per-statement.
        quarterly = _Collection([{
            "_id": ObjectId(),
            "cik": cik,
            "statement_type": "balancesheet",
            "concept": "us-gaap:Assets",
        }])
        annual = _Collection(_annual_rows(cik))

        copied = self._run({
            "normalized_concepts_annual": annual,
            "normalized_concepts_quarterly": quarterly,
        }, cik)

        self.assertEqual(copied, 0)
        self.assertEqual(quarterly.bulk_calls, 0)
        self.assertEqual(len(quarterly.docs), 1)
        self.assertEqual(annual.find_queries, [])

    def test_rerun_after_a_copy_is_a_noop(self):
        """No bookkeeping collection: the quarterly collection is the state."""
        cik = "000123"
        annual_docs = _annual_rows(cik)
        collections = {
            "normalized_concepts_annual": _Collection(annual_docs),
            "normalized_concepts_quarterly": _Collection(),
        }
        copied = self._run(collections, cik)
        self.assertEqual(copied, len(annual_docs))

        # Second run sees quarterly rows → skips entirely (no bulk write).
        quarterly = collections["normalized_concepts_quarterly"]
        copied_again = self._run(collections, cik)
        self.assertEqual(copied_again, 0)
        self.assertEqual(quarterly.bulk_calls, 1)

    def test_duplicate_ids_are_never_overwritten(self):
        """$setOnInsert means a re-copy never rewrites an existing row."""
        cik = "000123"
        annual_docs = _annual_rows(cik)
        quarterly = _Collection()
        collections = {
            "normalized_concepts_annual": _Collection(annual_docs),
            "normalized_concepts_quarterly": quarterly,
        }
        self._run(collections, cik)
        # Simulate a later quarterly edit, then force a re-copy by clearing the
        # company's rows except one (which suppresses the bootstrap anyway).
        first = dict(quarterly.docs[0])
        quarterly.docs = quarterly.docs[:1]
        self.assertEqual(self._run(collections, cik), 0)
        self.assertEqual(quarterly.docs[0], first)

    def test_no_annual_rows_leaves_quarterly_untouched(self):
        quarterly = _Collection()
        copied = self._run({
            "normalized_concepts_annual": _Collection(),
            "normalized_concepts_quarterly": quarterly,
        })
        self.assertEqual(copied, 0)
        self.assertEqual(quarterly.bulk_calls, 0)
        self.assertEqual(quarterly.docs, [])

    def test_quarterly_node_bootstraps_before_loading_concepts(self):
        events = []
        concepts = [{
            "_id": "5072e8e0c9d3f4a1b2c3d4e5",
            "concept": "us-gaap:Revenues",
            "taxonomy_key": "us-gaap:Revenues",
            "label": "Revenue",
            "statement_type": "income",
            "dimension": False,
            "dimension_concept": False,
            "calculated": False,
        }]
        state = {
            "ticker": "TEST",
            "target_statements": ["income"],
            "detected_period": {
                "period_type": "quarterly",
                "period_end": "2026-06-30",
                "quarter": 2,
                "fiscal_year": 2026,
                "period_label": "Three Months Ended June 30, 2026",
            },
        }
        with (
            patch(
                "earnings_agents.nodes.concepts.get_company_by_ticker",
                return_value={
                    "cik": "000123",
                    "name": "Test Co",
                    "fiscal_year_end_month": 12,
                    "fiscal_year_end_code": "1231",
                    "industry": {},
                },
            ),
            patch(
                "earnings_agents.nodes.concepts.bootstrap_quarterly_concepts_from_annual",
                side_effect=lambda cik: events.append(("bootstrap", cik)) or 3,
            ),
            patch(
                "earnings_agents.nodes.concepts.get_statement_concepts",
                side_effect=lambda *args, **kwargs: events.append(("load", kwargs["period"].period_type)) or concepts,
            ),
            patch(
                "earnings_agents.nodes.concepts.get_recently_valued_concept_ids",
                return_value=set(),
            ),
        ):
            result = load_company_concepts_node(state)

        self.assertEqual(events, [("bootstrap", "000123"), ("load", "quarterly")])
        self.assertEqual(len(result["target_concepts"]), 1)

    def test_annual_node_does_not_bootstrap_quarterly_concepts(self):
        events = []
        concepts = [{
            "_id": "5072e8e0c9d3f4a1b2c3d4e5",
            "concept": "us-gaap:Revenues",
            "taxonomy_key": "us-gaap:Revenues",
            "label": "Revenue",
            "statement_type": "income",
            "dimension": False,
            "dimension_concept": False,
            "calculated": False,
        }]
        state = {
            "ticker": "TEST",
            "target_statements": ["income"],
            "detected_period": {
                "period_type": "annual",
                "period_end": "2026-12-31",
                "quarter": None,
                "fiscal_year": 2026,
                "period_label": "Year Ended December 31, 2026",
            },
        }
        with (
            patch(
                "earnings_agents.nodes.concepts.get_company_by_ticker",
                return_value={
                    "cik": "000123",
                    "name": "Test Co",
                    "fiscal_year_end_month": 12,
                    "fiscal_year_end_code": "1231",
                    "industry": {},
                },
            ),
            patch(
                "earnings_agents.nodes.concepts.bootstrap_quarterly_concepts_from_annual",
                side_effect=lambda cik: events.append(("bootstrap", cik)),
            ) as bootstrap,
            patch(
                "earnings_agents.nodes.concepts.get_statement_concepts",
                side_effect=lambda *args, **kwargs: concepts,
            ),
            patch(
                "earnings_agents.nodes.concepts.get_recently_valued_concept_ids",
                return_value=set(),
            ),
        ):
            load_company_concepts_node(state)

        bootstrap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
