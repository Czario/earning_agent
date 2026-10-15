"""Tests for safe new-concept proposal validation and persistence."""
from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from bson import ObjectId

from earnings_agents.agent.new_concepts import prepare_new_concept_candidates
from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.integrations.normalize import persist_new_concept_candidates


class _Collection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])

    def find_one(self, query, projection=None):  # noqa: ARG002
        return next((d for d in self.docs if self._matches(d, query)), None)

    def find(self, query, projection=None):  # noqa: ARG002
        return [d for d in self.docs if self._matches(d, query)]

    def update_one(self, query, update, upsert=False):
        existing = next((d for d in self.docs if self._matches(d, query)), None)
        if existing is None and upsert:
            existing = {k: v for k, v in query.items() if not k.startswith("$")}
            self.docs.append(existing)
        if existing is not None and "$setOnInsert" in update and len(existing) == len(query):
            existing.update(update["$setOnInsert"])

    @staticmethod
    def _matches(doc, query):
        return all(doc.get(k) == v for k, v in query.items())


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


def _period(period_type="quarterly"):
    return DetectedPeriod(
        period_type=period_type,
        period_end=date(2026, 6, 30) if period_type == "quarterly" else date(2026, 12, 31),
        quarter=2 if period_type == "quarterly" else None,
        fiscal_year=2026,
        period_label="Three Months Ended June 30, 2026" if period_type == "quarterly" else "Year Ended December 31, 2026",
    )


class TestPrepareNewConceptCandidates(unittest.TestCase):
    def setUp(self):
        self.parent = {
            "_id": "5072e8e0c9d3f4a1b2c3d4e5",
            "concept": "us-gaap:Revenues",
            "taxonomy_key": "us-gaap:Revenues",
            "label": "Revenue",
            "statement_type": "income",
            "path": "001",
            "calculated": False,
        }

    def test_accepts_grounded_unmapped_value_with_existing_parent(self):
        candidates, issues = prepare_new_concept_candidates(
            [{
                "metric_key": "Revenue from new product",
                "label": "Revenue from new product",
                "statement_type": "income",
                "parent_key": "us-gaap:Revenues",
            }],
            metrics={"Revenue from new product": 25.0},
            target_concepts=[self.parent],
            mapped_keys=set(),
            derived_keys=set(),
            evidence_by_key={"Revenue from new product": {"lines": [10, 11], "scale": "millions", "currency": "USD"}},
            document_line_count=20,
            target_statements=["income"],
        )
        self.assertEqual(issues, [])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["parent_concept"], "us-gaap:Revenues")
        self.assertEqual(candidates[0]["parent_concept_id"], self.parent["_id"])
        self.assertEqual(candidates[0]["source_lines"], [10, 11])
        self.assertEqual(candidates[0]["currency"], "USD")

    def test_existing_mapping_wins_and_candidate_is_not_created(self):
        candidates, issues = prepare_new_concept_candidates(
            [{"metric_key": "Revenue", "label": "Revenue", "statement_type": "income", "parent_key": "ROOT"}],
            metrics={"Revenue": 100.0}, target_concepts=[self.parent],
            mapped_keys={"Revenue"}, derived_keys=set(),
            evidence_by_key={"Revenue": {"lines": [1, 1]}},
            document_line_count=1, target_statements=["income"],
        )
        self.assertEqual(candidates, [])
        self.assertEqual(issues, [])

    def test_rejects_missing_evidence_bad_parent_and_derived_values(self):
        proposals = [
            {"metric_key": "A", "label": "A new line", "statement_type": "income", "parent_key": "ROOT"},
            {"metric_key": "B", "label": "Bad parent", "statement_type": "income", "parent_key": "wrong:key"},
            {"metric_key": "C", "label": "Computed line", "statement_type": "income", "parent_key": "ROOT"},
        ]
        candidates, issues = prepare_new_concept_candidates(
            proposals,
            metrics={"A": 1.0, "B": 2.0, "C": 3.0},
            target_concepts=[self.parent], mapped_keys=set(), derived_keys={"C"},
            evidence_by_key={"B": {"lines": [1, 2]}, "C": {"lines": [1, 2]}},
            document_line_count=3, target_statements=["income"],
        )
        self.assertEqual(candidates, [])
        self.assertEqual(len(issues), 3)

    def test_dimensional_candidate_requires_parent(self):
        candidates, issues = prepare_new_concept_candidates(
            [{"metric_key": "Japan", "label": "Japan", "statement_type": "income", "parent_key": "ROOT", "dimension_concept": True}],
            metrics={"Japan": 5.0}, target_concepts=[self.parent], mapped_keys=set(),
            derived_keys=set(), evidence_by_key={"Japan": {"lines": [1, 2]}},
            document_line_count=2, target_statements=["income"],
        )
        self.assertEqual(candidates, [])
        self.assertIn("cannot be a statement root", issues[0])


class TestPersistNewConceptCandidates(unittest.TestCase):
    def setUp(self):
        self.parent_id = ObjectId()
        self.existing = {
            "_id": self.parent_id,
            "cik": "000123",
            "statement_type": "income",
            "concept": "us-gaap:Revenues",
            "label": "Revenue",
            "form_type": "10-Q",
            "path": "001",
            "order_key": "a",
            "dimension_concept": False,
        }
        self.collection = _Collection([self.existing])
        self.database = _Database({"normalized_concepts_quarterly": self.collection})

    def _persist(self, candidates, period=None):
        with patch(
            "earnings_agents.integrations.normalize._get_client",
            return_value=_Client(self.database),
        ):
            return persist_new_concept_candidates(
                "000123", period or _period(), candidates,
            )

    def test_creates_root_and_child_with_schema_fields_and_materialized_paths(self):
        result, issues = self._persist([
            {
                "metric_key": "Other revenue", "value": 2.0, "label": "Other revenue",
                "statement_type": "income", "parent_concept": None,
                "parent_concept_id": None, "parent_path": None,
                "dimension_concept": False,
            },
            {
                "metric_key": "Japan", "value": 1.0, "label": "Japan",
                "statement_type": "income", "parent_concept": "us-gaap:Revenues",
                "parent_concept_id": str(self.parent_id), "parent_path": "001",
                "dimension_concept": True, "dimension_member_label": "Japan",
                "dimension_axis": "custom:GeographyAxis",
            },
        ])
        self.assertEqual(issues, [])
        self.assertEqual(len(result), 2)
        root = result["Other revenue"]["concept"]
        self.assertEqual(root["form_type"], "10-Q")
        self.assertEqual(root["path"], "002")
        self.assertEqual(root["order_key"], "b")
        self.assertTrue(root["concept"].startswith("custom:OtherRevenue_"))
        member = result["Japan"]["concept"]
        self.assertEqual(member["path"], "001.001")
        self.assertTrue(member["dimension_concept"])
        self.assertEqual(member["concept_id"], self.parent_id)
        self.assertEqual(member["dimension_member_label"], "Japan")
        self.assertEqual(len(self.collection.docs), 3)

    def test_reprocessing_reuses_same_concept_and_id(self):
        candidate = {
            "metric_key": "Other revenue", "value": 2.0, "label": "Other revenue",
            "statement_type": "income", "parent_concept": None,
            "parent_concept_id": None, "parent_path": None,
            "dimension_concept": False,
        }
        first, issues = self._persist([candidate])
        second, issues2 = self._persist([candidate])
        self.assertEqual(issues, [])
        self.assertEqual(issues2, [])
        self.assertEqual(first["Other revenue"]["concept_id"], second["Other revenue"]["concept_id"])
        self.assertEqual(len(self.collection.docs), 2)
        self.assertFalse(second["Other revenue"]["created"])

    def test_unresolvable_parent_does_not_create_concept(self):
        result, issues = self._persist([{
            "metric_key": "Missing parent", "value": 3.0, "label": "Missing parent",
            "statement_type": "income", "parent_concept": "us-gaap:NotThere",
            "parent_concept_id": str(ObjectId()), "parent_path": "888",
            "dimension_concept": False,
        }])
        self.assertEqual(result, {})
        self.assertTrue(issues)
        self.assertEqual(len(self.collection.docs), 1)


if __name__ == "__main__":
    unittest.main()
