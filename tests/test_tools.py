"""Unit tests for the in-loop agent tools (map_concept/compute/detect_*)."""
from __future__ import annotations

import unittest

from earnings_agents.agent.tools import build_pi_tools


def _concept(_id, label, concept, dimension=False):
    return {
        "_id": _id,
        "label": label,
        "concept": concept,
        "taxonomy_key": concept,
        "calculated": False,
        "dimension": dimension,
        "dimension_concept": dimension,
    }


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


class TestMapConcept(unittest.TestCase):
    def setUp(self):
        self.concepts = [
            _concept("rev", "Revenue", "us-gaap:Revenues"),
            _concept("cor", "Cost of Revenue", "us-gaap:CostOfRevenue"),
            _concept("cloud", "Cloud and software", "custom:CloudExpense"),
        ]
        self.tools = build_pi_tools("line1\nline2", {}, target_concepts=self.concepts)

    def test_exact_label_match(self):
        out = _tool(self.tools, "map_concept").invoke({"metric_label": "Revenue"})
        self.assertIn("[us-gaap:Revenues]", out)
        self.assertIn("concept_id=rev", out)

    def test_whitespace_normalized_match(self):
        out = _tool(self.tools, "map_concept").invoke({"metric_label": "  revenue "})
        self.assertIn("[us-gaap:Revenues]", out)

    def test_no_match_returns_guidance(self):
        out = _tool(self.tools, "map_concept").invoke(
            {"metric_label": "Totally different row"}
        )
        self.assertIn("No concept found", out)

    def test_candidates_restrict_search(self):
        out = _tool(self.tools, "map_concept").invoke(
            {"metric_label": "Revenue", "candidates": ["Cost of Revenue"]}
        )
        self.assertIn("No concept found", out)


class TestCompute(unittest.TestCase):
    def test_exact_arithmetic(self):
        out = _tool(build_pi_tools("doc", {}), "compute").invoke(
            {"expression": "1000 - 400"}
        )
        self.assertEqual(out, "600")

    def test_parenthesized_is_negative(self):
        out = _tool(build_pi_tools("doc", {}), "compute").invoke(
            {"expression": "(175,685)"}
        )
        self.assertEqual(out, "-175685")


class TestScaleCurrencyTools(unittest.TestCase):
    def test_detect_scale_over_range(self):
        tools = build_pi_tools("Amounts in millions\nRevenue 1234", {})
        out = _tool(tools, "detect_scale").invoke({"start": 1, "end": 1})
        self.assertIn("millions", out)

    def test_detect_currency_over_range(self):
        tools = build_pi_tools("Amounts in euros\nRevenue 1234", {})
        out = _tool(tools, "detect_currency").invoke({"start": 1, "end": 1})
        self.assertIn("EUR", out)

    def test_invalid_range_rejected(self):
        tools = build_pi_tools("one\ntwo", {})
        out = _tool(tools, "detect_scale").invoke({"start": 5, "end": 9})
        self.assertIn("Invalid range", out)


if __name__ == "__main__":
    unittest.main()
