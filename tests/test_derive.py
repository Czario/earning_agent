"""Unit tests for hierarchy construction and derivation target selection."""
from __future__ import annotations

import unittest

from earnings_agents.agent.derive import (
    _build_hierarchy,
    build_calc_derivation_block,
    format_value_for_llm,
)


def _c(_id, path, order=None, concept="", calculated=False):
    return {
        "_id": _id,
        "path": path,
        "order_key": order,
        "concept": concept,
        "taxonomy_key": concept,
        "label": _id,
        "calculated": calculated,
        "dimension": False,
        "dimension_concept": False,
    }


class TestBuildHierarchy(unittest.TestCase):
    def test_direct_children_unambiguous(self):
        concepts = [
            _c("parent", "001"),
            _c("child1", "001.001"),
            _c("child2", "001.002"),
        ]
        parent_children, ambiguous = _build_hierarchy(concepts)
        self.assertEqual(set(parent_children["parent"]), {"child1", "child2"})
        self.assertEqual(ambiguous, set())

    def test_same_path_siblings_are_ambiguous(self):
        concepts = [
            _c("p1", "001", order=0),
            _c("p2", "001", order=1),
            _c("child", "001.001"),
        ]
        parent_children, ambiguous = _build_hierarchy(concepts)
        self.assertIn("001", ambiguous)
        self.assertNotIn("p1", parent_children)
        self.assertNotIn("p2", parent_children)

    def test_nested_paths_are_not_direct_children(self):
        concepts = [
            _c("parent", "001"),
            _c("mid", "001.001"),
            _c("leaf", "001.001.001"),
        ]
        parent_children, _ = _build_hierarchy(concepts)
        # "parent" only sees its direct child "mid", not the grandchild.
        self.assertEqual(set(parent_children["parent"]), {"mid"})


if __name__ == "__main__":
    unittest.main()


def _lc(_id, label, path, concept, calculated=False):
    return {
        "_id": _id,
        "label": label,
        "path": path,
        "concept": concept,
        "taxonomy_key": concept,
        "calculated": calculated,
        "dimension": False,
        "dimension_concept": False,
    }


class TestFormatValueForLlm(unittest.TestCase):
    def test_integral_values_keep_separators(self):
        self.assertEqual(format_value_for_llm(15400000000.0), "15,400,000,000")
        self.assertEqual(format_value_for_llm(-6798.0), "-6,798")

    def test_small_decimals_preserved(self):
        # EPS-style values must not collapse to "0" / "9" (the PDD loop).
        self.assertEqual(format_value_for_llm(0.32), "0.32")
        self.assertEqual(format_value_for_llm(8.94), "8.94")
        self.assertEqual(format_value_for_llm(-0.32), "-0.32")

    def test_trailing_zeros_stripped(self):
        self.assertEqual(format_value_for_llm(1.30), "1.3")


class TestCalcDerivationBlock(unittest.TestCase):
    """CALC concepts render as compute-only instructions for the agent."""

    def _pdd_like(self):
        return [
            _lc("rev", "Revenue", "001",
                "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"),
            _lc("cor", "Cost of Revenue", "002", "custom:CostOfRevenue"),
            _lc("gp", "Gross Profit", "003", "system:GrossProfit",
                calculated=True),
        ]

    def test_gross_profit_teaches_sign_convention(self):
        block, _ambiguous = build_calc_derivation_block(self._pdd_like())
        self.assertIn("[system:GrossProfit]", block)
        self.assertIn("Gross Profit", block)
        self.assertIn("NEVER subtract a negative", block)
        self.assertIn("NEVER 22,198", block)

    def test_sum_children_formula_lists_children(self):
        concepts = [
            _lc("parent", "Operating Expenses", "001", "system:OperatingExpenses",
                calculated=True),
            _lc("child1", "Sales and marketing", "001.001", "custom:SalesAndMarketing"),
            _lc("child2", "Research and development", "001.002", "custom:ResearchAndDevelopment"),
        ]
        block, _ambiguous = build_calc_derivation_block(concepts)
        self.assertIn("[system:OperatingExpenses]", block)
        self.assertIn("sum of", block)
        self.assertIn("Sales and marketing", block)
        self.assertIn("Research and development", block)

    def test_non_calc_concepts_are_not_rendered(self):
        block, _ambiguous = build_calc_derivation_block(
            [_lc("rev", "Revenue", "001", "us-gaap:Revenues")]
        )
        self.assertEqual(block, "")

    def test_ambiguous_paths_are_returned(self):
        concepts = [
            _lc("p1", "Parent A", "001", "system:OperatingExpenses", calculated=True),
            _lc("p2", "Parent B", "001", "system:OperatingExpenses", calculated=True),
            _lc("child", "Child", "001.001", "custom:Child"),
        ]
        block, ambiguous = build_calc_derivation_block(concepts)
        self.assertIn("001", ambiguous)
        self.assertIn("[system:OperatingExpenses]", block)
