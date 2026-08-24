"""Unit tests for hierarchy construction and derivation target selection."""
from __future__ import annotations

import unittest

from earnings_agents.agent.derive import (
    _build_derivation_prompt,
    _build_hierarchy,
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


class TestDerivationPrompt(unittest.TestCase):
    """Gross Profit derivation needs BOTH operands visible + sign teaching."""

    def _pdd_like(self):
        return [
            _lc("rev", "Revenue", "001",
                "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"),
            _lc("cor", "Cost of Revenue", "002", "custom:CostOfRevenue"),
            _lc("gp", "Gross Profit", "003", "system:GrossProfit",
                calculated=True),
        ]

    def test_gross_profit_prompt_shows_both_operands(self):
        # Cost of Revenue is an extracted LEAF (not a hierarchy parent), so
        # descendant expansion never surfaces it — it must be referenced
        # explicitly or GP is deterministically "not computable".
        prompt = _build_derivation_prompt(
            {"rev": 15400.0, "cor": -6798.0}, self._pdd_like()
        )
        self.assertIn("Revenue = 15,400", prompt)
        self.assertIn("Cost of Revenue = -6,798", prompt)

    def test_sign_convention_is_taught(self):
        prompt = _build_derivation_prompt(
            {"rev": 15400.0, "cor": -6798.0}, self._pdd_like()
        )
        self.assertIn("− |Cost of Revenue|", prompt)
        self.assertIn("NEVER subtract a negative", prompt)
        self.assertIn("NEVER 22,198", prompt)

    def test_missing_cor_operand_falls_back_to_full_block(self):
        concepts = [
            _lc("rev", "Revenue", "001",
                "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"),
            _lc("gp", "Gross Profit", "003", "system:GrossProfit",
                calculated=True),
        ]
        prompt = _build_derivation_prompt(
            {"rev": 15400.0, "unrelated": 5.0}, concepts
        )
        # Dependency uncertainty → full extracted block, not a lean one.
        self.assertIn("unrelated = 5", prompt)
