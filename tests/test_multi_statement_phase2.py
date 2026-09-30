"""Tests for Phase 2: Multi-statement concept loading, hierarchy disambiguation, and CALC formulas."""
from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from earnings_agents.agent.derive import (
    _build_hierarchy,
    build_calc_derivation_block,
    load_prior_values,
)
from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.nodes.concepts import load_company_concepts_node
from earnings_agents.state import EarningsAgentState


def _make_concept(
    _id: str,
    label: str,
    path: str,
    statement_type: str = "income",
    order_key: int | None = None,
    concept: str = "",
    calculated: bool = False,
) -> dict:
    return {
        "_id": _id,
        "label": label,
        "path": path,
        "statement_type": statement_type,
        "order_key": order_key,
        "concept": concept or _id,
        "taxonomy_key": concept or _id,
        "calculated": calculated,
        "dimension": False,
        "dimension_concept": False,
    }


class TestPhase2MultiStatementHierarchy(unittest.TestCase):
    def test_overlapping_paths_across_statements_do_not_collide(self):
        """Path '001' and '001.001' exist in both income and balancesheet statements.
        They must build independent parent-child trees and NOT be marked ambiguous."""
        concepts = [
            # Income statement hierarchy
            _make_concept("is_p", "Total Revenues", "001", statement_type="income"),
            _make_concept("is_c1", "Product Revenue", "001.001", statement_type="income"),
            _make_concept("is_c2", "Service Revenue", "001.002", statement_type="income"),
            # Balance sheet hierarchy with IDENTICAL paths
            _make_concept("bs_p", "Total Assets", "001", statement_type="balancesheet"),
            _make_concept("bs_c1", "Current Assets", "001.001", statement_type="balancesheet"),
            _make_concept("bs_c2", "Non-Current Assets", "001.002", statement_type="balancesheet"),
        ]

        parent_children, ambiguous_paths = _build_hierarchy(concepts)

        # Ambiguous paths must be empty because each statement has only 1 parent at '001'
        self.assertEqual(ambiguous_paths, set())

        # Income parent only receives income children
        self.assertEqual(parent_children.get("is_p"), ["is_c1", "is_c2"])

        # Balance sheet parent only receives balance sheet children
        self.assertEqual(parent_children.get("bs_p"), ["bs_c1", "bs_c2"])

    def test_same_statement_duplicate_path_is_still_ambiguous(self):
        """Two parent nodes sharing path '001' in the SAME statement must still be marked ambiguous."""
        concepts = [
            _make_concept("is_p1", "Rev Option 1", "001", statement_type="income", order_key=0),
            _make_concept("is_p2", "Rev Option 2", "001", statement_type="income", order_key=1),
            _make_concept("is_c", "Child Rev", "001.001", statement_type="income"),
        ]
        parent_children, ambiguous_paths = _build_hierarchy(concepts)
        self.assertIn("001", ambiguous_paths)
        self.assertNotIn("is_p1", parent_children)
        self.assertNotIn("is_p2", parent_children)

    def test_gross_profit_shortcut_only_applies_to_income_statement(self):
        """Gross Profit = Revenue - |Cost of Revenue| must only trigger for income statement concepts."""
        concepts = [
            _make_concept("c_is", "Gross Profit", "001", statement_type="income", calculated=True),
            _make_concept("c_bs", "Gross Profit Metric", "002", statement_type="balancesheet", calculated=True),
        ]
        block, _ = build_calc_derivation_block(concepts)
        # Income row gets the Revenue - |Cost of Revenue| rule
        self.assertIn("Gross Profit = Revenue − |Cost of Revenue|", block)
        # Balance sheet row does not get the GP formula; it falls back to general compute
        self.assertIn('"Gross Profit Metric"  ← COMPUTE from the related rows you extracted.', block)


class TestPhase2ConceptLoading(unittest.TestCase):
    @patch("earnings_agents.nodes.concepts.get_company_by_ticker")
    @patch("earnings_agents.nodes.concepts.get_statement_concepts")
    @patch("earnings_agents.nodes.concepts.get_recently_valued_concept_ids")
    def test_load_company_concepts_passes_target_statements(
        self, mock_get_recent, mock_get_concepts, mock_get_company
    ):
        mock_get_company.return_value = {
            "cik": "0001234567",
            "fiscal_year_end_month": 12,
            "fiscal_year_end_code": "1231",
            "industry": {"sic_code": "7372"},
        }
        mock_get_concepts.return_value = [
            _make_concept("c1", "Revenue", "001", statement_type="income"),
            _make_concept("c2", "Assets", "001", statement_type="balancesheet"),
        ]
        mock_get_recent.return_value = {"c1", "c2"}

        detected_period_dict = {
            "period_type": "quarterly",
            "period_end": date(2026, 6, 30),
            "quarter": 2,
            "fiscal_year": 2026,
            "period_label": "Three Months Ended June 30, 2026",
        }

        state: EarningsAgentState = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "discovered_file_url": None,
            "file_type": "html",
            "raw_text": None,
            "metrics": None,
            "error": None,
            "status": "pending",
            "findings": None,
            "detected_period": detected_period_dict,
            "target_statements": ["income", "balancesheet"],
        }

        new_state = load_company_concepts_node(state)
        from earnings_agents.agent.period import require_detected_period
        expected_period = require_detected_period(state)

        # Assert target_statements was passed to get_statement_concepts
        mock_get_concepts.assert_called_once_with(
            "0001234567",
            statement_types=["income", "balancesheet"],
            period=expected_period,
        )

        # Assert target_statements was passed to get_recently_valued_concept_ids
        mock_get_recent.assert_called_once_with(
            "0001234567",
            period=expected_period,
            n_periods=3,
            statement_types=["income", "balancesheet"],
        )

        self.assertEqual(len(new_state["target_concepts"]), 2)


if __name__ == "__main__":
    unittest.main()
