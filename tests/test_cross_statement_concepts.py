"""Unit tests for cross-statement concept handling.

Ensures that concepts existing in more than one statement (e.g. us-gaap:NetIncomeLoss
in both income and cashflow, or Cash in balancesheet and cashflow) are:
1. Disambiguated in get_statement_concepts with statement-qualified taxonomy keys.
2. Correctly mapped in map_concepts without filtering out or dropping rows across statements.
3. Accessible via bare keys, statement-qualified keys, and statement aliases (|is, |bs, |cf).
4. Correctly deaccumulated and marked derived in cashflow.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from earnings_agents.agent.deaccumulate import resolve_cashflow_concept
from earnings_agents.agent.derive import map_concepts
from earnings_agents.integrations.normalize import get_statement_concepts


def _make_concept(
    _id: str,
    label: str,
    path: str,
    statement_type: str = "income",
    concept: str = "",
    taxonomy_key: str | None = None,
) -> dict:
    return {
        "_id": _id,
        "label": label,
        "path": path,
        "statement_type": statement_type,
        "concept": concept or _id,
        "taxonomy_key": taxonomy_key or concept or _id,
        "calculated": False,
        "dimension": False,
        "dimension_concept": False,
    }


class TestCrossStatementConceptTaxonomyKeys(unittest.TestCase):
    @patch("earnings_agents.integrations.normalize._get_client")
    def test_cross_statement_concepts_get_statement_qualified_keys(self, mock_client):
        """When a concept exists in both income and cashflow, taxonomy_key must include |st."""
        db = MagicMock()
        mock_client.return_value = {"normalize_data": db}

        cursor_data = [
            {
                "_id": "id_cf",
                "concept": "us-gaap:NetIncomeLoss",
                "label": "Net Income (Loss) Attributable to Parent",
                "statement_type": "cashflow",
                "path": "001.001",
                "order_key": 1,
            },
            {
                "_id": "id_is",
                "concept": "us-gaap:NetIncomeLoss",
                "label": "Net Income (Loss) Attributable to Parent",
                "statement_type": "income",
                "path": "009",
                "order_key": 1,
            },
            {
                "_id": "id_rev",
                "concept": "us-gaap:Revenues",
                "label": "Total Net Sales",
                "statement_type": "income",
                "path": "008",
                "order_key": 1,
            },
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = cursor_data
        db["normalized_concepts_quarterly"].find.return_value = mock_cursor

        from datetime import date
        from earnings_agents.agent.period import DetectedPeriod

        period = DetectedPeriod(
            period_type="quarterly",
            period_end=date(2026, 6, 27),
            quarter=3,
            fiscal_year=2026,
            period_label="Three Months Ended June 27, 2026",
        )
        concepts = get_statement_concepts("0000320193", ["income", "cashflow"], period=period)
        c_by_id = {c["_id"]: c for c in concepts}

        self.assertEqual(c_by_id["id_cf"]["taxonomy_key"], "us-gaap:NetIncomeLoss|cashflow")
        self.assertEqual(c_by_id["id_is"]["taxonomy_key"], "us-gaap:NetIncomeLoss|income")
        # Single-statement concept is not qualified with statement
        self.assertEqual(c_by_id["id_rev"]["taxonomy_key"], "us-gaap:Revenues")


class TestCrossStatementConceptMapping(unittest.TestCase):
    def setUp(self):
        self.target_concepts = [
            _make_concept(
                _id="id_income",
                label="Net Income",
                path="009",
                statement_type="income",
                concept="us-gaap:NetIncomeLoss",
                taxonomy_key="us-gaap:NetIncomeLoss|income",
            ),
            _make_concept(
                _id="id_cashflow",
                label="Net Income",
                path="001.001",
                statement_type="cashflow",
                concept="us-gaap:NetIncomeLoss",
                taxonomy_key="us-gaap:NetIncomeLoss|cashflow",
            ),
            _make_concept(
                _id="id_bs_cash",
                label="Cash and Cash Equivalents",
                path="001.001",
                statement_type="balancesheet",
                concept="us-gaap:CashAndCashEquivalentsAtCarryingValue",
                taxonomy_key="us-gaap:CashAndCashEquivalentsAtCarryingValue|balancesheet",
            ),
            _make_concept(
                _id="id_cf_cash",
                label="Cash and Cash Equivalents, Ending",
                path="001.005",
                statement_type="cashflow",
                concept="us-gaap:CashAndCashEquivalentsAtCarryingValue",
                taxonomy_key="us-gaap:CashAndCashEquivalentsAtCarryingValue|cashflow",
            ),
        ]

    def test_bare_concept_key_maps_to_both_statements(self):
        """Reporting [us-gaap:NetIncomeLoss] must map to both income and cashflow concepts."""
        metrics = {"[us-gaap:NetIncomeLoss]": 29789000000}
        concept_metrics, reverse_map, mapped_keys = map_concepts(metrics, self.target_concepts)

        self.assertIn("id_income", concept_metrics)
        self.assertIn("id_cashflow", concept_metrics)
        self.assertEqual(concept_metrics["id_income"], 29789000000.0)
        self.assertEqual(concept_metrics["id_cashflow"], 29789000000.0)
        self.assertIn("[us-gaap:NetIncomeLoss]", mapped_keys)

    def test_statement_qualified_keys_map_distinct_values(self):
        """Reporting statement-qualified keys maps each value to its respective statement."""
        metrics = {
            "[us-gaap:CashAndCashEquivalentsAtCarryingValue|balancesheet]": 23426000000,
            "[us-gaap:CashAndCashEquivalentsAtCarryingValue|cashflow]": 31102000000,
        }
        concept_metrics, reverse_map, mapped_keys = map_concepts(metrics, self.target_concepts)

        self.assertEqual(concept_metrics["id_bs_cash"], 23426000000.0)
        self.assertEqual(concept_metrics["id_cf_cash"], 31102000000.0)
        self.assertEqual(
            reverse_map["id_bs_cash"],
            "[us-gaap:CashAndCashEquivalentsAtCarryingValue|balancesheet]",
        )
        self.assertEqual(
            reverse_map["id_cf_cash"],
            "[us-gaap:CashAndCashEquivalentsAtCarryingValue|cashflow]",
        )

    def test_statement_aliases_map_correctly(self):
        """Aliases |is, |bs, |cf must map to income, balancesheet, cashflow."""
        metrics = {
            "[us-gaap:NetIncomeLoss|is]": 29789000000,
            "[us-gaap:NetIncomeLoss|cf]": 29789000000,
        }
        concept_metrics, reverse_map, mapped_keys = map_concepts(metrics, self.target_concepts)

        self.assertEqual(concept_metrics["id_income"], 29789000000.0)
        self.assertEqual(concept_metrics["id_cashflow"], 29789000000.0)

    def test_label_mapping_across_statements(self):
        """Reporting row label 'Net Income' maps across both statements."""
        metrics = {"Net Income": 29789000000}
        concept_metrics, reverse_map, mapped_keys = map_concepts(metrics, self.target_concepts)

        self.assertIn("id_income", concept_metrics)
        self.assertIn("id_cashflow", concept_metrics)
        self.assertEqual(concept_metrics["id_income"], 29789000000.0)
        self.assertEqual(concept_metrics["id_cashflow"], 29789000000.0)

    def test_resolve_cashflow_concept_matches_bare_and_qualified_keys(self):
        """resolve_cashflow_concept matches whether bare, qualified, or aliased key is passed."""
        c1 = resolve_cashflow_concept("[us-gaap:NetIncomeLoss]", self.target_concepts)
        self.assertIsNotNone(c1)
        self.assertEqual(c1["_id"], "id_cashflow")

        c2 = resolve_cashflow_concept("[us-gaap:NetIncomeLoss|cashflow]", self.target_concepts)
        self.assertIsNotNone(c2)
        self.assertEqual(c2["_id"], "id_cashflow")

        c3 = resolve_cashflow_concept("[us-gaap:NetIncomeLoss|cf]", self.target_concepts)
        self.assertIsNotNone(c3)
        self.assertEqual(c3["_id"], "id_cashflow")


    def test_siblings_within_same_statement_not_mapped_by_bare_key(self):
        """Two siblings in the SAME statement sharing a base concept (e.g. ServiceMember under Rev and CoR)
        must NOT both be blindly assigned the bare key's value."""
        siblings = [
            _make_concept("c_rev_srv", "Services Revenue", "001.001", "income", "us-gaap:ServiceMember", "us-gaap:ServiceMember|001.001"),
            _make_concept("c_cor_srv", "Services Cost", "002.001", "income", "us-gaap:ServiceMember", "us-gaap:ServiceMember|002.001"),
        ]
        metrics = {"[us-gaap:ServiceMember]": 5000}
        concept_metrics, reverse_map, mapped_keys = map_concepts(metrics, siblings)
        # Ambiguous sibling mapping must be rejected (neither mapped)
        self.assertEqual(concept_metrics, {})

    def test_map_concept_with_statement_type_filter(self):
        """map_concept tool respects statement_type argument."""
        from earnings_agents.agent.tools import build_pi_tools
        tools = build_pi_tools(
            document_text="line 1\nline 2",
            target_concepts=self.target_concepts,
            cik="0000320193",
        )
        tool_dict = {t.name: t for t in tools}
        map_fn = tool_dict["map_concept"]

        res_cf = map_fn.invoke({"metric_label": "Net Income", "statement_type": "cashflow"})
        self.assertIn("[us-gaap:NetIncomeLoss|cashflow]", res_cf)

        res_is = map_fn.invoke({"metric_label": "Net Income", "statement_type": "income"})
        self.assertIn("[us-gaap:NetIncomeLoss|income]", res_is)


if __name__ == "__main__":
    unittest.main()
