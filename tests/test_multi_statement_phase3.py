"""Tests for Phase 3: Extraction Agent Capabilities, Tooling & Prompt Architecture."""
from __future__ import annotations

import unittest

from earnings_agents.agent.prompts import PIPELINE_SYSTEM_PROMPT, build_concept_list
from earnings_agents.agent.tools import build_pi_tools


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


class TestPhase3Tools(unittest.TestCase):
    def setUp(self):
        self.tools = build_pi_tools("line1\nline2", {})

    def test_verify_balance_sheet_identity_balanced(self):
        tool = _tool(self.tools, "verify_balance_sheet_identity")
        # Assets: 1000 = Liabilities: 400 + Equity: 600
        res = tool.invoke({
            "total_assets": 1000.0,
            "total_liabilities": 400.0,
            "total_equity": 600.0,
        })
        self.assertIn("✓ VERIFIED", res)
        self.assertIn("Balance Sheet is balanced", res)

    def test_verify_balance_sheet_identity_unbalanced(self):
        tool = _tool(self.tools, "verify_balance_sheet_identity")
        # Assets: 1000 != Liabilities: 400 + Equity: 500
        res = tool.invoke({
            "total_assets": 1000.0,
            "total_liabilities": 400.0,
            "total_equity": 500.0,
        })
        self.assertIn("✗ FAILED", res)
        self.assertIn("Difference: 100", res)

    def test_verify_cash_flow_identity_balanced(self):
        tool = _tool(self.tools, "verify_cash_flow_identity")
        # Net change: 250 = Operating: 500 + Investing: -150 + Financing: -100
        res = tool.invoke({
            "operating_cf": 500.0,
            "investing_cf": -150.0,
            "financing_cf": -100.0,
            "net_change": 250.0,
        })
        self.assertIn("✓ VERIFIED", res)
        self.assertIn("Cash flow section totals match net change", res)

    def test_verify_cash_flow_identity_unbalanced(self):
        tool = _tool(self.tools, "verify_cash_flow_identity")
        # Net change: 300 != Operating: 500 + Investing: -150 + Financing: -100 (250)
        res = tool.invoke({
            "operating_cf": 500.0,
            "investing_cf": -150.0,
            "financing_cf": -100.0,
            "net_change": 300.0,
        })
        self.assertIn("✗ FAILED", res)
        self.assertIn("Difference: 50", res)


class TestPhase3Prompts(unittest.TestCase):
    def test_prompt_no_longer_ignores_bs_or_cf(self):
        self.assertNotIn("Balance sheet data (unless you need share counts for EPS)", PIPELINE_SYSTEM_PROMPT)
        self.assertNotIn("• Cash flow statement data", PIPELINE_SYSTEM_PROMPT)
        self.assertIn("verify_balance_sheet_identity", PIPELINE_SYSTEM_PROMPT)
        self.assertIn("verify_cash_flow_identity", PIPELINE_SYSTEM_PROMPT)
        self.assertIn("MULTI-STATEMENT EXTRACTION & LABEL DISAMBIGUATION", PIPELINE_SYSTEM_PROMPT)

    def test_build_concept_list_grouped_by_statement(self):
        concepts = [
            {
                "_id": "c_is",
                "label": "Total Revenues",
                "taxonomy_key": "us-gaap:Revenues",
                "statement_type": "income",
            },
            {
                "_id": "c_bs",
                "label": "Total Assets",
                "taxonomy_key": "us-gaap:Assets",
                "statement_type": "balancesheet",
            },
            {
                "_id": "c_cf",
                "label": "Net Cash Operating",
                "taxonomy_key": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
                "statement_type": "cashflow",
            },
        ]
        out = build_concept_list(concepts)
        self.assertIn("### INCOME STATEMENT CONCEPTS (Duration / Flow)", out)
        self.assertIn("### BALANCE SHEET CONCEPTS (Point-in-Time / As of Period End)", out)
        self.assertIn("### CASH FLOW STATEMENT CONCEPTS (Duration / Flow)", out)

        # Confirm concepts are listed under their respective sections
        is_idx = out.index("### INCOME STATEMENT CONCEPTS")
        bs_idx = out.index("### BALANCE SHEET CONCEPTS")
        cf_idx = out.index("### CASH FLOW STATEMENT CONCEPTS")

        self.assertTrue(is_idx < out.index("us-gaap:Revenues") < bs_idx)
        self.assertTrue(bs_idx < out.index("us-gaap:Assets") < cf_idx)
        self.assertTrue(cf_idx < out.index("us-gaap:NetCashProvidedByUsedInOperatingActivities"))


if __name__ == "__main__":
    unittest.main()
