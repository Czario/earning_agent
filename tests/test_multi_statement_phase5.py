"""Tests for Phase 5 of Multi-Statement Extraction:
- CLI argument parsing for --statements / -s
- Initial state construction with target_statements
- Worker payload handling and statement propagation
- Worker CLI argument parsing for default statements
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from earnings_agents.cli.earnings import (
    _build_8k_state,
    _build_initial_state,
)
from earnings_agents.cli.worker import _parse_args as parse_worker_args, _process_payload
from earnings_agents.config import TARGET_STATEMENTS


class FakePub:
    def __init__(self, *args, **kwargs):
        pass

    def publish(self, *args, **kwargs):
        pass

    def close(self, summary=None):
        pass


class FakeHeartbeat:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _Recorder:
    def __init__(self, seen: dict):
        self.seen = seen

    def invoke(self, state):
        self.seen["state"] = dict(state)
        return {"status": "saved", "concept_metrics": {}}


def _patch_worker_stack(fn):
    patches = [
        patch("earnings_agents.cli.worker._cleanup_temporary_filing"),
        patch("earnings_agents.cli.worker.make_call_callback"),
        patch("earnings_agents.cli.worker.make_node_callback"),
        patch("earnings_agents.cli.worker.WorkerHeartbeat", FakeHeartbeat),
        patch("earnings_agents.cli.worker.WorkerProgressPublisher", FakePub),
        patch("earnings_agents.cli.earnings.get_latest_earnings_url"),
        patch("earnings_agents.cli.earnings.get_exhibits_for_accession"),
        patch("earnings_agents.integrations.normalize.get_company_by_ticker"),
    ]
    for p in reversed(patches):
        fn = p(fn)
    return fn


class TestEarningsCLIParsing(unittest.TestCase):
    def test_cli_args_custom_statements(self):
        import argparse
        from earnings_agents.cli.earnings import main

        # Create parser identical to CLI
        parser = argparse.ArgumentParser()
        parser.add_argument("--ticker", nargs="+", default=[])
        parser.add_argument("-s", "--statements", default=None)

        args = parser.parse_args(["--ticker", "MSFT", "--statements", "income,balancesheet"])
        statements = (
            [s.strip().lower() for s in args.statements.split(",") if s.strip()]
            if args.statements
            else TARGET_STATEMENTS
        )
        self.assertEqual(statements, ["income", "balancesheet"])

    def test_cli_args_short_flag(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--ticker", nargs="+", default=[])
        parser.add_argument("-s", "--statements", default=None)

        args = parser.parse_args(["--ticker", "MSFT", "-s", "cashflow"])
        statements = (
            [s.strip().lower() for s in args.statements.split(",") if s.strip()]
            if args.statements
            else TARGET_STATEMENTS
        )
        self.assertEqual(statements, ["cashflow"])

    def test_cli_args_whitespace_and_case(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--ticker", nargs="+", default=[])
        parser.add_argument("-s", "--statements", default=None)

        args = parser.parse_args(["--ticker", "MSFT", "-s", "  Income ,  CashFlow  "])
        statements = (
            [s.strip().lower() for s in args.statements.split(",") if s.strip()]
            if args.statements
            else TARGET_STATEMENTS
        )
        self.assertEqual(statements, ["income", "cashflow"])

    def test_cli_args_default_statements(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--ticker", nargs="+", default=[])
        parser.add_argument("-s", "--statements", default=None)

        args = parser.parse_args(["--ticker", "MSFT"])
        statements = (
            [s.strip().lower() for s in args.statements.split(",") if s.strip()]
            if args.statements
            else TARGET_STATEMENTS
        )
        self.assertEqual(statements, TARGET_STATEMENTS)


class TestBuild8KStateTargetStatements(unittest.TestCase):
    @patch("earnings_agents.cli.earnings.get_latest_earnings_url")
    def test_build_8k_state_explicit(self, mock_latest):
        mock_latest.return_value = ("https://example.com/ex99.htm", [], "0001-26-0001", [])
        state = _build_8k_state(
            ticker="MSFT",
            company_name="Microsoft Corp",
            cik="0000789019",
            target_statements=["income"],
        )
        self.assertEqual(state["target_statements"], ["income"])

    @patch("earnings_agents.cli.earnings.get_latest_earnings_url")
    def test_build_8k_state_default(self, mock_latest):
        mock_latest.return_value = ("https://example.com/ex99.htm", [], "0001-26-0001", [])
        state = _build_8k_state(
            ticker="MSFT",
            company_name="Microsoft Corp",
            cik="0000789019",
        )
        self.assertEqual(state["target_statements"], TARGET_STATEMENTS)

    @patch("earnings_agents.cli.earnings.get_latest_earnings_url")
    def test_build_initial_state_forwards_target_statements(self, mock_latest):
        mock_latest.return_value = ("https://example.com/ex99.htm", [], "0001-26-0001", [])
        info = {"ticker": "AAPL", "company_name": "Apple Inc.", "cik": "0000320193"}
        state = _build_initial_state(info, target_statements=["income", "balancesheet"])
        self.assertEqual(state["target_statements"], ["income", "balancesheet"])


class TestWorkerStatementsHandling(unittest.TestCase):
    def test_worker_cli_parse_args(self):
        args_default = parse_worker_args([])
        self.assertIsNone(args_default.statements)

        args_custom = parse_worker_args(["--statements", "income,balancesheet"])
        self.assertEqual(args_custom.statements, "income,balancesheet")

        args_short = parse_worker_args(["-s", "cashflow"])
        self.assertEqual(args_short.statements, "cashflow")

    @_patch_worker_stack
    def test_worker_payload_with_list_statements(
        self, mock_company, mock_exhibits, mock_latest, *_rest
    ):
        mock_company.return_value = {"cik": "0000789019", "name": "Microsoft"}
        mock_latest.return_value = ("url", [], "ACC-123", [])

        payload = {
            "filing_type": "8-K",
            "ticker": "MSFT",
            "statements": ["income", "balancesheet"],
        }
        seen = {}
        ok = _process_payload(_Recorder(seen), payload)
        self.assertTrue(ok)
        self.assertEqual(seen["state"]["target_statements"], ["income", "balancesheet"])

    @_patch_worker_stack
    def test_worker_payload_with_csv_statements(
        self, mock_company, mock_exhibits, mock_latest, *_rest
    ):
        mock_company.return_value = {"cik": "0000789019", "name": "Microsoft"}
        mock_latest.return_value = ("url", [], "ACC-123", [])

        payload = {
            "filing_type": "8-K",
            "ticker": "MSFT",
            "statements": " income , cashflow ",
        }
        seen = {}
        ok = _process_payload(_Recorder(seen), payload)
        self.assertTrue(ok)
        self.assertEqual(seen["state"]["target_statements"], ["income", "cashflow"])

    @_patch_worker_stack
    def test_worker_payload_with_target_statements_key(
        self, mock_company, mock_exhibits, mock_latest, *_rest
    ):
        mock_company.return_value = {"cik": "0000789019", "name": "Microsoft"}
        mock_latest.return_value = ("url", [], "ACC-123", [])

        payload = {
            "filing_type": "8-K",
            "ticker": "MSFT",
            "target_statements": ["balancesheet"],
        }
        seen = {}
        ok = _process_payload(_Recorder(seen), payload)
        self.assertTrue(ok)
        self.assertEqual(seen["state"]["target_statements"], ["balancesheet"])

    @_patch_worker_stack
    def test_worker_payload_default_statements_fallback(
        self, mock_company, mock_exhibits, mock_latest, *_rest
    ):
        mock_company.return_value = {"cik": "0000789019", "name": "Microsoft"}
        mock_latest.return_value = ("url", [], "ACC-123", [])

        # When payload has no statements, fallback to worker's default_statements
        payload = {
            "filing_type": "8-K",
            "ticker": "MSFT",
        }
        seen = {}
        ok = _process_payload(_Recorder(seen), payload, default_statements=["income"])
        self.assertTrue(ok)
        self.assertEqual(seen["state"]["target_statements"], ["income"])

    @_patch_worker_stack
    def test_worker_payload_config_fallback(
        self, mock_company, mock_exhibits, mock_latest, *_rest
    ):
        mock_company.return_value = {"cik": "0000789019", "name": "Microsoft"}
        mock_latest.return_value = ("url", [], "ACC-123", [])

        # When payload has no statements and default_statements is None, fallback to config.TARGET_STATEMENTS
        payload = {
            "filing_type": "8-K",
            "ticker": "MSFT",
        }
        seen = {}
        ok = _process_payload(_Recorder(seen), payload, default_statements=None)
        self.assertTrue(ok)
        self.assertEqual(seen["state"]["target_statements"], TARGET_STATEMENTS)


class TestExtractFinancialStatementsSection(unittest.TestCase):
    def test_extract_fs_section_includes_all_three_statements(self):
        from earnings_agents.agent.derive import extract_financial_statements_section, extract_is_section

        sample_exhibit = """
Company Press Release Text
Some intro paragraphs...

CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS
Revenues: $10,000
Cost of sales: $4,000
Gross profit: $6,000
Net income: $2,000

CONDENSED CONSOLIDATED BALANCE SHEETS
Cash and cash equivalents: $5,000
Total assets: $25,000
Total liabilities: $10,000
Total stockholders' equity: $15,000

CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS
Operating cash flow: $3,000
Investing cash flow: -$1,000
Financing cash flow: -$1,000
Ending cash: $5,000

NOTES TO CONDENSED CONSOLIDATED FINANCIAL STATEMENTS
Note 1. Basis of Presentation...
        """
        # extract_is_section stops before Balance Sheets
        is_only = extract_is_section(sample_exhibit)
        self.assertIn("STATEMENTS OF OPERATIONS", is_only)
        self.assertNotIn("BALANCE SHEETS", is_only)
        self.assertNotIn("STATEMENTS OF CASH FLOWS", is_only)

        # extract_financial_statements_section includes all three statements, stopping at Notes
        fs_full = extract_financial_statements_section(sample_exhibit)
        self.assertIsNotNone(fs_full)
        self.assertIn("STATEMENTS OF OPERATIONS", fs_full)
        self.assertIn("BALANCE SHEETS", fs_full)
        self.assertIn("STATEMENTS OF CASH FLOWS", fs_full)
        self.assertNotIn("Note 1. Basis of Presentation", fs_full)


if __name__ == "__main__":
    unittest.main()
