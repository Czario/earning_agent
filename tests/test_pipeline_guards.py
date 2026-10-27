"""Unit tests for agent-driven completeness findings and pipeline guards."""
from __future__ import annotations

import unittest

from earnings_agents.agent.pipeline import (
    _build_completeness_findings,
    check_company_identity,
    check_ticker_identity,
)


class TestCompletenessFindings(unittest.TestCase):
    def test_foreign_currency_is_not_a_blocker(self):
        # Phase 3: a confirmed native currency (EUR) is saved as-is — it must
        # NOT produce a high-severity blocking finding.
        findings = _build_completeness_findings({"currency": "EUR"}, [], None)
        self.assertFalse(any(f["type"] == "currency_unresolved" for f in findings))

    def test_unknown_currency_finding_is_medium_observability(self):
        findings = _build_completeness_findings({"currency": "unknown"}, [], None)
        hits = [f for f in findings if f["type"] == "currency_unresolved"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["severity"], "medium")

    def test_no_currency_finding_for_usd(self):
        findings = _build_completeness_findings({"currency": "USD"}, [], None)
        self.assertFalse(any(f["type"] == "currency_unresolved" for f in findings))

    def test_missing_concepts_are_medium_observability(self):
        findings = _build_completeness_findings(
            {"currency": "USD"}, [], ["[us-gaap:Revenues]", "Net income"]
        )
        missing = [f for f in findings if f["type"] == "missing_concept"]
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(f["severity"] == "medium" for f in missing))

    def test_truncated_exhibit_is_incomplete(self):
        dm = [{"exhibit": "EX-99.3", "url": "x", "truncated": True}]
        findings = _build_completeness_findings({"currency": "USD"}, dm, None)
        self.assertTrue(any(f["type"] == "incomplete_document" for f in findings))

    def test_image_skip_is_not_incomplete(self):
        dm = [{"exhibit": "logo", "url": "x", "skipped": True, "reason": "non-text exhibit"}]
        findings = _build_completeness_findings({"currency": "USD"}, dm, None)
        self.assertFalse(any(f["type"] == "incomplete_document" for f in findings))


class TestStatementCurrencyResolutionInPipeline(unittest.TestCase):
    """Phase 3: one confirmed currency per statement, USD preferred."""

    def test_usd_preferred_and_eur_dropped(self):
        from earnings_agents.agent.currency import resolve_statement_currencies

        result = resolve_statement_currencies(
            {"a": "income", "b": "income"}, {"a": "USD", "b": "EUR"},
        )
        self.assertEqual(result["currency_by_concept"], {"a": "USD"})
        self.assertEqual(result["dropped"]["income"], ["b"])


class TestCompanyIdentity(unittest.TestCase):
    def test_same_company_matches(self):
        self.assertTrue(check_company_identity("Oracle Corporation", "Oracle Corporation"))

    def test_suffix_variant_matches(self):
        # "Pdd Holdings Inc." vs "PDD Holdings Inc." must not fail on a
        # casing/token-subset difference.
        self.assertTrue(check_company_identity("Pdd Holdings Inc.", "PDD Holdings Inc."))

    def test_abbreviation_variant_matches(self):
        self.assertTrue(check_company_identity("NVIDIA Corporation", "NVIDIA Corp"))
        self.assertTrue(check_company_identity("ABC CO", "ABC Company"))
        self.assertTrue(check_company_identity("ABC CORP", "ABC Company"))
        self.assertTrue(
            check_company_identity(
                "IBM CORP", "International Business Machines Corporation"
            )
        )

    def test_semantic_llm_company_identity_resolves_holding_vs_brand(self):
        from unittest.mock import MagicMock, patch

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = '{"same_company": true, "confidence": "high"}'
        with patch("earnings_agents.llm.build_llm", return_value=fake_llm):
            self.assertTrue(
                check_company_identity(
                    "Alphabet Inc.", "Google", ticker="GOOGL", use_llm=True
                )
            )

        fake_llm_mismatch = MagicMock()
        fake_llm_mismatch.invoke.return_value = '{"same_company": false, "confidence": "high"}'
        with patch("earnings_agents.llm.build_llm", return_value=fake_llm_mismatch):
            self.assertFalse(
                check_company_identity(
                    "Oracle Corporation", "Netflix, Inc.", ticker="ORCL", use_llm=True
                )
            )

    def test_wrong_company_fails(self):
        self.assertFalse(check_company_identity("Oracle Corporation", "Netflix, Inc."))

    def test_missing_name_is_not_a_mismatch(self):
        self.assertTrue(check_company_identity("", "Netflix, Inc."))
        self.assertTrue(check_company_identity("Oracle Corporation", ""))

    def test_ticker_identity_only_considers_ticker(self):
        self.assertTrue(check_ticker_identity("AAPL", "AAPL"))
        self.assertTrue(check_ticker_identity("AAPL", "NASDAQ: AAPL"))
        self.assertTrue(check_ticker_identity("BRK.A", "BRK.B"))
        self.assertTrue(check_ticker_identity("AAPL", ""))
        self.assertFalse(check_ticker_identity("ORCL", "NFLX"))
        self.assertFalse(check_ticker_identity("ORCL", "NASDAQ: NFLX"))


if __name__ == "__main__":
    unittest.main()
