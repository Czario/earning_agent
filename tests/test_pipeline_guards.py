"""Unit tests for agent-driven completeness findings and pipeline guards."""
from __future__ import annotations

import unittest

from earnings_agents.agent.pipeline import (
    _build_completeness_findings,
    check_company_identity,
)


class TestCompletenessFindings(unittest.TestCase):
    def test_foreign_currency_finding(self):
        findings = _build_completeness_findings({"currency": "EUR"}, [], None)
        self.assertTrue(any(f["type"] == "unresolved_currency" for f in findings))

    def test_unknown_currency_finding(self):
        findings = _build_completeness_findings({"currency": "unknown"}, [], None)
        self.assertTrue(any(f["type"] == "unresolved_currency" for f in findings))

    def test_no_currency_finding_for_usd(self):
        findings = _build_completeness_findings({"currency": "USD"}, [], None)
        self.assertFalse(any(f["type"] == "unresolved_currency" for f in findings))

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


class TestCompanyIdentity(unittest.TestCase):
    def test_same_company_matches(self):
        self.assertTrue(check_company_identity("Oracle Corporation", "Oracle Corporation"))

    def test_suffix_variant_matches(self):
        # "Pdd Holdings Inc." vs "PDD Holdings Inc." must not fail on a
        # casing/token-subset difference.
        self.assertTrue(check_company_identity("Pdd Holdings Inc.", "PDD Holdings Inc."))

    def test_abbreviation_variant_matches(self):
        self.assertTrue(check_company_identity("NVIDIA Corporation", "NVIDIA Corp"))

    def test_wrong_company_fails(self):
        self.assertFalse(check_company_identity("Oracle Corporation", "Netflix, Inc."))

    def test_missing_name_is_not_a_mismatch(self):
        self.assertTrue(check_company_identity("", "Netflix, Inc."))
        self.assertTrue(check_company_identity("Oracle Corporation", ""))


if __name__ == "__main__":
    unittest.main()
