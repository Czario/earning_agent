"""Unit tests for agent-driven completeness findings and pipeline guards."""
from __future__ import annotations

import unittest

from earnings_agents.agent.pipeline import (
    _issue_signatures,
    _build_completeness_findings,
    _findings_from_verifier,
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


class TestVerifierFindings(unittest.TestCase):
    def _concept(self, _id, label, key):
        return {
            "_id": _id, "label": label, "concept": key, "taxonomy_key": key,
            "calculated": False, "dimension": False, "dimension_concept": False,
        }

    def test_high_value_mismatch_is_blocking(self):
        findings, briefing, retry_ids = _findings_from_verifier({
            "status": "issues_found",
            "issues": [{
                "type": "value_mismatch", "severity": "high",
                "concept": "[us-gaap:Revenues]", "message": "number differs",
                "lines": [10, 12],
            }],
        })
        self.assertTrue(any(f["severity"] == "high" for f in findings))
        self.assertIn("WRONG VALUE", briefing)

    def test_missing_row_builds_targeted_briefing(self):
        _findings, briefing, _retry_ids = _findings_from_verifier({
            "status": "issues_found",
            "issues": [{
                "type": "missing_row", "severity": "high",
                "concept": "[us-gaap:InterestExpense]",
                "message": "row printed at 300", "lines": [300, 302],
            }],
        })
        self.assertIn("MISSING", briefing)
        self.assertIn("300-302", briefing)

    def test_absent_ok_is_not_actionable(self):
        findings, briefing, _retry_ids = _findings_from_verifier({
            "status": "issues_found",
            "issues": [{"type": "absent_ok", "severity": "low",
                        "concept": "x", "message": "absent"}],
        })
        self.assertEqual(briefing, "")
        self.assertFalse(any(f["severity"] == "high" for f in findings))

    def test_no_report_yields_verification_unavailable(self):
        findings, briefing, retry_ids = _findings_from_verifier(None)
        self.assertEqual(briefing, "")
        self.assertEqual(retry_ids, set())
        self.assertTrue(any(f["type"] == "verification_unavailable" for f in findings))


class TestRetryScope(unittest.TestCase):
    """The retry pass is scoped to ONLY flagged/missing concepts."""

    def _concept(self, _id, label, key, calculated=False):
        return {
            "_id": _id, "label": label, "concept": key, "taxonomy_key": key,
            "calculated": calculated, "dimension": False, "dimension_concept": False,
        }

    def test_retry_ids_include_flagged_and_missing(self):
        concepts = [
            self._concept("rev", "Revenue", "us-gaap:Revenues"),
            self._concept("ie", "Interest Expense", "us-gaap:InterestExpense"),
            self._concept("ni", "Net Income", "us-gaap:NetIncomeLoss"),
        ]
        findings, briefing, retry_ids = _findings_from_verifier(
            {
                "status": "issues_found",
                "issues": [{
                    "type": "missing_row", "severity": "high",
                    "concept": "[us-gaap:InterestExpense]",
                    "message": "row printed at 300",
                }],
            },
            target_concepts=concepts,
            missing_labels=["Net Income"],
        )
        self.assertEqual(retry_ids, {"ie", "ni"})
        self.assertNotIn("rev", retry_ids)
        # The briefing lists both the flagged issue and the still-missing label.
        self.assertIn("[us-gaap:InterestExpense]", briefing)
        self.assertIn("Net Income", briefing)

    def test_calculated_concepts_never_in_retry_scope(self):
        # Derivation targets (calculated/system:) cannot be extracted by the
        # agent — retrying them is pure churn (the Gross Profit loop).
        concepts = [
            self._concept("rev", "Revenue", "us-gaap:Revenues"),
            self._concept("gp", "Gross Profit", "system:GrossProfit", calculated=True),
        ]
        _f, _b, retry_ids = _findings_from_verifier(
            {
                "status": "issues_found",
                "issues": [{
                    "type": "missing_row", "severity": "high",
                    "concept": "Gross Profit", "message": "row printed",
                }],
            },
            target_concepts=concepts,
            missing_labels=["Gross Profit"],
        )
        self.assertNotIn("gp", retry_ids)

    def test_needs_confirmation_does_not_drive_retry(self):
        # Verifier uncertainty is recorded but never burns an extraction pass.
        concepts = [self._concept("rev", "Revenue", "us-gaap:Revenues")]
        findings, briefing, retry_ids = _findings_from_verifier(
            {
                "status": "issues_found",
                "issues": [{
                    "type": "needs_confirmation", "severity": "medium",
                    "concept": "[us-gaap:Revenues]", "message": "not sure",
                }],
            },
            target_concepts=concepts,
        )
        self.assertEqual(briefing, "")
        self.assertEqual(retry_ids, set())
        self.assertTrue(any(f["type"] == "verifier_needs_confirmation" for f in findings))

    def test_no_wanted_concepts_yields_empty_scope(self):
        concepts = [self._concept("rev", "Revenue", "us-gaap:Revenues")]
        _f, _b, retry_ids = _findings_from_verifier(
            {"status": "issues_found", "issues": []},
            target_concepts=concepts,
        )
        self.assertEqual(retry_ids, set())


if __name__ == "__main__":
    unittest.main()


class TestNonBlockingVerifierIssues(unittest.TestCase):
    """Absence/uncertainty never blocks the save — the LLM's severity label
    is not trusted for missing_row / needs_confirmation."""

    def test_high_missing_row_never_blocks(self):
        findings, briefing, _ = _findings_from_verifier({
            "status": "issues_found",
            "issues": [{
                "type": "missing_row", "severity": "high",
                "concept": "[custom:404]", "message": "not printed anywhere",
            }],
        })
        self.assertFalse(any(f["severity"] == "high" for f in findings))
        # …but it still drives a targeted retry briefing.
        self.assertIn("MISSING", briefing)

    def test_high_needs_confirmation_never_blocks_or_retries(self):
        findings, briefing, retry_ids = _findings_from_verifier({
            "status": "issues_found",
            "issues": [{
                "type": "needs_confirmation", "severity": "high",
                "concept": "[us-gaap:Revenues]", "message": "unsure",
            }],
        })
        self.assertFalse(any(f["severity"] == "high" for f in findings))
        self.assertEqual(briefing, "")
        self.assertEqual(retry_ids, set())


class TestIssueSignatures(unittest.TestCase):
    """Non-convergence guard: identical issues after a retry → stop early."""

    def _report(self, found_value):
        return {
            "status": "issues_found",
            "issues": [{
                "type": "value_mismatch", "severity": "high",
                "concept": "[custom:Basic|015.001]",
                "message": "differs",
                "reported_value": 1.30, "found_value": found_value,
            }],
        }

    def test_identical_reports_have_equal_signatures(self):
        self.assertEqual(
            _issue_signatures(self._report(8.94)),
            _issue_signatures(self._report(8.94)),
        )

    def test_changed_value_changes_signature(self):
        self.assertNotEqual(
            _issue_signatures(self._report(8.94)),
            _issue_signatures(self._report(1.30)),
        )

    def test_absent_ok_is_not_actionable_in_signature(self):
        report = {
            "status": "verified",
            "issues": [{"type": "absent_ok", "severity": "low",
                        "concept": "x", "message": "absent"}],
        }
        self.assertEqual(_issue_signatures(report), frozenset())
        self.assertEqual(_issue_signatures(None), frozenset())
