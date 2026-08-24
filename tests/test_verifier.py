"""Tests for the verifier agent, scale detection, and evidence contract."""
from __future__ import annotations

import unittest

from earnings_agents.agent.loop import _parse_llm_response
from earnings_agents.agent.scale import detect_scale, scale_multiplier
from earnings_agents.agent.verify import _parse_verifier_report


class TestDetectScale(unittest.TestCase):
    def test_millions_heading(self):
        res = detect_scale("Amounts in millions of U.S. dollars\nRevenue 1,234")
        self.assertEqual(res["scale"], "millions")
        self.assertEqual(scale_multiplier(res["scale"]), 1_000_000)

    def test_parenthesized_thousands(self):
        res = detect_scale("(in thousands, except per share data)")
        self.assertEqual(res["scale"], "thousands")

    def test_mixed_scales_require_review(self):
        res = detect_scale("(in millions)\nmore\n(in thousands)")
        self.assertEqual(res["scale"], "mixed")
        self.assertTrue(res["requires_review"])

    def test_no_declaration(self):
        res = detect_scale("Revenue\n1,234\nNet income 456")
        self.assertIsNone(res["scale"])
        self.assertTrue(res["requires_review"])


class TestEvidenceContract(unittest.TestCase):
    def test_evidence_block_preserved(self):
        out = _parse_llm_response(
            '{"__scale__":"millions","__currency__":"USD",'
            '"__evidence__":{"[us-gaap:Revenues]":{"lines":[210,215],'
            '"scale":"millions","currency":"USD"}},'
            '"[us-gaap:Revenues]": 1234}'
        )
        self.assertEqual(out["__evidence__"]["[us-gaap:Revenues]"]["lines"], [210, 215])
        self.assertEqual(out["[us-gaap:Revenues]"], 1_234_000_000)

    def test_evidence_string_form_parsed(self):
        out = _parse_llm_response(
            '{"__currency__":"USD",'
            '"__evidence__": "{\\"[us-gaap:Revenues]\\": {\\"lines\\": [10, 12]}}",'
            '"[us-gaap:Revenues]": 100}'
        )
        self.assertEqual(out["__evidence__"]["[us-gaap:Revenues]"]["lines"], [10, 12])

    def test_evidence_never_scaled(self):
        out = _parse_llm_response(
            '{"__scale__":"thousands","__currency__":"USD",'
            '"__evidence__":{"x":{"lines":[1,2]}},"x": 5}'
        )
        self.assertEqual(out["x"], 5_000)
        self.assertEqual(out["__evidence__"]["x"]["lines"], [1, 2])


class TestVerifierReport(unittest.TestCase):
    def test_valid_report(self):
        report = _parse_verifier_report(
            '{"__verifier_status__":"issues_found",'
            '"__verifier_issues__":[{"type":"missing_row","severity":"high",'
            '"concept":"[us-gaap:Revenues]","message":"row printed",'
            '"lines":[300,302]}]}'
        )
        self.assertIsNotNone(report)
        self.assertEqual(report["status"], "issues_found")
        self.assertEqual(len(report["actionable_issues"]), 1)

    def test_absent_ok_not_actionable(self):
        report = _parse_verifier_report(
            '{"__verifier_status__":"issues_found",'
            '"__verifier_issues__":[{"type":"absent_ok","severity":"low",'
            '"concept":"x","message":"absent"}]}'
        )
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["actionable_issues"], [])

    def test_invalid_type_rejected(self):
        report = _parse_verifier_report(
            '{"__verifier_status__":"issues_found",'
            '"__verifier_issues__":[{"type":"bogus","severity":"high"}]}'
        )
        self.assertEqual(report["status"], "verified")

    def test_garbage_returns_none(self):
        self.assertIsNone(_parse_verifier_report("not json at all"))


if __name__ == "__main__":
    unittest.main()
