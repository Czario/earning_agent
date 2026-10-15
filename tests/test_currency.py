"""Unit tests for currency detection and the USD-preferred/native policy."""
from __future__ import annotations

import unittest

from earnings_agents.agent.currency import (
    choose_statement_currency,
    detect_currency,
    document_currency,
    is_currency_code,
    is_usd_safe,
    resolve_statement_currencies,
    usd_metadata,
    validate_currency_metadata,
)


class TestCurrencyDetection(unittest.TestCase):
    def test_usd_explicit_name(self):
        m = detect_currency("Amounts in millions of U.S. dollars")
        self.assertEqual(m["currency"], "USD")
        self.assertEqual(m["confidence"], "high")
        self.assertFalse(m["requires_review"])

    def test_usd_symbol_is_low_confidence(self):
        m = detect_currency("Revenue $1,234 in millions")
        self.assertEqual(m["currency"], "USD")
        self.assertEqual(m["confidence"], "low")
        self.assertFalse(m["requires_review"])

    def test_eur_explicit_is_usable(self):
        # A single explicit native currency is usable as-is (no conversion).
        m = detect_currency("Amounts in millions of euros")
        self.assertEqual(m["currency"], "EUR")
        self.assertFalse(m["requires_review"])

    def test_twd_detected(self):
        m = detect_currency("Amounts in millions of New Taiwan dollars")
        self.assertEqual(m["currency"], "TWD")

    def test_nt_dollar_symbol_is_not_mistaken_for_usd(self):
        m = detect_currency("Amounts in NT$ thousands: NT$ 1,234")
        self.assertEqual(m["currency"], "TWD")
        self.assertNotIn("USD", m["detected_codes"])

    def test_multiple_foreign_codes_are_mixed(self):
        m = detect_currency("reported in euros and pounds sterling")
        self.assertEqual(m["currency"], "mixed")
        self.assertTrue(m["requires_review"])

    def test_no_declaration_is_unknown(self):
        m = detect_currency("Revenue 1234 in millions")
        self.assertEqual(m["currency"], "unknown")
        self.assertTrue(m["requires_review"])

    def test_usd_wins_over_constant_currency_mention(self):
        m = detect_currency(
            "Results reported in U.S. dollars; constant currency in euros"
        )
        self.assertEqual(m["currency"], "USD")

    def test_usd_metadata_shape(self):
        meta = usd_metadata("Amounts in millions of U.S. dollars")
        self.assertEqual(meta["currency"], "USD")
        self.assertIn("evidence", meta)
        self.assertIn("detected_codes", meta)


class TestCurrencyCodes(unittest.TestCase):
    def test_confirmed_codes(self):
        for code in ("USD", "EUR", "TWD", "JPY", "GBP", "usd"):
            with self.subTest(code=code):
                self.assertTrue(is_currency_code(code))

    def test_unconfirmed_values(self):
        for value in (None, "", "unknown", "mixed", "UNK", "XXX", "XAU", "US"):
            with self.subTest(value=value):
                self.assertFalse(is_currency_code(value))

    def test_is_usd_safe_legacy(self):
        self.assertTrue(is_usd_safe("USD"))
        self.assertFalse(is_usd_safe("EUR"))
        self.assertFalse(is_usd_safe("mixed"))
        self.assertFalse(is_usd_safe("unknown"))
        self.assertFalse(is_usd_safe(None))

    def test_validate_metadata(self):
        self.assertEqual(validate_currency_metadata({"currency": "USD"}), (True, None))
        self.assertEqual(validate_currency_metadata({"currency": "EUR"}), (True, None))
        self.assertEqual(validate_currency_metadata({"currency": "TWD"}), (True, None))

        ok, err = validate_currency_metadata({"currency": "unknown"})
        self.assertFalse(ok)
        self.assertIn("could not be determined", err)

        ok, err = validate_currency_metadata({"currency": "mixed"})
        self.assertFalse(ok)

        ok, err = validate_currency_metadata(None)
        self.assertFalse(ok)
        self.assertIn("missing", err)


class TestStatementCurrencyResolution(unittest.TestCase):
    def test_usd_wins_when_present(self):
        code, reason = choose_statement_currency(["EUR", "USD", "EUR"])
        self.assertEqual(code, "USD")
        self.assertIn("USD", reason)

    def test_single_native_currency_used(self):
        code, _ = choose_statement_currency(["TWD", "TWD"])
        self.assertEqual(code, "TWD")

    def test_no_confirmed_currency(self):
        code, reason = choose_statement_currency(["unknown", None])
        self.assertIsNone(code)
        self.assertIn("no confirmed", reason)

    def test_mixed_foreign_currencies_unresolved(self):
        code, reason = choose_statement_currency(["EUR", "GBP"])
        self.assertIsNone(code)
        self.assertIn("mixed", reason)

    def test_resolution_drops_non_matching_values(self):
        result = resolve_statement_currencies(
            {"a": "income", "b": "income", "c": "income", "d": "balancesheet"},
            {"a": "USD", "b": "EUR", "c": "USD", "d": "TWD"},
        )
        # Income prefers USD → a and c survive, b (EUR) is dropped.
        self.assertEqual(result["chosen_by_statement"]["income"], "USD")
        self.assertEqual(
            result["currency_by_concept"], {"a": "USD", "c": "USD", "d": "TWD"},
        )
        self.assertEqual(result["dropped"], {"income": ["b"]})
        # Balance sheet has a single native currency → kept.
        self.assertEqual(result["chosen_by_statement"]["balancesheet"], "TWD")
        self.assertNotIn("balancesheet", result["dropped"])

    def test_resolution_drops_whole_statement_when_unknown(self):
        result = resolve_statement_currencies(
            {"a": "income", "b": "income"}, {"a": None, "b": "unknown"},
        )
        self.assertIsNone(result["chosen_by_statement"]["income"])
        self.assertEqual(result["currency_by_concept"], {})
        self.assertEqual(sorted(result["dropped"]["income"]), ["a", "b"])

    def test_document_currency_collapse(self):
        self.assertEqual(document_currency({"income": "USD"}), "USD")
        self.assertEqual(document_currency({"income": "USD", "balancesheet": "EUR"}), "mixed")
        self.assertEqual(document_currency({"income": None}), "unknown")


if __name__ == "__main__":
    unittest.main()
