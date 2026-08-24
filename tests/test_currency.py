"""Unit tests for deterministic, source-agnostic currency detection."""
from __future__ import annotations

import unittest

from earnings_agents.agent.currency import (
    detect_currency,
    is_usd_safe,
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

    def test_eur_explicit_requires_review(self):
        m = detect_currency("Amounts in millions of euros")
        self.assertEqual(m["currency"], "EUR")
        self.assertTrue(m["requires_review"])

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


class TestUsdValidation(unittest.TestCase):
    def test_is_usd_safe(self):
        self.assertTrue(is_usd_safe("USD"))
        self.assertFalse(is_usd_safe("EUR"))
        self.assertFalse(is_usd_safe("mixed"))
        self.assertFalse(is_usd_safe("unknown"))
        self.assertFalse(is_usd_safe(None))

    def test_validate_usd_metadata(self):
        ok, err = validate_currency_metadata({"currency": "USD"})
        self.assertTrue(ok)
        self.assertIsNone(err)

        ok, err = validate_currency_metadata({"currency": "EUR"})
        self.assertFalse(ok)
        self.assertIn("EUR", err)

        ok, err = validate_currency_metadata({"currency": "unknown"})
        self.assertFalse(ok)

        ok, err = validate_currency_metadata({"currency": "mixed"})
        self.assertFalse(ok)

        ok, err = validate_currency_metadata(None)
        self.assertFalse(ok)
        self.assertIn("missing", err)


if __name__ == "__main__":
    unittest.main()
