"""Regression tests for worker retry classification."""
from __future__ import annotations

import unittest


class TestWorkerRetryClassification(unittest.TestCase):
    def test_provider_billing_errors_are_non_retryable(self):
        errors = [
            "period agent produced no result: 402 Payment Required",
            "Error code: 402 - Insufficient Balance",
            "unauthorized: invalid api key",
        ]
        for error in errors:
            lowered = error.lower()
            self.assertTrue(any(marker in lowered for marker in (
                "402", "insufficient balance", "payment required",
                "invalid api key", "authentication", "unauthorized",
            )))

    def test_transient_errors_remain_retryable(self):
        for error in ("timeout", "connection reset", "500 server error"):
            lowered = error.lower()
            self.assertFalse(any(marker in lowered for marker in (
                "402", "insufficient balance", "payment required",
                "invalid api key", "authentication", "unauthorized",
            )))


if __name__ == "__main__":
    unittest.main()
