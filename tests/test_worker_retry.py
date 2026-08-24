"""Regression tests for worker retry classification."""
from __future__ import annotations

import unittest

from earnings_agents.cli.worker import _NON_RETRYABLE_MARKERS


def _is_non_retryable(error: str) -> bool:
    return any(marker in error.lower() for marker in _NON_RETRYABLE_MARKERS)


class TestWorkerRetryClassification(unittest.TestCase):
    def test_provider_billing_errors_are_non_retryable(self):
        errors = [
            "period agent produced no result: 402 Payment Required",
            "Error code: 402 - Insufficient Balance",
            "unauthorized: invalid api key",
        ]
        for error in errors:
            self.assertTrue(_is_non_retryable(error))

    def test_duplicate_key_write_errors_are_non_retryable(self):
        # A DB write conflict is deterministic: re-running the same filing
        # against the same data fails identically, so it must go to the DLQ
        # instead of burning job-level retries on a full re-extract.
        error = (
            "batch op errors occurred, full error: ... E11000 duplicate key "
            "error collection: normalize_data.concept_values_quarterly ..."
        )
        self.assertTrue(_is_non_retryable(error))

    def test_transient_errors_remain_retryable(self):
        for error in ("timeout", "connection reset", "500 server error"):
            self.assertFalse(_is_non_retryable(error))


if __name__ == "__main__":
    unittest.main()
