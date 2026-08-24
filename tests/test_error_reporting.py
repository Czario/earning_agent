"""Regression tests for preserving exact provider failures."""
from __future__ import annotations

import unittest

from earnings_agents.agent.loop import AgentProviderError


class TestErrorReporting(unittest.TestCase):
    def test_provider_error_is_distinct(self):
        exc = AgentProviderError(
            "LLM provider failed at extraction step 1: HTTPStatusError: 402 Payment Required"
        )
        self.assertIn("402 Payment Required", str(exc))
        self.assertIsInstance(exc, RuntimeError)


if __name__ == "__main__":
    unittest.main()
