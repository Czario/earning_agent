"""Quarterly-only behavior for historical earnings backfills."""
from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.nodes.check import check_period_node


class TestQuarterlyOnlyBackfill(unittest.TestCase):
    def _period(self, period_type: str, quarter: int | None, label: str):
        return DetectedPeriod(
            period_type=period_type,
            period_end=date(2025, 12, 31) if period_type == "annual" else date(2025, 9, 28),
            quarter=quarter,
            fiscal_year=2025,
            period_label=label,
        )

    def test_annual_period_is_skipped_before_period_lookup(self):
        state = {
            "ticker": "ASML", "cik": "0000937966",
            "quarterly_only": True,
            "detected_period": {
                "period_type": "annual", "period_end": "2025-12-31",
                "quarter": None, "fiscal_year": 2025,
                "period_label": "Fiscal Year Ended December 31, 2025",
            },
        }
        with patch(
            "earnings_agents.nodes.check.require_detected_period",
            return_value=self._period(
                "annual", None, "Fiscal Year Ended December 31, 2025"
            ),
        ), patch(
            "earnings_agents.nodes.check.fiscal_period_exists"
        ) as exists:
            out = check_period_node(state)
        self.assertEqual(out["status"], "skipped")
        self.assertIn("quarterly-only mode", out["error"])
        exists.assert_not_called()

    def test_quarterly_period_continues_through_normal_period_check(self):
        state = {
            "ticker": "ASML", "cik": "0000937966",
            "quarterly_only": True,
            "detected_period": {
                "period_type": "quarterly", "period_end": "2025-09-28",
                "quarter": 3, "fiscal_year": 2025,
                "period_label": "Three Months Ended September 28, 2025",
            },
        }
        with patch(
            "earnings_agents.nodes.check.require_detected_period",
            return_value=self._period(
                "quarterly", 3, "Three Months Ended September 28, 2025"
            ),
        ), patch(
            "earnings_agents.nodes.check.fiscal_period_exists",
            return_value=False,
        ) as exists:
            out = check_period_node(state)
        self.assertNotEqual(out.get("status"), "skipped")
        exists.assert_called_once()


class TestQuarterlyOnlyOptIn(unittest.TestCase):
    """The same mode is selectable outside the backfill runner."""

    def test_cli_flag_sets_the_state_key(self):
        from earnings_agents.cli.earnings import _build_8k_state

        state = _build_8k_state(
            "ASML", "ASML HOLDING NV", "0000937966",
            filing_url="https://ir.asml.com/q2.pdf",
            quarterly_only=True,
            printer=lambda *_: None,
        )
        self.assertTrue(state["quarterly_only"])

    def test_cli_defaults_to_off(self):
        from earnings_agents.cli.earnings import _build_8k_state

        state = _build_8k_state(
            "ASML", "ASML HOLDING NV", "0000937966",
            filing_url="https://ir.asml.com/q2.pdf",
            printer=lambda *_: None,
        )
        self.assertNotIn("quarterly_only", state)


if __name__ == "__main__":
    unittest.main()
