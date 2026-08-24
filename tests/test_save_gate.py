"""Tests for the STRICT_ACCURACY save gate and its --allow-inconsistent wiring.

The CLI's ``--allow-inconsistent`` flag flips ``config.STRICT_ACCURACY`` at
runtime; ``mongodb_save_node`` reads the knob lazily from the config module so
the override is honored.  The USD-only currency gate is a hard invariant and
must NOT be bypassable by relaxed accuracy.
"""
from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from earnings_agents.agent.period import DetectedPeriod
from earnings_agents import config as _config
from earnings_agents.nodes.save import mongodb_save_node


def _state(**overrides):
    state = {
        "ticker": "TEST",
        "company_name": "Test Co",
        "status": "extracted",
        "findings": [
            {"type": "incomplete_document", "severity": "high",
             "message": "Exhibit incomplete"},
        ],
        "currency_metadata": {"currency": "USD"},
        "cik": "000123",
        "concept_metrics": {"abc": 1.0},
        "detected_period": {
            "period_type": "quarterly",
            "period_end": "2026-06-30",
            "quarter": 2,
            "fiscal_year": 2026,
            "period_label": "Three Months Ended June 30, 2026",
        },
    }
    state.update(overrides)
    return state


class TestSaveGate(unittest.TestCase):
    def tearDown(self):
        _config.STRICT_ACCURACY = True  # restore default

    def _period(self):
        return DetectedPeriod(
            period_type="quarterly",
            period_end=date(2026, 6, 30),
            quarter=2,
            fiscal_year=2026,
            period_label="Three Months Ended June 30, 2026",
        )

    def test_high_finding_blocks_when_strict(self):
        _config.STRICT_ACCURACY = True
        out = mongodb_save_node(_state())
        self.assertEqual(out["status"], "failed")
        self.assertIn("Refusing to save", out["error"])

    def test_missing_metrics_never_block_the_save(self):
        # A few not-found metrics must not drop the whole period: whichever
        # values WERE found are still correct and get persisted.
        _config.STRICT_ACCURACY = True
        cases = [
            {"type": "missing_concept", "severity": "medium",
             "message": "Agent could not locate concept: [x]"},
        ]
        for finding in cases:
            with self.subTest(type=finding["type"]):
                with patch(
                    "earnings_agents.nodes.save.require_detected_period",
                    return_value=self._period(),
                ), patch(
                    "earnings_agents.integrations.normalize.upsert_concept_values",
                    return_value=1,
                ):
                    out = mongodb_save_node(_state(findings=[finding]))
                self.assertEqual(out["status"], "saved")

    def test_integrity_finding_still_blocks(self):
        # Integrity findings (a stored value is wrong / exhibit incomplete)
        # still block — the gate protects data that WOULD be written, not
        # metrics that are absent.
        _config.STRICT_ACCURACY = True
        out = mongodb_save_node(_state(findings=[
            {"type": "incomplete_document", "severity": "high",
             "message": "Exhibit incomplete: EX-99.1"},
        ]))
        self.assertEqual(out["status"], "failed")

    def test_allow_inconsistent_bypasses_strict_gate(self):
        # Simulates the CLI flag: config.STRICT_ACCURACY = False at runtime.
        _config.STRICT_ACCURACY = False
        with patch(
            "earnings_agents.nodes.save.require_detected_period",
            return_value=self._period(),
        ), patch(
            "earnings_agents.integrations.normalize.upsert_concept_values",
            return_value=1,
        ) as upsert:
            out = mongodb_save_node(_state())
        self.assertEqual(out["status"], "saved")
        upsert.assert_called_once()

    def test_currency_gate_is_not_bypassable(self):
        # Non-USD currency must fail even with STRICT_ACCURACY disabled.
        _config.STRICT_ACCURACY = False
        out = mongodb_save_node(_state(currency_metadata={"currency": "EUR"}))
        self.assertEqual(out["status"], "failed")
        self.assertIn("currency", out["error"].lower())

    def test_usd_currency_gate_passes_with_no_findings(self):
        _config.STRICT_ACCURACY = True
        with patch(
            "earnings_agents.nodes.save.require_detected_period",
            return_value=self._period(),
        ), patch(
            "earnings_agents.integrations.normalize.upsert_concept_values",
            return_value=1,
        ):
            out = mongodb_save_node(_state(findings=[]))
        self.assertEqual(out["status"], "saved")


if __name__ == "__main__":
    unittest.main()
