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

    def test_native_currency_is_saved_as_reported(self):
        # Phase 3: a foreign reporting currency is persisted as-is (no FX
        # conversion, no relabeling) — the run succeeds.
        _config.STRICT_ACCURACY = False
        with patch(
            "earnings_agents.nodes.save.require_detected_period",
            return_value=self._period(),
        ), patch(
            "earnings_agents.integrations.normalize.upsert_concept_values",
            return_value=1,
        ) as upsert:
            out = mongodb_save_node(_state(
                findings=[],
                currency_metadata={"currency": "EUR"},
                concept_metrics={"5072e8e0c9d3f4a1b2c3d4e5": 1.0},
                value_metadata_by_id={
                    "5072e8e0c9d3f4a1b2c3d4e5": {"currency": "EUR"},
                },
            ))
        self.assertEqual(out["status"], "saved")
        saved_meta = upsert.call_args.kwargs["value_metadata_by_id"]
        self.assertEqual(saved_meta["5072e8e0c9d3f4a1b2c3d4e5"]["currency"], "EUR")

    def test_unconfirmed_currency_value_is_dropped_not_saved(self):
        # No FX conversion and USD is never assumed: an unconfirmed currency
        # removes the value (the rest of the filing still saves).
        _config.STRICT_ACCURACY = False
        with patch(
            "earnings_agents.nodes.save.require_detected_period",
            return_value=self._period(),
        ), patch(
            "earnings_agents.integrations.normalize.upsert_concept_values",
            return_value=0,
        ) as upsert:
            out = mongodb_save_node(_state(
                findings=[],
                currency_metadata={"currency": "unknown"},
                concept_metrics={"5072e8e0c9d3f4a1b2c3d4e5": 1.0},
                value_metadata_by_id={
                    "5072e8e0c9d3f4a1b2c3d4e5": {"currency": "unknown"},
                },
            ))
        self.assertEqual(out["status"], "saved")
        # The value was dropped before the write; nothing is persisted as USD.
        self.assertEqual(out["concept_metrics"], {})
        self.assertFalse(upsert.called)

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

    def test_new_concept_is_materialized_only_after_gates_then_value_is_saved(self):
        _config.STRICT_ACCURACY = True
        cid = "5072e8e0c9d3f4a1b2c3d4e5"
        candidate = {
            "metric_key": "Revenue from new product",
            "value": 25.0,
            "label": "Revenue from new product",
            "statement_type": "income",
            "parent_concept": "us-gaap:Revenues",
            "parent_concept_id": "5072e8e0c9d3f4a1b2c3d4e6",
            "parent_path": "001",
            "dimension_concept": False,
            "source_lines": [10, 11],
            "scale": "millions",
            "currency": "USD",
        }
        persisted = {
            "concept_id": cid,
            "concept": {"concept": "custom:RevenueFromNewProduct_hash"},
            "created": True,
        }
        with patch(
            "earnings_agents.nodes.save.require_detected_period",
            return_value=self._period(),
        ), patch(
            "earnings_agents.integrations.normalize.persist_new_concept_candidates",
            return_value=({candidate["metric_key"]: persisted}, []),
        ) as create_concepts, patch(
            "earnings_agents.integrations.normalize.upsert_concept_values",
            return_value=1,
        ) as upsert:
            out = mongodb_save_node(_state(
                findings=[],
                concept_metrics={},
                value_metadata_by_id={},
                new_concept_candidates=[candidate],
            ))

        create_concepts.assert_called_once()
        saved_metrics = upsert.call_args.kwargs["concept_metrics"]
        self.assertEqual(saved_metrics, {cid: 25.0})
        saved_meta = upsert.call_args.kwargs["value_metadata_by_id"][cid]
        self.assertEqual(saved_meta["scale"], "millions")
        self.assertEqual(saved_meta["source_lines"], [10, 11])
        self.assertEqual(out["status"], "saved")

    def test_strict_gate_prevents_new_concept_creation(self):
        _config.STRICT_ACCURACY = True
        with patch(
            "earnings_agents.integrations.normalize.persist_new_concept_candidates"
        ) as create_concepts:
            out = mongodb_save_node(_state(new_concept_candidates=[{"label": "New row"}]))
        self.assertEqual(out["status"], "failed")
        create_concepts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
