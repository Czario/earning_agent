"""Tests for Phase 1: Multi-statement configuration, state, and isolated stale sweep."""
from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.state import EarningsAgentState


class _FakeCollection:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.ops: list = []
        self._last_delete_filter: dict | None = None

    def bulk_write(self, ops, ordered=False):  # noqa: ARG002
        self.events.append("bulk_write")
        self.ops = list(ops)
        return MagicMock(acknowledged=True)

    def delete_many(self, filt):
        self.events.append("delete_many")
        self._last_delete_filter = filt
        return MagicMock(deleted_count=1)


class TestPhase1MultiStatement(unittest.TestCase):
    def test_config_target_statements_defaults(self):
        from earnings_agents import config
        self.assertEqual(config.TARGET_STATEMENTS, ["income", "balancesheet", "cashflow"])

    def test_state_typeddict_target_statements(self):
        state: EarningsAgentState = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "discovered_file_url": None,
            "file_type": "html",
            "raw_text": None,
            "metrics": None,
            "error": None,
            "status": "pending",
            "findings": None,
            "target_statements": ["income", "balancesheet"],
        }
        self.assertEqual(state.get("target_statements"), ["income", "balancesheet"])

    def test_stale_sweep_scoped_by_target_statements(self):
        period = DetectedPeriod(
            period_type="quarterly",
            period_end=date(2026, 6, 30),
            quarter=2,
            fiscal_year=2026,
            period_label="Three Months Ended June 30, 2026",
        )
        col = _FakeCollection()
        with patch(
            "earnings_agents.integrations.normalize._values_collection",
            return_value="concept_values_quarterly",
        ):
            import earnings_agents.integrations.normalize as _norm

            db = MagicMock()
            db.__getitem__.return_value = col
            client = MagicMock()
            client.__getitem__.return_value = db
            _norm._get_client = lambda: client  # noqa: SLF001

            # 1. Target income only
            _norm.upsert_concept_values(
                cik="000123",
                company_name="Test Co",
                concept_metrics={"5072e8e0c9d3f4a1b2c3d4e5": 100.0},
                period=period,
                target_statements=["income"],
            )
            self.assertIn("statement_type", col._last_delete_filter)
            self.assertEqual(col._last_delete_filter["statement_type"], {"$in": ["income"]})

            # 2. Target multiple statements
            _norm.upsert_concept_values(
                cik="000123",
                company_name="Test Co",
                concept_metrics={"5072e8e0c9d3f4a1b2c3d4e5": 100.0},
                period=period,
                target_statements=["income", "balancesheet"],
            )
            self.assertEqual(
                col._last_delete_filter["statement_type"],
                {"$in": ["income", "balancesheet"]},
            )

            # 3. Fallback when target_statements is None
            _norm.upsert_concept_values(
                cik="000123",
                company_name="Test Co",
                concept_metrics={"5072e8e0c9d3f4a1b2c3d4e5": 100.0},
                period=period,
                target_statements=None,
                statement_type="income",
            )
            self.assertEqual(col._last_delete_filter["statement_type"], "income")

    def test_doc_inherits_statement_type_from_metadata(self):
        period = DetectedPeriod(
            period_type="quarterly",
            period_end=date(2026, 6, 30),
            quarter=2,
            fiscal_year=2026,
            period_label="Three Months Ended June 30, 2026",
        )
        col = _FakeCollection()
        with patch(
            "earnings_agents.integrations.normalize._values_collection",
            return_value="concept_values_quarterly",
        ):
            import earnings_agents.integrations.normalize as _norm

            db = MagicMock()
            db.__getitem__.return_value = col
            client = MagicMock()
            client.__getitem__.return_value = db
            _norm._get_client = lambda: client  # noqa: SLF001

            cid_bs = "5072e8e0c9d3f4a1b2c3d4e6"
            _norm.upsert_concept_values(
                cik="000123",
                company_name="Test Co",
                concept_metrics={cid_bs: 500.0},
                period=period,
                statement_type="income",  # default
                value_metadata_by_id={
                    cid_bs: {
                        "statement_type": "balancesheet",
                    }
                },
                target_statements=["income", "balancesheet"],
            )
            op = col.ops[0]
            # Doc statement_type should be "balancesheet" from metadata, overriding fallback "income"
            self.assertEqual(op._doc["$set"]["statement_type"], "balancesheet")


if __name__ == "__main__":
    unittest.main()
