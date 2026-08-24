"""Tests for atomic (write-first) concept-value persistence."""
from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from earnings_agents.agent.period import DetectedPeriod


class _FakeCollection:
    """Records call order: bulk_write must precede the stale sweep."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.ops: list = []

    def bulk_write(self, ops, ordered=False):  # noqa: ARG002
        self.events.append("bulk_write")
        self.ops = list(ops)
        return MagicMock(acknowledged=True)

    def delete_many(self, filt):
        self.events.append("delete_many")
        self._last_delete_filter = filt
        return MagicMock(deleted_count=1)


class TestAtomicUpsert(unittest.TestCase):
    def test_write_before_stale_sweep(self):
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
            upsert_concept_values = _norm.upsert_concept_values

            # _get_client() → client; client[db] → db; db[collection] → col.
            db = MagicMock()
            db.__getitem__.return_value = col
            client = MagicMock()
            client.__getitem__.return_value = db
            _norm._get_client = lambda: client  # noqa: SLF001
            n = upsert_concept_values(
                cik="000123",
                company_name="Test Co",
                concept_metrics={
                    "5072e8e0c9d3f4a1b2c3d4e5": 1_000_000.0,
                },
                period=period,
            )
        # The stale sweep must only run AFTER the write.
        self.assertEqual(col.events, ["bulk_write", "delete_many"])
        self.assertIn("save_token", col._last_delete_filter)
        self.assertEqual(
            col._last_delete_filter["save_token"],
            {"$ne": col.ops[0]._doc["$set"]["save_token"]},
        )
        self.assertEqual(n, 1)

    def test_dimensional_filter_is_unique_index_key(self):
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
            upsert_concept_values = _norm.upsert_concept_values

            db = MagicMock()
            db.__getitem__.return_value = col
            client = MagicMock()
            client.__getitem__.return_value = db
            _norm._get_client = lambda: client  # noqa: SLF001
            upsert_concept_values(
                cik="000123",
                company_name="Test Co",
                concept_metrics={"5072e8e0c9d3f4a1b2c3d4e5": 5.0},
                period=period,
                value_metadata_by_id={
                    "5072e8e0c9d3f4a1b2c3d4e5": {
                        "dimension": True,
                        "dimension_member": "us-gaap:ProductMember",
                        "dimension_axis": "us-gaap:BusinessSegmentAxis",
                    },
                },
            )
        op = col.ops[0]
        # The upsert filter is EXACTLY the collection's unique index key
        # (cik, concept_id, fiscal_year, quarter).  concept_id already
        # identifies the row, so dimension_member/axis are metadata only and
        # must NOT be in the filter (they caused E11000 collisions on re-runs).
        self.assertEqual(op._filter["cik"], "000123")
        self.assertEqual(str(op._filter["concept_id"]), "5072e8e0c9d3f4a1b2c3d4e5")
        self.assertEqual(op._filter["reporting_period.fiscal_year"], 2026)
        self.assertEqual(op._filter["reporting_period.quarter"], 2)
        self.assertNotIn("dimension_member", op._filter)
        self.assertNotIn("dimension_axis", op._filter)
        # ...but the stored document still carries the member/axis metadata.
        self.assertEqual(op._doc["$set"]["dimension_member"], "us-gaap:ProductMember")
        self.assertEqual(op._doc["$set"]["dimension_axis"], "us-gaap:BusinessSegmentAxis")


if __name__ == "__main__":
    unittest.main()
