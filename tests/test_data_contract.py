"""Baseline tests for the existing normalize_data value-document contract.

These tests intentionally capture current earning-agent behavior. Phase 0 is a
schema audit only; source-form and native-currency policy changes belong to
later implementation phases.
"""
from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from earnings_agents.agent.period import DetectedPeriod
from bson import ObjectId  # noqa: F401
from earnings_agents.integrations.normalize import upsert_concept_values


class _Collection:
    def __init__(self) -> None:
        self.ops = []
        self.delete_filter = None

    def bulk_write(self, ops, ordered=False):  # noqa: ARG002
        self.ops = list(ops)

    def delete_many(self, filt):
        self.delete_filter = filt
        return type("DeleteResult", (), {"deleted_count": 0})()


class _Database:
    def __init__(self) -> None:
        self.collections: dict[str, _Collection] = {}

    def __getitem__(self, name: str) -> _Collection:
        return self.collections.setdefault(name, _Collection())


class _Client:
    def __init__(self, db: _Database) -> None:
        self.db = db

    def __getitem__(self, name: str) -> _Database:  # noqa: ARG002
        return self.db


class TestNormalizeDataValueContract(unittest.TestCase):
    def _write(self, period: DetectedPeriod, currency: str = "EUR") -> tuple[str, dict]:
        db = _Database()
        concept_id = "5072e8e0c9d3f4a1b2c3d4e5"
        with patch(
            "earnings_agents.integrations.normalize._get_client",
            return_value=_Client(db),
        ):
            count = upsert_concept_values(
                cik="0001234567",
                company_name="Example Foreign Issuer",
                concept_metrics={concept_id: 123_456.0},
                period=period,
                statement_type="income",
                accession_number="0001234567-26-000001",
                value_metadata_by_id={
                    concept_id: {
                        "currency": currency,
                        "source_lines": [10, 11],
                        "scale": "millions",
                    },
                },
            )
        self.assertEqual(count, 1)
        collection_name, collection = next(iter(db.collections.items()))
        return collection_name, collection.ops[0]._doc["$set"]

    def test_quarterly_value_document_contract(self):
        period = DetectedPeriod(
            period_type="quarterly",
            period_end=date(2026, 6, 30),
            quarter=2,
            fiscal_year=2026,
            period_label="Three Months Ended June 30, 2026",
        )
        collection_name, doc = self._write(period)

        self.assertEqual(collection_name, "concept_values_quarterly")
        self.assertEqual(doc["cik"], "0001234567")
        self.assertEqual(doc["statement_type"], "income")
        # Legacy meaning: period-family form, not the original SEC filing form.
        self.assertEqual(doc["form_type"], "10-Q")
        self.assertEqual(doc["value"], 123_456.0)
        self.assertEqual(doc["currency"], "EUR")
        self.assertEqual(doc["scale"], "millions")
        # Evidence line ranges are deliberately NOT persisted.
        self.assertNotIn("source_lines", doc)
        self.assertNotIn("save_token", doc)
        self.assertEqual(doc["accession_number"], "0001234567-26-000001")
        self.assertEqual(doc["reporting_period"]["end_date"], datetime(
            2026, 6, 30, tzinfo=timezone.utc
        ))
        self.assertEqual(doc["reporting_period"]["period_date"], "2026-06-30")
        self.assertEqual(doc["reporting_period"]["fiscal_year"], 2026)
        self.assertEqual(doc["reporting_period"]["quarter"], 2)
        self.assertTrue(doc["earning_data"])

    def test_annual_value_document_contract_omits_quarter(self):
        period = DetectedPeriod(
            period_type="annual",
            period_end=date(2026, 12, 31),
            quarter=None,
            fiscal_year=2026,
            period_label="Year Ended December 31, 2026",
        )
        collection_name, doc = self._write(period)

        self.assertEqual(collection_name, "concept_values_annual")
        self.assertEqual(doc["form_type"], "10-K")
        self.assertEqual(doc["reporting_period"]["period_date"], "2026-12-31")
        self.assertEqual(doc["reporting_period"]["fiscal_year"], 2026)
        self.assertNotIn("quarter", doc["reporting_period"])
        self.assertNotIn("start_date", doc["reporting_period"])


class TestFormTypeProvenance(unittest.TestCase):
    """form_type must record the REAL filed SEC form, not the period family."""

    def _write(self, source_form_type, period_type="quarterly", currency="USD"):
        period = DetectedPeriod(
            period_type=period_type,
            period_end=date(2026, 6, 30) if period_type == "quarterly" else date(2026, 12, 31),
            quarter=2 if period_type == "quarterly" else None,
            fiscal_year=2026,
            period_label="Three Months Ended June 30, 2026"
            if period_type == "quarterly" else "Year Ended December 31, 2026",
        )
        db = _Database()
        cid = "5072e8e0c9d3f4a1b2c3d4e5"
        with patch(
            "earnings_agents.integrations.normalize._get_client",
            return_value=_Client(db),
        ):
            upsert_concept_values(
                cik="0000937966",
                company_name="ASML HOLDING NV",
                concept_metrics={cid: 100.0},
                period=period,
                value_metadata_by_id={cid: {"currency": currency}},
                source_form_type=source_form_type,
            )
        _, collection = next(iter(db.collections.items()))
        return collection.ops[0]._doc["$set"]

    def test_6k_is_recorded_as_6k(self):
        doc = self._write("6-K")
        self.assertEqual(doc["form_type"], "6-K")
        # The real form lives in `form_type` only — no duplicate field.
        self.assertNotIn("source_form_type", doc)

    def test_8k_is_recorded_as_8k(self):
        self.assertEqual(self._write("8-K")["form_type"], "8-K")

    def test_amendment_is_recorded(self):
        self.assertEqual(self._write("6-K/A")["form_type"], "6-K/A")

    def test_annual_foreign_form_is_recorded(self):
        doc = self._write("20-F", period_type="annual")
        self.assertEqual(doc["form_type"], "20-F")

    def test_missing_source_form_falls_back_to_period_family(self):
        # Provenance unknown (e.g. an IR PDF) → the period-family label keeps
        # the field non-empty instead of guessing a real form.
        self.assertEqual(self._write(None)["form_type"], "10-Q")
        self.assertEqual(
            self._write(None, period_type="annual")["form_type"], "10-K"
        )

    def test_normalize_form_type_rejects_sentinels(self):
        from earnings_agents.integrations.normalize import normalize_form_type

        self.assertIsNone(normalize_form_type("unknown"))
        self.assertIsNone(normalize_form_type("mixed"))
        self.assertIsNone(normalize_form_type(""))
        self.assertIsNone(normalize_form_type(None))
        self.assertEqual(normalize_form_type(" 6-k "), "6-K")


class TestAccessionFromUrl(unittest.TestCase):
    def test_edgar_archive_url(self):
        from earnings_agents.integrations.edgar import accession_from_url

        self.assertEqual(
            accession_from_url(
                "https://www.sec.gov/Archives/edgar/data/937966/"
                "000162828026048235/pressreleasefinancialresul.htm"
            ),
            "0001628280-26-048235",
        )

    def test_dashed_accession_in_filename(self):
        from earnings_agents.integrations.edgar import accession_from_url

        self.assertEqual(
            accession_from_url("https://x/uploads/0000937966-26-000012.pdf"),
            "0000937966-26-000012",
        )

    def test_unrelated_url(self):
        from earnings_agents.integrations.edgar import accession_from_url

        self.assertIsNone(accession_from_url("https://ir.co.com/q3-2026.pdf"))
        self.assertIsNone(accession_from_url(None))


if __name__ == "__main__":
    unittest.main()
