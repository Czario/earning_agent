"""Guardrail: no bookkeeping/ledger collections may be reintroduced.

The pipeline persists ONLY to the normalized concept/value collections (plus
``companies`` for the identity lookup).  Resume state comes from
``accession_number`` on the saved values, and the quarterly concept bootstrap's
state is the quarterly concept collection itself — so there is nothing to
create, migrate, or clean up.

Two independent checks:

1. **Static** — the forbidden collection names appear nowhere in ``src/``.
2. **Behavioural** — a guarded Mongo client raises if any flow asks for a
   collection outside the allowed set.  This catches a reintroduction even if
   it is built by string concatenation.
"""
from __future__ import annotations

import re
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from bson import ObjectId

# Collections this project must never create or write again.
FORBIDDEN_COLLECTIONS = frozenset({
    "earnings_backfill_ledger",
    "quarterly_concept_bootstrap",
})

# Everything the normalized-data flows may touch.
ALLOWED_COLLECTIONS = frozenset({
    "companies",
    "normalized_concepts_annual",
    "normalized_concepts_quarterly",
    "concept_values_annual",
    "concept_values_quarterly",
})


class ForbiddenCollectionError(AssertionError):
    """Raised when code asks for a collection it must not use."""


class _GuardedCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.docs: list[dict] = []

    def find_one(self, query, projection=None):  # noqa: ARG002
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return doc
        return None

    def find(self, query, projection=None):  # noqa: ARG002
        return [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]

    def distinct(self, field, query):  # noqa: ARG002
        return sorted({
            d.get(field) for d in self.docs
            if all(d.get(k) == v for k, v in query.items()) and d.get(field)
        })

    def update_one(self, query, update, upsert=False):  # noqa: ARG002
        return type("R", (), {"upserted_id": None, "modified_count": 0})()

    def bulk_write(self, ops, ordered=False):  # noqa: ARG002
        return type("R", (), {"upserted_count": len(ops)})()

    def delete_many(self, filt):  # noqa: ARG002
        return type("R", (), {"deleted_count": 0})()

    def count_documents(self, query):  # noqa: ARG002
        return 0

    def aggregate(self, *args, **kwargs):  # noqa: ARG002
        return []


class _GuardedDb:
    """DB handle that refuses forbidden collections and records every access."""

    def __init__(self) -> None:
        self.accessed: set[str] = set()
        self._collections: dict[str, _GuardedCollection] = {}

    def __getitem__(self, name: str) -> _GuardedCollection:
        self.accessed.add(name)
        if name in FORBIDDEN_COLLECTIONS:
            raise ForbiddenCollectionError(
                f"forbidden collection {name!r} was requested — this project must "
                "derive resume/bootstrap state from the existing concept/value "
                "collections instead of a bookkeeping collection"
            )
        return self._collections.setdefault(name, _GuardedCollection(name))


class _GuardedClient:
    def __init__(self, db: _GuardedDb) -> None:
        self.db = db

    def __getitem__(self, name: str) -> _GuardedDb:  # noqa: ARG002
        return self.db


class TestNoBookkeepingCollectionsInSource(unittest.TestCase):
    def test_forbidden_names_do_not_appear_in_src(self):
        src_root = Path(__file__).resolve().parent.parent / "src"
        offenders: list[str] = []
        for path in sorted(src_root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name in FORBIDDEN_COLLECTIONS:
                if re.search(re.escape(name), text):
                    offenders.append(f"{path.relative_to(src_root.parent)}: {name}")
        self.assertEqual(
            offenders,
            [],
            "bookkeeping collections must never be reintroduced in src/: "
            + ", ".join(offenders),
        )

    def test_scan_control_detects_the_names_when_present(self):
        """Control: the static scan really matches these identifiers."""
        this_file = Path(__file__).read_text(encoding="utf-8")
        for name in FORBIDDEN_COLLECTIONS:
            self.assertIsNotNone(
                re.search(re.escape(name), this_file),
                f"scan control failed for {name!r}",
            )


class TestNoBookkeepingCollectionsAtRuntime(unittest.TestCase):
    def _db(self) -> _GuardedDb:
        db = _GuardedDb()
        # Seed the company's annual concepts so the bootstrap has work to do.
        parent_id = ObjectId()
        db["normalized_concepts_annual"].docs.extend([
            {"_id": parent_id, "cik": "0001", "statement_type": "income",
             "concept": "us-gaap:Revenues", "label": "Revenue", "path": "001"},
            {"_id": ObjectId(), "cik": "0001", "statement_type": "income",
             "concept": "us-gaap:CostOfRevenue", "label": "Cost of revenue",
             "path": "001.001", "parent_concept": "us-gaap:Revenues",
             "parent_concept_id": parent_id},
        ])
        return db

    def test_concept_bootstrap_uses_only_concept_collections(self):
        from earnings_agents.integrations.normalize import (
            bootstrap_quarterly_concepts_from_annual,
        )

        db = self._db()
        with patch(
            "earnings_agents.integrations.normalize._get_client",
            return_value=_GuardedClient(db),
        ):
            copied = bootstrap_quarterly_concepts_from_annual("0001")

        self.assertEqual(copied, 2)
        self.assertTrue(db.accessed <= ALLOWED_COLLECTIONS, db.accessed)

    def test_backfill_resume_uses_only_value_collections(self):
        from earnings_agents.backfill import run_backfill

        db = self._db()
        db["concept_values_quarterly"].docs.append(
            {"cik": "0001", "accession_number": "ALREADY-DONE"}
        )

        class _Graph:
            def invoke(self, state):
                return {"status": "saved", "concept_metrics": {"c": 1}}

        candidates = [
            {"form": "8-K", "accession": "ALREADY-DONE", "filing_date": "2026-01-01"},
            {"form": "8-K", "accession": "NEW-ONE", "filing_date": "2026-01-02"},
        ]
        with patch(
            "earnings_agents.integrations.normalize._get_client",
            return_value=_GuardedClient(db),
        ):
            summary = run_backfill(
                graph=_Graph(), cik="0001", ticker="T", company_name="Test Co",
                candidates=candidates, printer=lambda *_: None,
                state_builder=lambda info, **kw: {
                    "status": "discovered", "accession_number": kw["accession"],
                },
            )

        self.assertEqual(summary["resumed"], 1)
        self.assertEqual(summary["saved"], 1)
        self.assertTrue(db.accessed <= ALLOWED_COLLECTIONS, db.accessed)

    def test_value_writer_uses_only_its_value_collection(self):
        from earnings_agents.agent.period import DetectedPeriod
        from earnings_agents.integrations.normalize import upsert_concept_values

        db = self._db()
        period = DetectedPeriod(
            period_type="quarterly", period_end=date(2026, 6, 30), quarter=2,
            fiscal_year=2026, period_label="Three Months Ended June 30, 2026",
        )
        cid = "5072e8e0c9d3f4a1b2c3d4e5"
        with patch(
            "earnings_agents.integrations.normalize._get_client",
            return_value=_GuardedClient(db),
        ):
            n = upsert_concept_values(
                cik="0001", company_name="Test Co",
                concept_metrics={cid: 100.0}, period=period,
                value_metadata_by_id={cid: {"currency": "USD"}},
            )
        self.assertEqual(n, 1)
        self.assertTrue(db.accessed <= ALLOWED_COLLECTIONS, db.accessed)

    def test_guarded_client_actually_catches_forbidden_access(self):
        """Control: the guard is not silently passing everything."""
        db = _GuardedDb()
        with self.assertRaises(ForbiddenCollectionError):
            db["earnings_backfill_ledger"]
        with self.assertRaises(ForbiddenCollectionError):
            db["quarterly_concept_bootstrap"]


if __name__ == "__main__":
    unittest.main()
