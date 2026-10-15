"""Phase 5 — historical earnings backfill (manifest + resumable runner)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from earnings_agents.backfill import processed_accessions, run_backfill
from earnings_agents.hooks import _thread_local
from earnings_agents.integrations.edgar import (
    build_earnings_manifest,
    fiscal_year_of,
    get_company_filings,
)


# ── EDGAR historical submissions ────────────────────────────────────────────

def _page(forms, dates, accessions=None, items=None, report_dates=None):
    n = len(forms)
    return {
        "form": list(forms),
        "filingDate": list(dates),
        "accessionNumber": list(accessions or [f"ACC-{i}" for i in range(n)]),
        "items": list(items or [""] * n),
        "reportDate": list(report_dates or [""] * n),
        "primaryDocument": [""] * n,
    }


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestHistoricalSubmissions(unittest.TestCase):
    def _run(self, submissions, pages, **kwargs):
        def _get(url, **kw):  # noqa: ARG001
            for name, payload in pages.items():
                if name in url:
                    return _Resp(payload)
            return _Resp(submissions)

        with patch("earnings_agents.integrations.edgar._edgar_get", side_effect=_get):
            return get_company_filings("0000000001", **kwargs)

    def test_reads_recent_and_historical_pages(self):
        submissions = {
            "filings": {
                "recent": _page(["8-K"], ["2026-02-01"]),
                "files": [
                    {"name": "old-1.json", "filingTo": "2022-12-31"},
                    {"name": "old-2.json", "filingTo": "2020-12-31"},
                ],
            }
        }
        pages = {
            "old-1.json": _page(["8-K"], ["2022-05-01"]),
            "old-2.json": _page(["8-K"], ["2020-05-01"]),
        }
        filings = self._run(submissions, pages, forms=("8-K",))
        self.assertEqual(
            [f["filing_date"] for f in filings], ["2026-02-01", "2022-05-01", "2020-05-01"]
        )

    def test_year_range_filters_and_skips_old_pages_without_fetching(self):
        submissions = {
            "filings": {
                "recent": _page(["8-K"], ["2026-02-01"]),
                "files": [{"name": "old-2.json", "filingTo": "2020-12-31"}],
            }
        }
        pages = {"old-2.json": _page(["8-K"], ["2020-05-01"])}
        fetched: list[str] = []

        def _get(url, **kw):  # noqa: ARG001
            if "old-2.json" in url:
                fetched.append(url)
                return _Resp(pages["old-2.json"])
            return _Resp(submissions)

        with patch("earnings_agents.integrations.edgar._edgar_get", side_effect=_get):
            filings = get_company_filings(
                "0000000001", forms=("8-K",), start_year=2025
            )
        self.assertEqual([f["filing_date"] for f in filings], ["2026-02-01"])
        # The entirely-too-old page was skipped without an HTTP request.
        self.assertEqual(fetched, [])

    def test_form_filter(self):
        submissions = {"filings": {"recent": _page(["10-Q", "8-K", "6-K"],
                                                    ["2026-01-01", "2026-02-01", "2026-03-01"])}}
        filings = self._run(submissions, {}, forms=("6-K",))
        self.assertEqual([f["form"] for f in filings], ["6-K"])


class TestFiscalYear(unittest.TestCase):
    def test_report_date_year_is_the_fiscal_year(self):
        self.assertEqual(
            fiscal_year_of({"report_date": "2025-12-31", "filing_date": "2026-01-28"}),
            2025,
        )

    def test_falls_back_to_filing_year_when_report_date_missing(self):
        self.assertEqual(
            fiscal_year_of({"report_date": "", "filing_date": "2026-02-01"}), 2026
        )

    def test_no_dates(self):
        self.assertIsNone(fiscal_year_of({"report_date": "", "filing_date": ""}))


class TestBuildEarningsManifest(unittest.TestCase):
    def _manifest(self, filings, exhibits=None, primary=None, **kwargs):
        """Manifest with the EDGAR filing index stubbed out.

        ``exhibits`` maps accession → EX-99 exhibit list; ``primary`` maps
        accession → primary-document dict (for the 6-K body-only case).
        """
        def _index(_cik_int, acc, _acc_nodash):
            return (
                list((exhibits or {}).get(acc, [])),
                (primary or {}).get(acc),
            )

        with patch(
            "earnings_agents.integrations.edgar.get_company_filings",
            return_value=filings,
        ), patch(
            "earnings_agents.integrations.edgar._parse_filing_index",
            side_effect=_index,
        ):
            return build_earnings_manifest("0000000001", **kwargs)

    def test_year_bounds_are_fiscal_years_not_filing_years(self):
        # FY2024 report filed in 2025 must be EXCLUDED from --from-year 2025.
        # get_company_filings returns newest-first (mirrored here).
        filings = [
            {"form": "8-K", "accession": "FY25-Q4", "filing_date": "2026-01-28",
             "report_date": "2025-12-31", "items": "2.02", "primary_document": ""},
            {"form": "8-K", "accession": "FY25-Q1", "filing_date": "2025-04-16",
             "report_date": "2025-03-30", "items": "2.02", "primary_document": ""},
            {"form": "8-K", "accession": "FY24", "filing_date": "2025-01-29",
             "report_date": "2024-12-31", "items": "2.02", "primary_document": ""},
        ]
        manifest = self._manifest(filings, start_year=2025, end_year=2025)
        self.assertEqual([m["accession"] for m in manifest], ["FY25-Q4", "FY25-Q1"])
        self.assertTrue(all(m["fiscal_year"] == 2025 for m in manifest))

    def test_entry_carries_fiscal_year_and_stays_newest_first(self):
        filings = [
            {"form": "8-K", "accession": "NEW", "filing_date": "2026-07-20",
             "report_date": "2026-06-30", "items": "2.02", "primary_document": ""},
            {"form": "8-K", "accession": "OLD", "filing_date": "2026-04-20",
             "report_date": "2026-03-31", "items": "2.02", "primary_document": ""},
        ]
        manifest = self._manifest(filings, start_year=2026)
        self.assertEqual([m["accession"] for m in manifest], ["NEW", "OLD"])
        self.assertEqual(manifest[0]["fiscal_year"], 2026)

    def test_8k_item_202_is_high_confidence(self):
        filings = [
            {"form": "8-K", "accession": "A1", "filing_date": "2026-01-01",
             "report_date": "", "items": "2.02,9.01", "primary_document": ""},
            {"form": "8-K", "accession": "A2", "filing_date": "2025-12-01",
             "report_date": "", "items": "5.02", "primary_document": ""},
        ]
        manifest = self._manifest(filings)
        self.assertEqual([m["accession"] for m in manifest], ["A1"])
        self.assertEqual(manifest[0]["confidence"], "high")
        self.assertEqual(manifest[0]["reason"], "item_2.02")

    def test_low_confidence_8k_only_with_flag(self):
        filings = [
            {"form": "8-K", "accession": "A2", "filing_date": "2025-12-01",
             "report_date": "", "items": "5.02", "primary_document": ""},
        ]
        self.assertEqual(self._manifest(filings), [])
        manifest = self._manifest(filings, include_low_confidence=True)
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["confidence"], "low")

    def test_6k_requires_earnings_like_exhibit_metadata(self):
        filings = [
            {"form": "6-K", "accession": "E1", "filing_date": "2026-01-01",
             "report_date": "2025-12-31", "items": "", "primary_document": ""},
            {"form": "6-K", "accession": "X1", "filing_date": "2025-11-01",
             "report_date": "", "items": "", "primary_document": ""},
            {"form": "6-K", "accession": "X2", "filing_date": "2025-10-01",
             "report_date": "", "items": "", "primary_document": ""},
        ]
        exhibits = {
            "E1": [{"exhibit": "EX-99.1", "description": "EX-99.1",
                    "url": "https://x/pressreleasefinancialresul.htm"}],
            "X1": [{"exhibit": "EX-99.1", "description": "EX-99.1",
                    "url": "https://x/noticeannualgeneralmeeti.htm"}],
            "X2": [],
        }
        manifest = self._manifest(filings, exhibits=exhibits, forms=("6-K",))
        self.assertEqual([m["accession"] for m in manifest], ["E1"])
        self.assertEqual(manifest[0]["reason"], "earnings_exhibit_metadata")

    def test_report_sized_6k_body_is_found_without_flags(self):
        """Ericsson/TSM put the report in the 6-K body with an opaque filename.

        The payload size is the only metadata signal, and it must be enough to
        surface the candidate with NO accession and NO --include-unverified.
        """
        filings = [
            {"form": "6-K", "accession": "BIG", "filing_date": "2026-04-17",
             "report_date": "2026-04-17", "items": "", "primary_document": "d6k.htm",
             "size": 3_418_000},
            # Weekly notice 6-K must NOT be offered.
            {"form": "6-K", "accession": "SMALL", "filing_date": "2026-04-10",
             "report_date": "2026-04-10", "items": "", "primary_document": "n6k.htm",
             "size": 49_000},
        ]
        manifest = self._manifest(filings, forms=("6-K",))
        self.assertEqual([m["accession"] for m in manifest], ["BIG"])
        self.assertEqual(manifest[0]["confidence"], "high")
        self.assertEqual(manifest[0]["reason"], "large_6k_report_body")

    def test_primary_document_only_6k_is_low_confidence_and_opt_in(self):
        """Ericsson puts the report in the 6-K body with no EX-99 exhibit."""
        filings = [
            {"form": "6-K", "accession": "P1", "filing_date": "2026-04-17",
             "report_date": "2026-04-17", "items": "", "primary_document": "d6k.htm"},
        ]
        primary = {"P1": {"exhibit": "6-K", "description": "6-K", "url": "https://x/d6k.htm"}}

        # Safe by default: no earnings metadata signal → not offered.
        self.assertEqual(self._manifest(filings, primary=primary, forms=("6-K",)), [])

        # Explicitly opted in → offered as LOW confidence for review.
        manifest = self._manifest(
            filings, primary=primary, forms=("6-K",), include_low_confidence=True,
        )
        self.assertEqual([m["accession"] for m in manifest], ["P1"])
        self.assertEqual(manifest[0]["confidence"], "low")
        self.assertEqual(manifest[0]["reason"], "6k_primary_document_only")

    def test_fiscal_year_and_quarter_filter_is_report_date_based(self):
        filings = [
            {"form": "8-K", "accession": "Q1", "filing_date": "2026-04-20",
             "report_date": "2026-03-31", "items": "2.02", "primary_document": ""},
            {"form": "8-K", "accession": "Q2", "filing_date": "2026-07-20",
             "report_date": "2026-06-30", "items": "2.02", "primary_document": ""},
        ]
        manifest = self._manifest(filings, fiscal_year=2026, fiscal_quarter=2)
        self.assertEqual([m["accession"] for m in manifest], ["Q2"])

    def test_limit_caps_the_manifest(self):
        filings = [
            {"form": "8-K", "accession": f"A{i}", "filing_date": f"2026-0{i}-01",
             "report_date": "", "items": "2.02", "primary_document": ""}
            for i in range(1, 5)
        ]
        manifest = self._manifest(filings, limit=2)
        self.assertEqual(len(manifest), 2)


# ── Resumable runner ────────────────────────────────────────────────────────

class _FakeValues:
    """Stand-in for concept_values_{quarterly,annual} — the resume source."""

    def __init__(self, accessions=None, cik="0001"):
        self.docs = [{"cik": cik, "accession_number": a} for a in (accessions or [])]

    def distinct(self, field, query):  # noqa: ARG002
        return sorted({
            d.get(field) for d in self.docs
            if all(d.get(k) == v for k, v in query.items()) and d.get(field)
        })

    def find(self, query, projection=None):  # noqa: ARG002
        return [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]


def _value_db(accessions=(), cik="0001"):
    """Fake normalize_data DB exposing the two value collections."""
    return {
        "concept_values_quarterly": _FakeValues(accessions, cik),
        "concept_values_annual": _FakeValues((), cik),
    }


class _Graph:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def invoke(self, state):
        self.calls.append(state)
        return self.outcomes.pop(0)


def _candidates(*accessions):
    return [
        {"form": "8-K", "accession": acc, "filing_date": "2026-01-01",
         "confidence": "high", "reason": "item_2.02"}
        for acc in accessions
    ]


def _builder(info, printer=None, accession=None, source_form_type=None, **kw):  # noqa: ARG001
    return {"status": "discovered", "accession_number": accession,
            "source_form_type": source_form_type, "cik": info["cik"],
            "ticker": info["ticker"]}


class TestRunBackfill(unittest.TestCase):
    def _run(self, graph, candidates, values=None, **kwargs):
        db = values if values is not None else _value_db()
        with patch("earnings_agents.integrations.normalize._get_client",
                   return_value={"normalize_data": db}):
            return run_backfill(
                graph=graph, cik="0001", ticker="T", company_name="Test Co",
                candidates=candidates, printer=lambda *_: None,
                state_builder=_builder, **kwargs,
            )

    def test_processes_each_candidate_and_values_become_the_resume_record(self):
        graph = _Graph([{"status": "saved"}, {"status": "saved"}])
        summary = self._run(graph, _candidates("A1", "A2"))
        self.assertEqual(summary["saved"], 2)
        self.assertEqual(len(graph.calls), 2)
        self.assertEqual(summary["accessions_completed"], ["A1", "A2"])
        self.assertEqual(summary["accessions_failed"], [])
        self.assertTrue(all(state.get("quarterly_only") for state in graph.calls))

    def test_resume_skips_accessions_already_saved_in_value_collections(self):
        graph = _Graph([{"status": "saved"}])
        summary = self._run(
            graph, _candidates("A1", "A2"), _value_db(["A1"]),
        )
        self.assertEqual(summary["resumed"], 1)
        self.assertEqual(summary["saved"], 1)
        self.assertEqual(len(graph.calls), 1)
        self.assertEqual(graph.calls[0]["accession_number"], "A2")

    def test_resume_reads_the_annual_collection_too(self):
        db = _value_db()
        db["concept_values_annual"] = _FakeValues(["A1"])
        graph = _Graph([{"status": "saved"}])
        summary = self._run(graph, _candidates("A1", "A2"), db)
        self.assertEqual(summary["resumed"], 1)
        self.assertEqual(summary["saved"], 1)

    def test_processed_accessions_reads_both_value_collections(self):
        db = _value_db(["A1"])
        db["concept_values_annual"] = _FakeValues(["A9"])
        with patch("earnings_agents.integrations.normalize._get_client",
                   return_value={"normalize_data": db}):
            found = processed_accessions("0001")
        self.assertEqual(found, {"A1", "A9"})

    def test_reload_reprocesses_already_saved_accessions(self):
        graph = _Graph([{"status": "saved"}, {"status": "saved"}])
        summary = self._run(
            graph, _candidates("A1", "A2"), _value_db(["A1"]), reload=True,
        )
        self.assertEqual(summary["saved"], 2)
        self.assertEqual(summary["resumed"], 0)
        self.assertEqual(len(graph.calls), 2)

    def test_dry_run_never_invokes_graph_and_is_not_resumed(self):
        graph = _Graph([])
        summary = self._run(graph, _candidates("A1"), _value_db(["A1"]), dry_run=True)
        self.assertEqual(graph.calls, [])
        self.assertEqual(summary["skipped"], 1)

    def test_failed_filing_writes_nothing_so_resume_retries_it(self):
        graph = _Graph([
            {"status": "failed", "error": "period agent failed"},
            {"status": "saved"},
        ])
        summary = self._run(graph, _candidates("A1", "A2"))
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["saved"], 1)
        self.assertEqual(summary["accessions_failed"], ["A1"])
        self.assertEqual(summary["accessions_completed"], ["A2"])

    def test_graph_exception_is_isolated_per_filing(self):
        class _Boom(_Graph):
            def invoke(self, state):
                self.calls.append(state)
                if len(self.calls) == 1:
                    raise RuntimeError("provider down")
                return {"status": "saved"}

        summary = self._run(_Boom([]), _candidates("A1", "A2"))
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["saved"], 1)

    def test_state_build_failure_is_isolated(self):
        graph = _Graph([])

        def _bad_builder(info, **kw):  # noqa: ARG001
            raise RuntimeError("no documents")

        with patch("earnings_agents.integrations.normalize._get_client",
                   return_value={"normalize_data": _value_db()}):
            summary = run_backfill(
                graph=graph, cik="0001", ticker="T", company_name="Test Co",
                candidates=_candidates("A1"), printer=lambda *_: None,
                state_builder=_bad_builder,
            )
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["accessions_failed"], ["A1"])

    def test_presenter_streams_pipeline_lines_and_node_stage(self):
        """The shared RunPresenter prints the same lines a live run prints."""
        from earnings_agents.cli.run_ui import RunPresenter

        presenter = RunPresenter("T")

        class _GraphWithEvents(_Graph):
            def invoke(self, state):
                from earnings_agents.hooks import report_call

                # Simulate a node lifecycle plus a pipeline [db] line, exactly
                # as with_hooks/report_call do during a real run.
                cb = getattr(_thread_local, "node_callback", None)
                if cb:
                    cb("fetch_filing_node", "start", state.get("ticker", "?"))
                report_call("  [db]  query normalized_concepts_quarterly")
                if cb:
                    cb("fetch_filing_node", "end", state.get("ticker", "?"), None, 12.0)
                return super().invoke(state)

        rendered: list[str] = []

        class _Console:
            width = 120

            def print(self, markup):
                rendered.append(str(markup))

        presenter._console = _Console()
        self._run(
            _GraphWithEvents([{"status": "saved"}]),
            _candidates("A1"),
            presenter=presenter,
        )
        self.assertTrue(any("[db]" in line for line in rendered))
        self.assertTrue(any("fetch" in line for line in rendered))
        self.assertEqual(presenter.llm_calls, 0)

    def test_callbacks_are_cleared_after_each_filing(self):
        from earnings_agents.cli.run_ui import RunPresenter
        from earnings_agents.hooks import get_call_callback

        self._run(
            _Graph([{"status": "saved"}, {"status": "saved"}]),
            _candidates("A1", "A2"),
            presenter=RunPresenter("T"),
        )
        self.assertIsNone(get_call_callback())
        self.assertIsNone(getattr(_thread_local, "node_callback", None))
        self.assertIsNone(getattr(_thread_local, "detail_callback", None))

    def test_filing_lifecycle_hooks_fire_on_every_path(self):
        started: list[str] = []
        ended: list[str] = []
        graph = _Graph([{"status": "failed"}, {"status": "saved"}])

        def _start(cand, index, total):
            started.append(cand["accession"])

        def _end(cand, result):
            ended.append(f"{cand['accession']}:{result.get('status')}")

        self._run(
            graph, _candidates("A1", "A2"),
            on_filing_start=_start, on_filing_end=_end,
        )
        self.assertEqual(started, ["A1", "A2"])
        self.assertEqual(ended, ["A1:failed", "A2:saved"])

    def test_candidates_are_processed_in_given_order(self):
        graph = _Graph([{"status": "saved"}, {"status": "saved"}])
        self._run(graph, _candidates("NEWEST", "OLDEST"))
        self.assertEqual(
            [s.get("accession_number") for s in graph.calls], ["NEWEST", "OLDEST"]
        )

    def test_resume_survives_database_read_failure(self):
        graph = _Graph([{"status": "saved"}])
        with patch("earnings_agents.backfill._value_collections",
                   side_effect=RuntimeError("mongo down")):
            summary = run_backfill(
                graph=graph, cik="0001", ticker="T", company_name="Test Co",
                candidates=_candidates("A1"), printer=lambda *_: None,
                state_builder=_builder,
            )
        # DB unavailable → nothing is considered processed → the filing runs.
        self.assertEqual(summary["saved"], 1)
        self.assertEqual(summary["resumed"], 0)


if __name__ == "__main__":
    unittest.main()
