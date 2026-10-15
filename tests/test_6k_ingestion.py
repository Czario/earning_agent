"""Phase 4 — 6-K (foreign private issuer) earnings ingestion.

Covers: generalized EDGAR discovery (8-K preference preserved, conservative
6-K candidate selection), source-form provenance through the shared state
builder, worker acceptance of 6-K payloads, and persistence of
``source_form_type`` on values and guidance.
"""
from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.cli.earnings import _build_8k_state, _forms_for_source_form
from earnings_agents.cli.worker import _SUPPORTED_FORMS, _process_payload
from earnings_agents.integrations.edgar import (
    EARNINGS_FORMS,
    _looks_like_earnings_6k,
    _parse_filing_index,
    get_exhibits_for_accession,
    get_latest_earnings_url,
    get_latest_filing_url,
)


# ── EDGAR discovery ─────────────────────────────────────────────────────────

def _submissions(forms, items=None, accessions=None, primary=None):
    n = len(forms)
    return {
        "filings": {
            "recent": {
                "form": list(forms),
                "items": list(items or [""] * n),
                "accessionNumber": list(
                    accessions or [f"0000000000-26-0000{i:02d}" for i in range(n)]
                ),
                "primaryDocument": list(primary or [""] * n),
            }
        }
    }


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestGeneralizedDiscovery(unittest.TestCase):
    def _run(self, submissions, exhibits_by_acc, cik="0000937966", forms=EARNINGS_FORMS):
        def _edgar_get(url, **kwargs):  # noqa: ARG001
            return _Resp(submissions)

        def _exhibits(cik_int, acc, acc_nodash):  # noqa: ARG001
            return list(exhibits_by_acc.get(acc, []))

        with patch("earnings_agents.integrations.edgar._edgar_get", side_effect=_edgar_get), \
             patch("earnings_agents.integrations.edgar._find_all_ex_99_exhibits",
                   side_effect=_exhibits):
            return get_latest_filing_url(cik, forms=forms)

    def test_8k_item_202_is_preferred(self):
        subs = _submissions(
            ["6-K", "8-K", "8-K"],
            items=["", "2.02", "7.01"],
            accessions=["acc-6k", "acc-8k-202", "acc-8k-other"],
        )
        url, supp, acc, exhibits, form = self._run(subs, {
            "acc-8k-202": [{"exhibit": "EX-99.1", "description": "Press Release",
                            "url": "https://x/8k202.htm"}],
        })
        self.assertEqual(acc, "acc-8k-202")
        self.assertEqual(form, "8-K")
        self.assertEqual(url, "https://x/8k202.htm")

    def test_latest_earnings_url_stays_8k_only(self):
        # A 6-K-only issuer must not be picked by the legacy 8-K wrapper.
        subs = _submissions(["6-K"], accessions=["acc-6k"])
        with patch("earnings_agents.integrations.edgar._edgar_get",
                   side_effect=lambda url, **k: _Resp(subs)), \
             patch("earnings_agents.integrations.edgar._find_all_ex_99_exhibits",
                   return_value=[{"exhibit": "EX-99.1", "description": "Earnings",
                                  "url": "https://x/6k.htm"}]):
            url, supp, acc, exhibits = get_latest_earnings_url("0000937966")
        self.assertIsNone(url)
        self.assertEqual(exhibits, [])

    def test_6k_with_earnings_like_exhibit_metadata_is_selected(self):
        subs = _submissions(["6-K"], accessions=["acc-6k"])
        url, supp, acc, exhibits, form = self._run(subs, {
            "acc-6k": [{"exhibit": "EX-99.1",
                        "description": "Q1 2026 quarterly results press release",
                        "url": "https://x/asml-q1-2026.htm"}],
        })
        self.assertEqual(acc, "acc-6k")
        self.assertEqual(form, "6-K")
        self.assertEqual(url, "https://x/asml-q1-2026.htm")

    def test_6k_with_unrelated_exhibit_is_not_selected(self):
        subs = _submissions(["6-K"], accessions=["acc-6k"])
        url, supp, acc, exhibits, form = self._run(subs, {
            "acc-6k": [{"exhibit": "EX-99.1",
                        "description": "Notice of annual general meeting of shareholders",
                        "url": "https://x/agm-notice.htm"}],
        })
        self.assertIsNone(url)
        self.assertIsNone(acc)
        self.assertIsNone(form)

    def test_6k_without_exhibits_is_not_selected(self):
        subs = _submissions(["6-K"], accessions=["acc-6k"])
        url, supp, acc, exhibits, form = self._run(subs, {})
        self.assertIsNone(url)

    def test_unsupported_form_yields_nothing(self):
        subs = _submissions(["10-Q"], accessions=["acc-10q"])
        url, supp, acc, exhibits, form = self._run(subs, {}, forms=("20-F",))
        self.assertIsNone(url)


class TestEarningsLikeMetadata(unittest.TestCase):
    def test_concatenated_press_release_filename_matches(self):
        # Real ASML exhibit filename (no separators) — word-boundary regex alone
        # cannot see "press release" here, so the compact-stem check catches it.
        self.assertTrue(_looks_like_earnings_6k("EX-99.1 pressreleasefinancialresul.htm"))

    def test_financial_statements_exhibit_matches(self):
        self.assertTrue(_looks_like_earnings_6k("EX-99.3 financialstatementsusgaa.htm"))

    def test_agm_results_is_not_an_earnings_release(self):
        self.assertFalse(_looks_like_earnings_6k("EX-99.1 asmldiscloses2026agmresults.htm"))

    def test_opaque_issuer_filename_is_not_assumed(self):
        # TSM earnings exhibits are opaque (a2q26e_withguidancexfinal.htm) — the
        # pipeline must NOT guess; pinned accession is required.
        self.assertFalse(_looks_like_earnings_6k("EX-99.1 a2q26e_withguidancexfinal.htm"))

    def test_periodic_report_filenames_match(self):
        # Ericsson names its earnings exhibits ``a26q2report.htm`` and puts the
        # report in the 6-K body when there is no EX-99 exhibit at all.
        self.assertTrue(_looks_like_earnings_6k("EX-99.1 a26q2report.htm"))
        self.assertTrue(_looks_like_earnings_6k("EX-99.1 a25q3report.htm"))
        self.assertTrue(_looks_like_earnings_6k("EX-99.1 Q2 2026 report"))
        self.assertTrue(_looks_like_earnings_6k("EX-99.1 second quarter report"))

    def test_weekly_buyback_notices_do_not_match(self):
        # Ericsson files a share-buyback 6-K every week; it must never be
        # mistaken for an earnings release.
        self.assertFalse(
            _looks_like_earnings_6k("EX-99.1 sharebuybacksinericssondur.htm")
        )
        self.assertFalse(
            _looks_like_earnings_6k("EX-99.1 ericssonannounceschangetot.htm")
        )

    def test_descriptive_text_matches(self):
        self.assertTrue(_looks_like_earnings_6k("EX-99.1 Q1 2026 quarterly results press release"))
        self.assertFalse(_looks_like_earnings_6k("EX-99.1 Notice of annual general meeting"))


_INDEX_6K_PRIMARY_ONLY = """
<html><body><table class="tableFile">
<tr><td>1</td><td>6-K</td><td><a href="/Archives/edgar/data/1046179/000104617926000459/tsm-monthend6kx20260724.htm">tsm-monthend6kx20260724.htm</a></td><td>6-K</td><td>42</td></tr>
<tr><td></td><td>Complete submission text file</td><td><a href="/Archives/edgar/data/1046179/000104617926000459/0001046179-26-000459.txt">x.txt</a></td><td></td><td>1</td></tr>
</table></body></html>
"""


class TestReportBodySizeSignal(unittest.TestCase):
    """A 6-K whose payload is a full periodic report is detectable from the
    submissions-API size alone — no EX-99 exhibit, opaque filename.

    Measured: Ericsson reports 1.8-5.1 MB vs notices 19-49 KB; TSM's earnings
    6-Ks are also report-sized with opaque exhibit names.
    """

    def test_report_sized_payloads_match(self):
        from earnings_agents.integrations.edgar import _looks_like_report_body_6k

        self.assertTrue(_looks_like_report_body_6k(5_086_000))   # Ericsson Q4
        self.assertTrue(_looks_like_report_body_6k(1_861_000))   # Ericsson Q2
        self.assertTrue(_looks_like_report_body_6k(3_418_000))   # Ericsson Q1

    def test_notice_sized_payloads_do_not_match(self):
        from earnings_agents.integrations.edgar import _looks_like_report_body_6k

        self.assertFalse(_looks_like_report_body_6k(49_000))     # buyback notice
        self.assertFalse(_looks_like_report_body_6k(19_000))
        self.assertFalse(_looks_like_report_body_6k(None))
        self.assertFalse(_looks_like_report_body_6k(0))


class TestPrimaryDocumentResolution(unittest.TestCase):
    def test_6k_primary_document_is_recognized(self):
        with patch("earnings_agents.integrations.edgar._edgar_get") as get:
            get.return_value = _Resp(1)
            get.return_value.text = _INDEX_6K_PRIMARY_ONLY
            exhibits, primary = _parse_filing_index(
                "1046179", "0001046179-26-000459", "000104617926000459"
            )
        self.assertEqual(exhibits, [])
        self.assertIsNotNone(primary)
        self.assertEqual(primary["exhibit"], "6-K")
        self.assertTrue(primary["url"].endswith("tsm-monthend6kx20260724.htm"))

    def test_exhibits_for_accession_falls_back_to_6k_primary(self):
        with patch(
            "earnings_agents.integrations.edgar._parse_filing_index",
            return_value=([], {"exhibit": "6-K", "description": "6-K",
                               "url": "https://x/tsm-6k.htm"}),
        ):
            docs = get_exhibits_for_accession("0001046179", "0001046179-26-000459")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["url"], "https://x/tsm-6k.htm")


class TestFormSelectionHelper(unittest.TestCase):
    def test_no_form_yields_nothing(self):
        # There is no AUTO mode and no default filing: an empty form means
        # "not supplied", and callers must refuse EDGAR discovery.
        self.assertEqual(_forms_for_source_form(None), ())
        self.assertEqual(_forms_for_source_form(""), ())

    def test_explicit_form_narrows_the_search(self):
        self.assertEqual(_forms_for_source_form("8-K"), ("8-K",))
        self.assertEqual(_forms_for_source_form("6-K"), ("6-K",))

    def test_6k_is_6k_only(self):
        self.assertEqual(_forms_for_source_form("6-k"), ("6-K",))


# ── shared state builder ────────────────────────────────────────────────────

class TestBuildStateSourceForm(unittest.TestCase):
    def test_pinned_6k_accession_records_source_form(self):
        exhibits = [{"exhibit": "EX-99.1", "description": "Results",
                     "url": "https://x/asml6k.htm"}]
        with patch(
            "earnings_agents.cli.earnings.get_exhibits_for_accession",
            return_value=exhibits,
        ), patch(
            "earnings_agents.cli.earnings.get_filing_form_type",
        ) as lookup:
            state = _build_8k_state(
                "ASML", "ASML HOLDING NV", "0000937966",
                accession="0000937966-26-000012",
                source_form_type="6-K",
                printer=lambda *_: None,
            )
        self.assertEqual(state["status"], "discovered")
        self.assertEqual(state["source_form_type"], "6-K")
        self.assertEqual(state["accession_number"], "0000937966-26-000012")
        self.assertEqual(state["discovered_file_url"], "https://x/asml6k.htm")
        # The caller supplied the form — no extra submissions lookup needed.
        lookup.assert_not_called()

    def test_pinned_accession_without_form_looks_it_up(self):
        exhibits = [{"exhibit": "EX-99.1", "description": "Results",
                     "url": "https://x/tsm6k.htm"}]
        with patch(
            "earnings_agents.cli.earnings.get_exhibits_for_accession",
            return_value=exhibits,
        ), patch(
            "earnings_agents.cli.earnings.get_filing_form_type",
            return_value="6-K",
        ) as lookup:
            state = _build_8k_state(
                "TSM", "TAIWAN SEMICONDUCTOR", "0001046179",
                accession="0001046179-26-000005",
                printer=lambda *_: None,
            )
        self.assertEqual(state["source_form_type"], "6-K")
        lookup.assert_called_once_with("0001046179", "0001046179-26-000005")

    def test_6k_auto_discovery_uses_generalized_lookup(self):
        with patch(
            "earnings_agents.cli.earnings.get_latest_earnings_url"
        ) as legacy, patch(
            "earnings_agents.cli.earnings.get_latest_filing_url",
            return_value=("https://x/asml6k.htm", [], "acc-6k",
                          [{"exhibit": "EX-99.1", "description": "Results",
                            "url": "https://x/asml6k.htm"}], "6-K"),
        ) as generalized:
            state = _build_8k_state(
                "ASML", "ASML HOLDING NV", "0000937966",
                source_form_type="6-K",
                printer=lambda *_: None,
            )
        legacy.assert_not_called()
        generalized.assert_called_once_with("0000937966", forms=("6-K",))
        self.assertEqual(state["source_form_type"], "6-K")

    def test_explicit_8k_keeps_the_legacy_call(self):
        with patch(
            "earnings_agents.cli.earnings.get_latest_earnings_url",
            return_value=("https://x/8k.htm", [], "acc-8k",
                          [{"exhibit": "EX-99.1", "description": "PR",
                            "url": "https://x/8k.htm"}]),
        ) as legacy, patch(
            "earnings_agents.cli.earnings.get_latest_filing_url"
        ) as generalized:
            state = _build_8k_state(
                "PYPL", "PayPal", "0001633917",
                source_form_type="8-K",
                printer=lambda *_: None,
            )
        generalized.assert_not_called()
        legacy.assert_called_once_with("0001633917")
        self.assertEqual(state["source_form_type"], "8-K")

    def test_no_form_refuses_edgar_discovery(self):
        # Nothing may be fetched without an explicit form.
        with patch(
            "earnings_agents.cli.earnings.get_latest_filing_url"
        ) as generalized, patch(
            "earnings_agents.cli.earnings.get_latest_earnings_url"
        ) as legacy:
            state = _build_8k_state(
                "ASML", "ASML HOLDING NV", "0000937966",
                printer=lambda *_: None,   # no source_form_type
            )
        generalized.assert_not_called()
        legacy.assert_not_called()
        self.assertEqual(state["status"], "failed")
        self.assertIn("source form", state["error"])

    def test_explicit_6k_uses_the_generalized_discovery(self):
        with patch(
            "earnings_agents.cli.earnings.get_latest_filing_url",
            return_value=("https://x/6k.htm", [], "acc-6k",
                          [{"exhibit": "EX-99.1", "description": "Results",
                            "url": "https://x/6k.htm"}], "6-K"),
        ) as generalized, patch(
            "earnings_agents.cli.earnings.get_latest_earnings_url"
        ) as legacy:
            state = _build_8k_state(
                "ASML", "ASML HOLDING NV", "0000937966",
                source_form_type="6-K",
                printer=lambda *_: None,
            )
        legacy.assert_not_called()
        generalized.assert_called_once_with("0000937966", forms=("6-K",))
        self.assertEqual(state["source_form_type"], "6-K")

    def test_manual_edgar_url_derives_the_source_form(self):
        """No EDGAR discovery for a manual URL — recover the form from the
        accession embedded in the URL instead of guessing."""
        with patch(
            "earnings_agents.cli.earnings.get_filing_form_type",
            return_value="6-K",
        ) as lookup:
            state = _build_8k_state(
                "ASML", "ASML HOLDING NV", "0000937966",
                filing_url=(
                    "https://www.sec.gov/Archives/edgar/data/937966/"
                    "000162828026048235/pressreleasefinancialresul.htm"
                ),
                printer=lambda *_: None,
            )
        self.assertEqual(state["source_form_type"], "6-K")
        self.assertEqual(state["accession_number"], "0001628280-26-048235")
        lookup.assert_called_once_with("0000937966", "0001628280-26-048235")

    def test_manual_url_without_accession_leaves_form_unset(self):
        with patch(
            "earnings_agents.cli.earnings.get_filing_form_type"
        ) as lookup:
            state = _build_8k_state(
                "ASML", "ASML HOLDING NV", "0000937966",
                filing_url="https://ir.asml.com/q3-2026-results.pdf",
                printer=lambda *_: None,
            )
        lookup.assert_not_called()
        self.assertNotIn("source_form_type", state)

    def test_local_pdf_with_accession_filename_derives_the_form(self):
        with patch(
            "earnings_agents.cli.earnings.get_filing_form_type",
            return_value="8-K",
        ) as lookup:
            state = _build_8k_state(
                "AAPL", "Apple Inc.", "0000320193",
                local_filing_path="/uploads/0000320193-26-000011.pdf",
                printer=lambda *_: None,
            )
        self.assertEqual(state["source_form_type"], "8-K")
        lookup.assert_called_once_with("0000320193", "0000320193-26-000011")

    def test_no_filing_error_names_the_requested_form(self):
        with patch(
            "earnings_agents.cli.earnings.get_latest_filing_url",
            return_value=(None, [], None, [], None),
        ):
            state = _build_8k_state(
                "ASML", "ASML HOLDING NV", "0000937966",
                source_form_type="6-K",
                printer=lambda *_: None,
            )
        self.assertEqual(state["status"], "failed")
        self.assertIn("6-K", state["error"])


# ── worker acceptance ───────────────────────────────────────────────────────

class _Recorder:
    def __init__(self, seen):
        self.seen = seen

    def invoke(self, state):
        self.seen["state"] = dict(state)
        return {"status": "saved", "concept_metrics": {}}


class TestWorkerAccepts6K(unittest.TestCase):
    def test_supported_forms(self):
        self.assertIn("8-K", _SUPPORTED_FORMS)
        self.assertIn("6-K", _SUPPORTED_FORMS)

    def test_6k_payload_sets_source_form_type(self):
        seen: dict = {}
        exhibits = [{"exhibit": "EX-99.1", "description": "Results",
                     "url": "https://x/asml6k.htm"}]
        with patch("earnings_agents.cli.worker.WorkerProgressPublisher"), \
             patch("earnings_agents.cli.worker.WorkerHeartbeat"), \
             patch("earnings_agents.cli.worker.make_call_callback"), \
             patch("earnings_agents.cli.worker.make_node_callback"), \
             patch("earnings_agents.cli.worker._cleanup_temporary_filing"), \
             patch(
                 "earnings_agents.integrations.normalize.get_company_by_ticker",
                 return_value={"cik": "0000937966", "name": "ASML HOLDING NV"},
             ), \
             patch(
                 "earnings_agents.cli.earnings.get_exhibits_for_accession",
                 return_value=exhibits,
             ):
            ok = _process_payload(
                _Recorder(seen),
                {
                    "filing_type": "6-K",
                    "ticker": "ASML",
                    "accession_number": "0000937966-26-000012",
                    "temporary_filing_id": None,
                },
            )
        self.assertTrue(ok)
        self.assertEqual(seen["state"]["source_form_type"], "6-K")


# ── persistence provenance ──────────────────────────────────────────────────

class _Collection:
    def __init__(self):
        self.ops = []

    def bulk_write(self, ops, ordered=False):  # noqa: ARG002
        self.ops = list(ops)

    def delete_many(self, filt):  # noqa: ARG002
        return MagicMock(deleted_count=0)


class TestSourceFormPersistence(unittest.TestCase):
    def test_value_doc_form_type_is_the_real_form_only(self):
        from earnings_agents.integrations.normalize import upsert_concept_values

        period = DetectedPeriod(
            period_type="quarterly", period_end=date(2026, 3, 31), quarter=1,
            fiscal_year=2026, period_label="Three Months Ended March 31, 2026",
        )
        cid = "5072e8e0c9d3f4a1b2c3d4e5"
        col = _Collection()
        db = MagicMock()
        db.__getitem__.return_value = col
        client = MagicMock()
        client.__getitem__.return_value = db

        import earnings_agents.integrations.normalize as _norm
        with patch("earnings_agents.integrations.normalize._get_client",
                   return_value=client):
            n = upsert_concept_values(
                cik="0000937966", company_name="ASML HOLDING NV",
                concept_metrics={cid: 1000.0}, period=period,
                value_metadata_by_id={cid: {"currency": "EUR"}},
                accession_number="0000937966-26-000012",
                source_form_type="6-K",
            )
        self.assertEqual(n, 1)
        doc = col.ops[0]._doc["$set"]
        # form_type records the REAL filed form, and there is no duplicate
        # provenance field.
        self.assertEqual(doc["form_type"], "6-K")
        self.assertNotIn("source_form_type", doc)
        self.assertEqual(doc["currency"], "EUR")

    def test_guidance_uses_source_form(self):
        from earnings_agents.nodes.save_guidance import save_guidance_node

        state = {
            "ticker": "ASML",
            "cik": "0000937966",
            "source_form_type": "6-K",
            "accession_number": "0000937966-26-000012",
            "guidance_records": [{"metric": "revenue"}],
            "detected_period": {
                "period_type": "quarterly", "period_end": "2026-03-31",
                "quarter": 1, "fiscal_year": 2026,
                "period_label": "Three Months Ended March 31, 2026",
            },
        }
        with patch(
            "earnings_agents.integrations.guidance.upsert_guidance_records",
            return_value={"n_records": 1, "upserted": 1, "skipped_manual": 0,
                          "demoted": 0},
        ) as upsert, patch(
            "earnings_agents.integrations.guidance.score_guidance_for_cik",
            return_value={"checked": 0, "scored": 0, "unresolved": 0,
                          "errors": 0, "currency_mismatch": 0},
        ):
            save_guidance_node(state)
        self.assertEqual(upsert.call_args.kwargs["form_type"], "6-K")


if __name__ == "__main__":
    unittest.main()
