"""Regression tests: RSS-polled messages (carrying accession_number) must pin
the worker to the EXACT filing the poller saw — no submissions-API "latest
8-K" scan (that scan races with fresh filings and caused last-quarter
extraction).  Manual triggers (empty accession) keep their current behavior.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from earnings_agents.cli.worker import _process_payload

PYPL_ACC = "0001633917-26-000080"
PYPL_EX99 = (
    "https://www.sec.gov/Archives/edgar/data/1633917/000163391726000080/"
    "pypl2q-26earningsrelease.htm"
)
PYPL_EXHIBITS = [{"exhibit": "EX-99.1", "description": "EX-99.1", "url": PYPL_EX99}]


class FakePub:
    def __init__(self, *args, **kwargs):
        pass

    def publish(self, *args, **kwargs):
        pass

    def close(self):
        pass


class FakeHeartbeat:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _Recorder:
    """Graph stand-in capturing the state passed to graph.invoke()."""

    def __init__(self, seen: dict):
        self.seen = seen

    def invoke(self, state):
        self.seen["state"] = dict(state)
        return {"status": "saved", "concept_metrics": {}}


def _run_payload(payload):
    seen: dict = {}
    ok = _process_payload(_Recorder(seen), payload)
    return ok, seen


def _patch_stack(fn):
    """Patch all external deps of _process_payload for a hermetic test."""
    patches = [
        patch("earnings_agents.cli.worker._cleanup_temporary_filing"),
        patch("earnings_agents.cli.worker.make_call_callback"),
        patch("earnings_agents.cli.worker.make_node_callback"),
        patch("earnings_agents.cli.worker.WorkerHeartbeat", FakeHeartbeat),
        patch("earnings_agents.cli.worker.WorkerProgressPublisher", FakePub),
        patch("earnings_agents.cli.earnings.get_latest_earnings_url"),
        patch("earnings_agents.cli.earnings.get_exhibits_for_accession"),
        patch("earnings_agents.integrations.normalize.get_company_by_ticker"),
    ]
    for p in reversed(patches):
        fn = p(fn)
    return fn


class TestProcessPayloadPinnedAccession(unittest.TestCase):
    @_patch_stack
    def test_polled_message_pins_the_exact_accession(
        self, mock_company, mock_exhibits, mock_latest, *_rest
    ):
        mock_company.return_value = {
            "cik": "0001633917",
            "name": "PayPal Holdings, Inc.",
        }
        mock_exhibits.return_value = PYPL_EXHIBITS

        payload = {
            "filing_type": "8-K",
            "ticker": "PYPL",
            "load_request_id": "abc",
            "accession_number": PYPL_ACC,
            "filing_href": (
                "https://www.sec.gov/Archives/edgar/data/1633917/000163391726000080/"
                f"{PYPL_ACC}-index.htm"
            ),
            "filing_date": "2026-07-28",
            "filing_url": None,
            "temporary_filing_id": None,
        }
        ok, seen = _run_payload(payload)

        self.assertTrue(ok)
        state = seen["state"]
        self.assertEqual(state["ticker"], "PYPL")
        self.assertEqual(state["accession_number"], PYPL_ACC)
        self.assertEqual(state["discovered_file_url"], PYPL_EX99)
        # Pinned resolution used the exact accession — never the latest scan.
        mock_exhibits.assert_called_once_with("0001633917", PYPL_ACC)
        mock_latest.assert_not_called()

    @_patch_stack
    def test_manual_message_without_accession_keeps_latest_scan(
        self, mock_company, mock_exhibits, mock_latest, *_rest
    ):
        mock_company.return_value = {
            "cik": "0001633917",
            "name": "PayPal Holdings, Inc.",
        }
        mock_latest.return_value = ("url", [], "ACC-MANUAL", [])

        payload = {
            "filing_type": "8-K",
            "ticker": "PYPL",
            "load_request_id": "abc",
            "accession_number": "",  # manual trigger
            "filing_url": None,
            "temporary_filing_id": None,
        }
        ok, seen = _run_payload(payload)

        self.assertTrue(ok)
        self.assertEqual(seen["state"]["accession_number"], "ACC-MANUAL")
        mock_exhibits.assert_not_called()
        mock_latest.assert_called_once_with("0001633917")


if __name__ == "__main__":
    unittest.main()
