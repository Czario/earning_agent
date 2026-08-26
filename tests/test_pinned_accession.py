"""Tests for accession-pinned filing resolution (RSS polling path).

Regression coverage for the polling race: the 8-K worker must extract the
EXACT accession the RSS poller queued (resolved from that filing's index),
never re-scan the submissions API for "the latest 8-K" — the submissions
API lags fresh filings / returns empty ``items`` metadata, which previously
made polling extract the previous quarter's release.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from earnings_agents.cli.earnings import _build_8k_state
from earnings_agents.integrations.edgar import get_exhibits_for_accession

_NOPRINT = lambda _: None  # noqa: E731


class TestGetExhibitsForAccession(unittest.TestCase):
    @patch("earnings_agents.integrations.edgar._parse_filing_index")
    def test_returns_ex99_exhibits_when_present(self, mock_parse):
        exhibits = [
            {"exhibit": "EX-99.1", "description": "The Press Release", "url": "u1"},
            {"exhibit": "EX-99.2", "description": "Presentation", "url": "u2"},
        ]
        mock_parse.return_value = (exhibits, None)
        self.assertEqual(get_exhibits_for_accession("0000046619", "0000046619-26-000018"), exhibits)
        # CIK normalized to the archive path form (no leading zeros).
        mock_parse.assert_called_once_with("46619", "0000046619-26-000018", "000004661926000018")

    @patch("earnings_agents.integrations.edgar._parse_filing_index")
    def test_falls_back_to_primary_document_when_no_ex99(self, mock_parse):
        primary = {"exhibit": "8-K", "description": "8-K", "url": "u-primary"}
        mock_parse.return_value = ([], primary)
        result = get_exhibits_for_accession("46619", "0000046619-26-000018")
        self.assertEqual(result, [primary])

    @patch("earnings_agents.integrations.edgar._parse_filing_index")
    def test_returns_empty_when_no_documents(self, mock_parse):
        mock_parse.return_value = ([], None)
        self.assertEqual(get_exhibits_for_accession("46619", "0000046619-26-000018"), [])


class TestBuild8kStatePinnedAccession(unittest.TestCase):
    def _mock_exhibits(self):
        return [
            {"exhibit": "EX-99.1", "description": "The Press Release",
             "url": "https://www.sec.gov/Archives/edgar/data/46619/000004661926000018/a07312026ex991earningsrele.htm"},
            {"exhibit": "EX-99.2", "description": "Presentation",
             "url": "https://www.sec.gov/Archives/edgar/data/46619/000004661926000018/ex992.htm"},
        ]

    @patch("earnings_agents.cli.earnings.get_latest_earnings_url")
    @patch("earnings_agents.cli.earnings.get_exhibits_for_accession")
    def test_pinned_accession_uses_index_not_submissions_scan(
        self, mock_exhibits, mock_latest
    ):
        """Pinned accession → exhibits from that filing's index, NO latest scan."""
        mock_exhibits.return_value = self._mock_exhibits()

        state = _build_8k_state(
            "HEI", "HEICO CORP", "0000046619",
            accession="0000046619-26-000018",
            printer=_NOPRINT,
        )

        mock_exhibits.assert_called_once_with("0000046619", "0000046619-26-000018")
        mock_latest.assert_not_called()  # the race is gone
        self.assertEqual(state["status"], "discovered")
        self.assertEqual(
            state["discovered_file_url"],
            "https://www.sec.gov/Archives/edgar/data/46619/000004661926000018/a07312026ex991earningsrele.htm",
        )
        self.assertEqual(
            state["supplemental_file_urls"],
            ["https://www.sec.gov/Archives/edgar/data/46619/000004661926000018/ex992.htm"],
        )
        self.assertEqual(state["accession_number"], "0000046619-26-000018")
        self.assertEqual(len(state["exhibit_meta"]), 2)

    @patch("earnings_agents.cli.earnings.get_latest_earnings_url")
    @patch("earnings_agents.cli.earnings.get_exhibits_for_accession")
    def test_unresolvable_pinned_accession_falls_back_to_latest(
        self, mock_exhibits, mock_latest
    ):
        """Pinned accession with no documents → logged fallback to latest 8-K."""
        mock_exhibits.return_value = []
        mock_latest.return_value = (
            "https://www.sec.gov/Archives/edgar/data/46619/000004661926000018/a07312026ex991earningsrele.htm",
            [],
            "0000046619-26-000018",
            [{"exhibit": "EX-99.1", "description": "", "url": "https://www.sec.gov/Archives/edgar/data/46619/000004661926000018/a07312026ex991earningsrele.htm"}],
        )

        state = _build_8k_state(
            "HEI", "HEICO CORP", "0000046619",
            accession="0000046619-26-000099",  # stale / removed accession
            printer=_NOPRINT,
        )

        mock_exhibits.assert_called_once_with("0000046619", "0000046619-26-000099")
        mock_latest.assert_called_once_with("0000046619")
        self.assertEqual(state["status"], "discovered")
        self.assertEqual(state["accession_number"], "0000046619-26-000018")

    @patch("earnings_agents.cli.earnings.get_latest_earnings_url")
    @patch("earnings_agents.cli.earnings.get_exhibits_for_accession")
    def test_no_accession_keeps_latest_scan(self, mock_exhibits, mock_latest):
        """CLI / manual-trigger path (no accession) → unchanged latest scan."""
        mock_latest.return_value = ("url", [], "ACC-1", [])
        state = _build_8k_state(
            "HEI", "HEICO CORP", "0000046619",
            printer=_NOPRINT,
        )
        mock_exhibits.assert_not_called()
        mock_latest.assert_called_once_with("0000046619")
        self.assertEqual(state["status"], "discovered")


if __name__ == "__main__":
    unittest.main()
