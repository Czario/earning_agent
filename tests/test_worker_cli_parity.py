"""CLI ⇄ worker parity — both paths must be 100% identical.

The worker consumes admin_backend's ``sec:filings:8k`` messages (payload:
``{filing_type, ticker, load_request_id, queued_at}`` — no accession/URL) and
must behave EXACTLY like ``uv run earnings --ticker X``:

  * same initial-state builder (``_build_initial_state`` → ``_build_8k_state``)
  * same graph (``build_graph()``: fetch → period agent → check_period →
    concepts → extraction → save)

This test locks the contract: for identical inputs the worker hands the graph
the identical state dict the CLI builds, and any divergence in the future
fails here.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_worker_payload_produces_state_identical_to_cli():
    from earnings_agents.cli import worker
    from earnings_agents.cli import earnings as cli

    fake_graph = MagicMock()
    fake_graph.invoke = MagicMock(return_value={
        "status": "saved",
        "concept_metrics": {"cid": 1.0},
        "sec_report_date": "2026-07-26",
        "_pending_replace": None,
    })

    payload = {"ticker": "amat", "filing_type": "8-K", "load_request_id": None}

    company = {
        "cik": "0000006951",
        "name": "APPLIED MATERIALS INC",
        "fiscal_year_end_month": 10,
        "fiscal_year_end_code": "1026",
    }
    edgar_result = (
        "https://www.sec.gov/Archives/edgar/data/6951/000162828026056699/exhibit991q32026earningsre.htm",
        [],
        "2026-07-27",           # raw EDGAR reportDate (informational)
        "0001628280-26-056699",  # accession
        "2026-07-27",            # filing date
        [{"exhibit": "EX-99.1", "description": "The Press Release",
          "url": "https://www.sec.gov/Archives/edgar/data/6951/000162828026056699/exhibit991q32026earningsre.htm"}],
    )

    with (
        patch("earnings_agents.cli.worker.WorkerProgressPublisher", return_value=MagicMock()),
        patch("earnings_agents.cli.worker.WorkerHeartbeat"),
        patch("earnings_agents.cli.worker.make_node_callback", return_value=MagicMock()),
        patch("earnings_agents.cli.worker.make_call_callback", return_value=MagicMock()),
        patch("earnings_agents.integrations.normalize.get_company_by_ticker",
              return_value=company),
        patch("earnings_agents.cli.earnings._has_existing_period_data", return_value=True),
        patch("earnings_agents.cli.earnings.get_latest_earnings_url",
              return_value=edgar_result),
    ):
        ok = worker._process_payload(fake_graph, payload)

    assert ok is True
    worker_state = fake_graph.invoke.call_args.args[0]

    # CLI for identical inputs builds its state through the same builder.
    with (
        patch("earnings_agents.cli.earnings._has_existing_period_data", return_value=True),
        patch("earnings_agents.cli.earnings.get_latest_earnings_url",
              return_value=edgar_result),
    ):
        cli_state = cli._build_initial_state(
            {
                "ticker": "AMAT",
                "company_name": "APPLIED MATERIALS INC",
                "cik": "0000006951",
            },
            printer=lambda *_: None,
        )

    # The exact keys that determine pipeline behavior must match:
    for key in (
        "ticker", "company_name", "status", "discovered_file_url",
        "supplemental_file_urls", "filing_date", "accession_number",
        "exhibit_meta",
    ):
        assert worker_state.get(key) == cli_state.get(key), (
            f"divergence on {key!r}: worker={worker_state.get(key)!r} "
            f"cli={cli_state.get(key)!r}"
        )

    # Period detection happens INSIDE the graph — the pre-graph state carries
    # no period fields in either path.
    for period_key in ("detected_period_type", "detected_quarter", "period_label",
                       "sec_report_date", "detected_period"):
        assert worker_state.get(period_key) is None
        assert cli_state.get(period_key) is None
