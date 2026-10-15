"""Historical earnings-filing backfill runner.

Phase 5.  Runs a bounded set of *already selected* filings (see
``integrations/edgar.py::build_earnings_manifest``) through the SAME graph the
CLI/worker use — one filing per graph invocation, exactly like a live run.

Guarantees:

* **No period inference.**  The manifest's ``filing_date``/``report_date`` only
  choose which filings to process; the period agent decides the reporting
  period from each document.
* **Idempotent / resumable WITHOUT a side ledger.**  The saved value documents
  already record the filing they came from (``accession_number``), so resume
  reads that from ``concept_values_quarterly`` / ``concept_values_annual`` —
  there is no bookkeeping collection to create or maintain.  A filing whose
  accession appears in either collection was processed successfully and is
  skipped; ``reload`` re-runs it regardless.
* **No accidental period deletion.**  Nothing here deletes data.  Re-processing
  a period goes through the existing atomic write-first replace inside
  ``upsert_concept_values``, so a failed run can never leave a period empty.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# The two collections that hold saved extraction results.  A filing is
# considered already processed when its accession_number is present in either.
_VALUE_COLLECTIONS = ("concept_values_quarterly", "concept_values_annual")


def _value_collections() -> list[Any]:
    """Return the saved-value collections in the configured normalize DB.

    Imported lazily so tests can patch the client without a live connection at
    module import time.
    """
    from earnings_agents.integrations.normalize import _NORMALIZE_DB, _get_client
    db = _get_client()[_NORMALIZE_DB]
    return [db[name] for name in _VALUE_COLLECTIONS]


def processed_accessions(cik: str) -> set[str]:
    """Return the accessions already saved for *cik* (resume source of truth).

    Derived from ``accession_number`` on the saved value documents — the write
    path already stamps every value with the filing it came from, so no extra
    bookkeeping collection is needed.

    Note: a filing that produced NO persisted values (every metric missing, or
    every monetary value dropped for an unconfirmed currency) leaves no record
    and is therefore retried on the next resume — a safe, idempotent outcome.

    Never raises — a database read failure degrades to "nothing processed"
    (the worst case is redundant LLM work, never a lost or duplicated value).
    """
    out: set[str] = set()
    try:
        for col in _value_collections():
            for accession in col.distinct("accession_number", {"cik": cik}):
                if accession:
                    out.add(str(accession))
    except Exception as exc:  # noqa: BLE001 — resume is an optimization
        logger.warning("backfill resume lookup failed for CIK %s: %s", cik, exc)
        return set()
    return out


def run_backfill(
    *,
    graph: Any,
    cik: str,
    ticker: str,
    company_name: str,
    candidates: list[dict],
    target_statements: list[str] | None = None,
    resume: bool = True,
    reload: bool = False,
    dry_run: bool = False,
    run_id: str | None = None,
    printer: Callable[[str], None] = print,
    state_builder: Callable[..., dict] | None = None,
    presenter: Any | None = None,
    on_filing_start: Callable[[dict, int, int], None] | None = None,
    on_filing_end: Callable[[dict, dict], None] | None = None,
) -> dict[str, Any]:
    """Process *candidates* (from the manifest) through the earnings graph.

    Candidates are processed in the order given (the manifest is newest-first,
    so a default backfill runs newest → oldest, matching the single-filing
    experience).

    Resume: when *resume* is set (and not *reload*), any candidate whose
    accession already appears on a saved value document is skipped — the value
    collections *are* the resume record, so no ledger collection is written.

    *presenter* is a ``cli.run_ui.RunPresenter`` — the SAME object the
    single-ticker CLI uses.  Its hooks callbacks are installed around each graph
    invocation, so a backfill prints byte-identical stage / step-summary /
    ``[llm]`` / ``[tool]`` / ``[db]`` lines and the same ``(N LLM calls)``
    footer.  ``on_filing_start`` / ``on_filing_end`` let the caller bind the
    presenter to a fresh progress row per filing.

    Returns a summary dict with per-status counts and the accessions processed.
    The graph is invoked once per filing — identical to a live CLI run — so all
    existing guardrails (period agent authority, save gate, atomic replace)
    apply unchanged.
    """
    if state_builder is None:
        from earnings_agents.cli.earnings import _build_initial_state
        state_builder = _build_initial_state
    from earnings_agents.cli.run_ui import RunPresenter as _RunPresenter

    info = {"ticker": ticker, "company_name": company_name, "cik": cik}
    processed: set[str] = set()
    if resume and not reload and not dry_run:
        processed = processed_accessions(cik)

    summary: dict[str, Any] = {
        "cik": cik,
        "ticker": ticker,
        "run_id": run_id,
        "total": len(candidates),
        "dry_run": dry_run,
        "saved": 0,
        "failed": 0,
        "skipped": 0,
        "resumed": 0,
        "accessions_completed": [],
        "accessions_failed": [],
    }

    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        accession = str(candidate.get("accession") or "")
        form = str(candidate.get("form") or "")
        filing_date = str(candidate.get("filing_date") or "")
        prefix = f"[{index}/{total}] {ticker or cik} {form} {filing_date} {accession}"
        final: dict[str, Any] = {}

        if not accession:
            summary["skipped"] += 1
            printer(f"{prefix}  skip — manifest entry has no accession")
            continue
        if dry_run:
            summary["skipped"] += 1
            printer(f"{prefix}  dry-run — would process ({candidate.get('reason') or '?'})")
            continue
        if accession in processed:
            summary["resumed"] += 1
            printer(f"{prefix}  resume — already saved in the value collections")
            continue

        # Bind this filing to a fresh live progress row BEFORE resolving, so the
        # bar shows the same "resolving…" stage a single CLI run starts with.
        if on_filing_start is not None:
            on_filing_start(candidate, index, total)

        try:
            try:
                state = state_builder(
                    info,
                    printer=lambda _msg: None,
                    accession=accession,
                    source_form_type=form or None,
                    target_statements=target_statements,
                )
                # Backfills are quarterly-only.  The period agent still
                # identifies the period; check_period_node skips an annual
                # result before concepts/extraction/save, and graph routing
                # suppresses Q4 derivation for every backfill run.
                state["quarterly_only"] = True
            except Exception as exc:  # noqa: BLE001 — one bad filing must not stop the run
                final = {"status": "failed", "error": f"state build failed: {exc}"}
                summary["failed"] += 1
                summary["accessions_failed"].append(accession)
                printer(f"{prefix}  ✗ state build failed — {exc}")
                continue

            if state.get("status") == "failed":
                final = dict(state)
                summary["failed"] += 1
                summary["accessions_failed"].append(accession)
                printer(f"{prefix}  ✗ {state.get('error')}")
                continue

            try:
                # Surface the SAME operational lines a single CLI run prints
                # ([llm]/[tool]/[db] + node stage/step summaries) via the shared
                # RunPresenter.
                if presenter is not None:
                    presenter.install()
                final = graph.invoke(state)
            except Exception as exc:  # noqa: BLE001 — keep going through the range
                final = {"status": "failed", "error": f"graph failed: {exc}"}
                summary["failed"] += 1
                summary["accessions_failed"].append(accession)
                printer(f"{prefix}  ✗ graph failed — {exc}")
                continue
            finally:
                # Callbacks are thread-local; clear them so nothing leaks into
                # the next filing (or a later single-run path).
                _RunPresenter.uninstall()

            status = str(final.get("status") or "?")
            period_label = None
            try:
                from earnings_agents.agent.period import (
                    format_period_label,
                    require_detected_period,
                )
                period_label = format_period_label(require_detected_period(final))
            except Exception:  # noqa: BLE001 — label is observability only
                period_label = None
            concept_count = len(final.get("concept_metrics") or {}) or None

            if status == "saved":
                summary["saved"] += 1
                summary["accessions_completed"].append(accession)
                # The accession is now recorded on the saved values, so a later
                # resume skips this filing without any extra bookkeeping.
                processed.add(accession)
            elif status == "failed":
                # The graph short-circuited (period agent / save gate / identity
                # check); nothing was written, so a resume retries it.
                summary["failed"] += 1
                summary["accessions_failed"].append(accession)
            else:
                summary["skipped"] += 1

            detail = f"{period_label or '?'}"
            if concept_count:
                detail += f", {concept_count} concept(s)"
            printer(f"{prefix}  {status}  ({detail})")
        finally:
            # Always release the live progress row for this filing, including
            # on the failure paths above.
            if on_filing_end is not None:
                on_filing_end(candidate, final)

    return summary
