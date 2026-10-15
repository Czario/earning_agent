"""Command-line interface for the earnings scraping pipeline.

Pass one or more company identifiers and the graph runs for each.

Usage examples:
    uv run earnings --cik 0000320193
    uv run earnings --ticker AAPL MSFT GOOGL
    uv run earnings --cik 0000320193 0000789019
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys

import requests
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from earnings_agents.config import (  # noqa: E402
    GEMINI_API_KEY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TARGET_STATEMENTS,
)
from earnings_agents.nodes.detect import detect_document_type_node  # noqa: E402
from earnings_agents.graph import build_graph  # noqa: E402
from earnings_agents.registry import lookup_by_cik, lookup_by_ticker  # noqa: E402
from earnings_agents.integrations.edgar import (  # noqa: E402
    EARNINGS_FORMS,
    accession_from_url,
    get_exhibits_for_accession,
    get_filing_form_type,
    get_latest_earnings_url,
    get_latest_filing_url,
)
from earnings_agents.hooks import set_detail_callback, set_node_callback  # noqa: E402

SEP = "=" * 64

_logger = logging.getLogger(__name__)


# Progress presentation lives in ``cli/run_ui.py`` so the single-ticker CLI and
# the historical backfill CLI print the identical stage/step/[llm] trace.  The
# names are re-exported here because ``progress.py`` imports them.
from earnings_agents.cli.run_ui import (  # noqa: E402
    NODE_LABELS as _NODE_LABELS,
    SHORT_STAGE as _SHORT_STAGE,
    RunPresenter,
    fmt_duration as _fmt_dur,
    format_step_line as _format_step_line,
)


def _check_ollama() -> tuple[bool, str]:
    """Return (ok, detail) for Ollama reachability and model availability."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False, f"Ollama responded {resp.status_code}"
        available = [m["name"] for m in resp.json().get("models", [])]
        base_name = OLLAMA_MODEL.split(":")[0]
        if not any(m.startswith(base_name) for m in available):
            return False, f"Model '{OLLAMA_MODEL}' not pulled (available: {available or 'none'})"
        return True, f"Ollama OK — model '{OLLAMA_MODEL}' available"
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama unreachable: {exc}"


def _check_llm() -> tuple[bool, str]:
    """Return (ok, detail) for the selected LLM provider configuration."""
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            return False, "Groq API key missing (set GROQ_API_KEY in .env)"
        return True, f"Groq configured — model '{GROQ_MODEL}'"
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            return False, "Gemini API key missing (set GEMINI_API_KEY in .env)"
        return True, f"Gemini configured — model '{GEMINI_MODEL}'"
    if LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            return False, "DeepSeek API key missing (set DEEPSEEK_API_KEY in .env)"
        return True, f"DeepSeek configured — model '{DEEPSEEK_MODEL}'"
    return _check_ollama()


def _check_mongodb() -> tuple[bool, str]:
    """Return (ok, detail) for MongoDB reachability."""
    try:
        from pymongo import MongoClient  # local import — only needed in dry-run path

        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return True, f"MongoDB OK — {MONGODB_DB}.{MONGODB_COLLECTION}"
    except Exception as exc:  # noqa: BLE001
        return False, f"MongoDB unreachable: {exc}"


def _print_insertion_summary(results: list[dict], printer=print) -> None:
    """Print a clean table of all inserted concept values for every saved result.

    Shown once at the very end of the run so operators can quickly verify what
    was upserted.  Values computed by the Tier-3 derivation engine are tagged
    with ``[DERIVED]`` so they are visually distinct from values read directly
    from the filing.
    """
    saved = [r for r in results if r.get("status") == "saved" and r.get("concept_metrics")]
    if not saved:
        return

    for result in saved:
        ticker = result.get("ticker", "?")
        concept_metrics: dict = result.get("concept_metrics") or {}
        target_concepts: list = result.get("target_concepts") or []
        calculated_concepts: list = result.get("calculated_concepts") or []
        derived_ids: set[str] = set(result.get("derived_concept_ids") or [])

        all_concepts = target_concepts + calculated_concepts
        id_to_label: dict[str, str] = {
            c["_id"]: c.get("label", c["_id"]) for c in all_concepts
        }

        try:
            from earnings_agents.agent.period import require_detected_period
            detected_period = require_detected_period(result)
            period = detected_period.period_end.isoformat()
        except Exception:
            period = ""
        header = f"Inserted concept values for {ticker}"
        if period:
            header += f"  ({period})"
        printer("")
        printer(f"┌─ {header} {'─' * max(1, 66 - len(header))}")

        rows = sorted(
            concept_metrics.items(),
            key=lambda kv: id_to_label.get(kv[0], kv[0]).lower(),
        )
        from_filing = [(cid, v) for cid, v in rows if cid not in derived_ids]
        derived_rows = [(cid, v) for cid, v in rows if cid in derived_ids]

        for cid, value in from_filing:
            label = id_to_label.get(cid, cid)
            printer(f"│  {label:<45} {value}")

        if derived_rows:
            printer(f"├─ Derived ({len(derived_rows)}) ─{'─' * 54}")
            for cid, value in derived_rows:
                label = id_to_label.get(cid, cid)
                printer(f"│  {label:<45} {value}  [DERIVED]")

        absent = sorted(c["label"] for c in target_concepts if c["_id"] not in concept_metrics)
        if absent:
            printer(f"├─ Not in filing ({len(absent)}) ─{'─' * 51}")
            for lbl in absent:
                printer(f"│  {lbl}")

        printer(f"└{'─' * 68}")


def _resolve_companies(ciks: list[str], tickers: list[str]) -> list[dict]:
    """Return a list of company dicts ready to feed into the graph."""
    companies: list[dict] = []

    for cik in ciks:
        info = lookup_by_cik(cik)
        if not info:
            print(f"[WARN] CIK {cik} not found in tickers.json — skipping")
            continue
        companies.append(info)

    for t in tickers:
        info = lookup_by_ticker(t)
        if not info:
            print(f"[WARN] Ticker {t.upper()} not found in tickers.json — skipping")
            continue
        companies.append(info)

    return companies


def _forms_for_source_form(source_form_type: str | None) -> tuple[str, ...]:
    """Map a requested source form to the EDGAR forms to search.

    The form is ALWAYS explicit — there is no AUTO mode and no default filing:
    callers must name the form they want (``8-K`` for US issuers, ``6-K`` for
    foreign private issuers such as ASML/TSM).  An empty value yields ``()``,
    which callers treat as "no form supplied" and refuse EDGAR discovery.

    This only chooses which filing to fetch — the period agent decides the
    reporting period after the document is read.
    """
    requested = str(source_form_type or "").strip().upper()
    if not requested:
        return ()
    return (requested,)


def _build_8k_state(
    ticker: str,
    company_name: str,
    cik: str,
    *,
    filing_url: str | None = None,
    local_filing_path: str | None = None,
    supplemental_urls: list[str] | None = None,
    accession: str | None = None,
    source_form_type: str | None = None,
    quarterly_only: bool = False,
    printer=print,
    dry_run: bool = False,
    target_statements: list[str] | None = None,
) -> dict:
    """Build the initial state for an 8-K pipeline run — shared by CLI and worker.

    Both the CLI (``_build_initial_state``) and the Redis worker
    (``_process_payload``) call this function so URL/accession resolution and
    state construction are identical in every code path.

    Period detection does NOT happen here — the period agent reads the filing
    document inside the graph (detect_period node).  This function only
    resolves the filing URL and accession.  No period value is ever seeded
    into the state here: ``detected_period`` is written ONLY by
    ``detect_period_node``.  No accession or period existence checks —
    the graph's check_period node sees whether the exact fiscal period is
    already stored and schedules a deferred replace.

    When *filing_url* is not provided, the function fetches the latest 8-K
    from SEC EDGAR (CLI path).  When *accession* IS provided (worker path —
    RSS-polled filing), the function resolves the exhibit documents of that
    EXACT accession from its filing index and does NOT scan the submissions
    API for "the latest 8-K" (no lag/empty-items race on fresh filings).
    When *filing_url* IS provided (worker path — manual trigger with a
    press-release HTML or shareholder-letter PDF URL), the function skips the
    EDGAR lookup and uses the pre-resolved URL directly.  Reporting-period
    identity is determined only by the period agent after the filing is
    fetched.
    """
    _base = {
        "ticker": ticker or cik,
        "company_name": company_name,
        "file_type": None,
        "raw_text": None,
        "local_filing_path": local_filing_path,
        "metrics": None,
        "error": None,
        "exhibit_meta": [],
        "target_statements": target_statements if target_statements is not None else TARGET_STATEMENTS,
    }
    if quarterly_only:
        # Opt-in: the period agent still decides the period, but an annual
        # result is skipped before extraction/save and Q4 is never derived.
        _base["quarterly_only"] = True

    # ── Resolve filing URL, accession, and exhibit list ───────────────────
    exhibits: list[dict] = []
    forms = _forms_for_source_form(source_form_type)

    def _discover() -> tuple[str | None, list[str], str | None, list[dict], str | None]:
        """Search EDGAR for the latest earnings filing of the requested form.

        The 8-K path keeps calling ``get_latest_earnings_url`` (its historical
        signature/behavior); other forms use the generalized discovery.
        """
        if forms == ("8-K",):  # explicit --form 8-K keeps the legacy path
            url, supp, acc, exb = get_latest_earnings_url(cik)
            return url, supp, acc, exb, "8-K"
        return get_latest_filing_url(cik, forms=forms)

    # A form is required whenever we must SEARCH EDGAR for a filing.  A pinned
    # accession or a supplied URL needs no form: the accession resolves its own
    # index and the form is looked up / derived from it.
    if not filing_url and not local_filing_path and not accession and not forms:
        return {
            **_base,
            "discovered_file_url": None,
            "supplemental_file_urls": [],
            "status": "failed",
            "error": (
                "No source form supplied — EDGAR discovery requires an explicit "
                "form (--form 8-K or --form 6-K). Nothing is fetched by default."
            ),
        }

    if not filing_url and not local_filing_path:
        if accession:
            # RSS-polled filing: the accession identifies the EXACT filing the
            # poller saw (8-K or 6-K).  Resolve its exhibits directly from the
            # filing index — NO submissions-API "latest filing" scan, so there
            # is no race with submissions-API lag / empty items on a brand-new
            # filing.  Works for any form because it is accession-pinned.
            printer(
                f"  [EDGAR]  {company_name} ({ticker or cik}) resolving pinned "
                f"accession {accession}..."
            )
            exhibits = get_exhibits_for_accession(cik, accession)
            if exhibits:
                filing_url = exhibits[0]["url"]
                supplemental_urls = [e["url"] for e in exhibits[1:]]
            else:
                printer(
                    f"  [EDGAR]  pinned accession {accession} has no documents "
                    f"— falling back to latest earnings filing"
                )
                (
                    filing_url,
                    supplemental_urls,
                    accession,
                    exhibits,
                    discovered_form,
                ) = _discover()
                source_form_type = source_form_type or discovered_form
            # Provenance: the caller (worker payload) usually knows the form;
            # otherwise look it up from the submissions API (best effort).
            if not source_form_type and accession:
                source_form_type = get_filing_form_type(cik, accession)
        else:
            printer(
                f"  [EDGAR]  {company_name} ({ticker or cik}) querying SEC EDGAR "
                f"for {'/'.join(forms)}..."
            )
            (
                filing_url,
                supplemental_urls,
                accession,
                exhibits,
                discovered_form,
            ) = _discover()
            source_form_type = source_form_type or discovered_form
    elif filing_url:
        printer(
            f"  [URL]    {company_name} ({ticker or cik}) using provided filing URL "
            f"— EDGAR lookup skipped"
        )
    else:
        printer(
            f"  [file]   {company_name} ({ticker or cik}) using local PDF "
            f"— EDGAR lookup skipped"
        )

    # Provenance for manual runs: no EDGAR lookup happened, so the source SEC
    # form is unknown.  Recover it best-effort when the URL/filename embeds an
    # accession (EDGAR archive links and pasted accession filenames); otherwise
    # leave it unset and the value writer falls back to the period-family label.
    if not source_form_type and (filing_url or local_filing_path):
        derived_acc = accession_from_url(filing_url) or accession_from_url(
            local_filing_path
        )
        if derived_acc:
            if not accession:
                accession = derived_acc
            source_form_type = get_filing_form_type(cik, derived_acc)
            printer(
                f"  [form]   derived source form from URL accession "
                f"{derived_acc}: {source_form_type or 'unknown'}"
            )

    if accession:
        _base["accession_number"] = accession
    if source_form_type:
        _base["source_form_type"] = str(source_form_type).strip().upper()
    if exhibits:
        _base["exhibit_meta"] = exhibits

    if not filing_url and not local_filing_path:
        return {
            **_base,
            "discovered_file_url": None,
            "supplemental_file_urls": [],
            "status": "failed",
            "error": (
                f"No {'/'.join(forms)} earnings filing found on SEC EDGAR "
                f"for CIK {cik}"
            ),
        }

    if supplemental_urls:
        printer(
            f"  [EDGAR]  +{len(supplemental_urls)} supplemental exhibit(s): "
            f"{[u.rsplit('/', 1)[-1] for u in supplemental_urls]}"
        )

    return {
        **_base,
        "discovered_file_url": filing_url,
        "local_filing_path": local_filing_path,
        "supplemental_file_urls": supplemental_urls or [],
        "status": "discovered",
    }


def _build_initial_state(
    info: dict,
    printer=print,
    dry_run: bool = False,
    *,
    filing_url: str | None = None,
    local_filing_path: str | None = None,
    supplemental_urls: list[str] | None = None,
    accession: str | None = None,
    source_form_type: str | None = None,
    quarterly_only: bool = False,
    target_statements: list[str] | None = None,
) -> dict:
    """Build the LangGraph initial state for one company (CLI path).

    Thin wrapper around :func:`_build_8k_state` — resolves the filing URL
    from SEC EDGAR and delegates all period detection to the shared function.
    When *filing_url* is provided (Redis worker manual trigger), the EDGAR
    lookup is skipped and the given URL is used directly.  *source_form_type*
    requests 8-K (default) or 6-K discovery for the auto-fetch path.
    """
    return _build_8k_state(
        ticker=info.get("ticker") or "",
        company_name=info["company_name"],
        cik=info["cik"],
        filing_url=filing_url,
        local_filing_path=local_filing_path,
        supplemental_urls=supplemental_urls,
        accession=accession,
        source_form_type=source_form_type,
        quarterly_only=quarterly_only,
        printer=printer,
        dry_run=dry_run,
        target_statements=target_statements,
    )


def _run_company(
    graph,
    info: dict,
    printer=print,
    *,
    target_statements: list[str] | None = None,
) -> dict:
    label = f"{info['company_name']} ({info.get('ticker') or info['cik']})"
    printer(f"\n{SEP}")
    printer(f"  Company : {label}")
    printer(f"  CIK     : {info['cik']}")
    if info.get("source_form_type"):
        printer(f"  Form    : {info['source_form_type']}")

    state = _build_initial_state(
        info,
        printer=printer,
        target_statements=target_statements,
        filing_url=info.get("filing_url"),
        accession=info.get("accession"),
        source_form_type=info.get("source_form_type"),
        quarterly_only=bool(info.get("quarterly_only")),
    )

    if state["status"] == "failed":
        printer(f"  [SKIP]  {state['error']}")
        printer(SEP)
        return state

    if state["status"] == "skipped":
        printer(f"  [SKIP]  {state['error']}")
        printer(SEP)
        return state

    if state["status"] != "failed":
        if state.get("discovered_file_url"):
            printer(f"  Filing  : {state['discovered_file_url']}")
    printer(SEP)

    final = graph.invoke(state)

    printer(f"\n  Status  : {final.get('status')}")
    printer(f"  File URL: {final.get('discovered_file_url')}")
    printer(f"  Type    : {final.get('file_type')}")
    if final.get("metrics"):
        m = final["metrics"]
        printer(f"  Metrics ({len(m)} fields):")
        for lbl, value in m.items():
            printer(f"    {lbl:<40} {value}")
    if final.get("error"):
        printer(f"  Error   : {final.get('error')}")
    printer(SEP)

    return final


def _dry_run_company(
    info: dict,
    printer=print,
    *,
    target_statements: list[str] | None = None,
) -> dict:
    """Resolve URLs and check service connectivity without running the LLM or saving.

    Returns the initial state augmented with a ``_dry_run_verdict`` key:
    ``"ready"`` | ``"warning"`` | ``"blocked"``.
    """
    label = f"{info['company_name']} ({info.get('ticker') or info['cik']})"
    printer(f"\n{SEP}")
    printer(f"  DRY-RUN : {label}")
    printer(f"  CIK     : {info['cik']}")

    state = _build_initial_state(
        info,
        printer=printer,
        dry_run=True,
        target_statements=target_statements,
        filing_url=info.get("filing_url"),
        accession=info.get("accession"),
        source_form_type=info.get("source_form_type"),
        quarterly_only=bool(info.get("quarterly_only")),
    )

    llm_ok, llm_detail = _check_llm()
    mongo_ok, mongo_detail = _check_mongodb()

    file_type: str | None = None
    url_blocked = state.get("status") == "failed"
    if not url_blocked and state.get("discovered_file_url"):
        dt = detect_document_type_node(state)
        file_type = dt.get("file_type")

    if state.get("status") == "skipped":
        verdict = "skipped"
    elif url_blocked:
        verdict = "blocked"
    elif not llm_ok or not mongo_ok:
        verdict = "warning"
    else:
        verdict = "ready"

    url_display = state.get("discovered_file_url") or "(none)"
    printer(SEP)
    printer(f"  URL     : {url_display}")
    if file_type:
        printer(f"  DocType : {file_type}")
    printer(f"  LLM     : {'OK  ' if llm_ok else 'FAIL'} — {llm_detail}")
    printer(f"  MongoDB : {'OK  ' if mongo_ok else 'FAIL'} — {mongo_detail}")
    printer(f"  Verdict : {verdict.upper()}")
    if verdict == "blocked":
        printer(f"  Reason  : {state.get('error', 'URL resolution failed')}")
    elif verdict == "skipped":
        printer(f"  Reason  : {state.get('error', 'Filing discovery skipped')}")
    elif verdict == "warning":
        if not llm_ok:
            if LLM_PROVIDER == "ollama":
                printer(
                    "  Action  : Start Ollama and pull the required model (see OLLAMA_MODEL in .env)"
                )
            else:
                printer("  Action  : Set GROQ_API_KEY (and optionally GROQ_MODEL) in .env")
        if not mongo_ok:
            printer("  Action  : Start MongoDB or check MONGODB_URI in .env")
    else:
        printer("  Action  : Run without --dry-run to execute the full pipeline")
    printer(SEP)

    return {**state, "_dry_run_verdict": verdict}


# ── Thread-pool workers ──────────────────────────────────────────────────────

def _is_already_saved(ticker: str) -> bool:
    """Return True if an earnings document for *ticker* already exists in MongoDB for the current year."""
    from datetime import datetime, timezone
    from earnings_agents.integrations.mongo import get_collection
    year = datetime.now(timezone.utc).year
    doc_id = f"{ticker}_{year}_latest"
    try:
        return get_collection().count_documents({"_id": doc_id}, limit=1) > 0
    except Exception:  # noqa: BLE001
        return False


def _run_company_parallel(args: tuple) -> dict:
    """Thread worker: run one company and update the shared rich Progress."""
    if len(args) >= 6:
        graph, info, skip_existing, progress, overall_task, target_statements = args[:6]
    else:
        graph, info, skip_existing, progress, overall_task = args
        target_statements = None
    ticker = info.get("ticker", "")
    name = ticker or info.get("company_name", "?")
    label = f"[cyan]{name}[/]"

    if skip_existing and ticker and _is_already_saved(ticker):
        progress.update(overall_task, advance=1)
        return {"status": "skipped", "ticker": ticker}

    company_task = progress.add_task(f"{label}  resolving\u2026", total=None)

    # The SAME presenter the backfill CLI uses — one implementation, so both
    # paths print byte-identical stage/step/[llm]/[tool]/[db] output.
    presenter = RunPresenter(
        name, label=label, progress=progress, task=company_task,
    )
    presenter.install()
    result = _run_company(graph, info, printer=lambda _: None, target_statements=target_statements)
    RunPresenter.uninstall()

    desc, _ok = presenter.result_description(result)
    progress.update(company_task, total=1, completed=1, description=desc)
    progress.update(overall_task, advance=1)
    return result


def _dry_run_company_parallel(args: tuple) -> dict:
    """Thread worker: dry-run one company and update the shared rich Progress."""
    if len(args) >= 5:
        info, skip_existing, progress, overall_task, target_statements = args[:5]
    else:
        info, skip_existing, progress, overall_task = args
        target_statements = None
    ticker = info.get("ticker", "")
    name = ticker or info.get("company_name", "?")
    label = f"[cyan]{name}[/]"

    if skip_existing and ticker and _is_already_saved(ticker):
        progress.update(overall_task, advance=1)
        return {"_dry_run_verdict": "ready", "status": "skipped", "ticker": ticker}

    company_task = progress.add_task(f"{label}  checking\u2026", total=None)
    result = _dry_run_company(info, printer=lambda _: None, target_statements=target_statements)
    verdict = result.get("_dry_run_verdict", "?")
    color = {"ready": "green", "warning": "yellow", "blocked": "red"}.get(verdict, "white")
    verdict_label = verdict
    progress.update(company_task, total=1, completed=1,
                    description=f"[{color}]{name}[/]  {verdict_label}")
    progress.update(overall_task, advance=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the earnings scraping pipeline for one or more companies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run earnings --cik 0000320193\n"
            "  uv run earnings --ticker AAPL MSFT\n"
            "  uv run earnings --ticker ASML --form 6-K\n"
            "  uv run earnings --ticker ASML --accession 0000937966-26-000012\n"
            "  uv run earnings --ticker TSM --filing-url https://.../tsm-6k.htm\n"
        ),
    )
    parser.add_argument(
        "--cik",
        nargs="+",
        metavar="CIK",
        default=[],
        help="One or more CIK numbers (with or without leading zeros)",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        metavar="TICKER",
        default=[],
        help="One or more ticker symbols (e.g. AAPL MSFT)",
    )
    parser.add_argument(
        "--quarterly-only",
        action="store_true",
        default=False,
        help=(
            "Process reported QUARTERLY periods only: the period agent still "
            "identifies the period, but an annual filing is skipped before "
            "extraction/save and Q4 is never derived."
        ),
    )
    parser.add_argument(
        "--form",
        "--form-type",
        dest="source_form_type",
        choices=["8-K", "6-K"],
        required=True,
        help=(
            "REQUIRED. Source SEC form to fetch: 8-K (US issuers, Item 2.02) or "
            "6-K (foreign private issuers such as ASML/TSM). There is no "
            "default and no auto-detection — nothing is fetched without it. "
            "The form never decides the reporting period; the period agent "
            "reads it from the document."
        ),
    )
    parser.add_argument(
        "--filing-url",
        metavar="URL",
        default=None,
        help=(
            "Use this exact filing document URL (press-release HTML or PDF) "
            "instead of SEC EDGAR discovery."
        ),
    )
    parser.add_argument(
        "--accession",
        metavar="ACCESSION",
        default=None,
        help=(
            "Pin the exact SEC accession number (e.g. 0000937966-26-000012) — "
            "resolves its exhibits from the filing index with no 'latest "
            "filing' scan. Works for 8-K and 6-K."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Preview mode: resolve URLs and check service connectivity without running LLM "
            "extraction or saving to MongoDB. Prints a ready/warning/blocked verdict per company."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        metavar="N",
        help="Maximum parallel workers when processing multiple companies (default: 8).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip companies that already have a saved document in MongoDB for the current year.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show full DEBUG-level log output from every pipeline node.",
    )
    parser.add_argument(
        "--allow-inconsistent",
        action="store_true",
        default=False,
        help=(
"Save the document to MongoDB even when high-severity findings are unresolved."
        ),
    )
    parser.add_argument(
        "-s",
        "--statements",
        metavar="STATEMENTS",
        default=None,
        help=(
            "Comma-separated list of financial statements to extract "
            "(e.g. 'income,balancesheet,cashflow'). Defaults to TARGET_STATEMENTS config."
        ),
    )
    args = parser.parse_args()

    if not args.cik and not args.ticker:
        parser.error("Provide at least one --cik or --ticker argument.")

    target_statements: list[str] = (
        [s.strip().lower() for s in args.statements.split(",") if s.strip()]
        if args.statements
        else TARGET_STATEMENTS
    )

    if args.allow_inconsistent:
        # Flip the REAL config knob — save.py reads it lazily from the config
        # module at save time, so this override is honored.  (Setting
        # graph.STRICT_ACCURACY did nothing: the graph never reads it.)
        import earnings_agents.config as _cfg
        _cfg.STRICT_ACCURACY = False
        print("  [config]  --allow-inconsistent: STRICT_ACCURACY disabled "
              "(high-severity findings will not block the save; values with "
              "an unconfirmed currency are still omitted, never assumed USD)")

    companies = _resolve_companies(args.cik, args.ticker)
    if not companies:
        print("No valid companies resolved. Exiting.")
        sys.exit(1)

    # Per-company filing-selection options.  A pinned accession or a manual URL
    # takes precedence over EDGAR discovery; the source form only selects which
    # form to look for (it never decides the reporting period).
    if args.accession and len(companies) > 1:
        parser.error("--accession pins ONE filing — pass a single --ticker/--cik")
    for _info in companies:
        _info["source_form_type"] = args.source_form_type
        _info["filing_url"] = args.filing_url
        _info["accession"] = args.accession
        _info["quarterly_only"] = args.quarterly_only

    _progress = Progress(
        SpinnerColumn(finished_text="[green]\u2713[/]"),
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=36),
        TimeElapsedColumn(),
    )

    # Route all logging through rich's console so log lines print above the
    # live progress display instead of writing raw bytes to stderr and
    # breaking the ANSI cursor control.
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=_progress.console,
                show_path=args.verbose,
                rich_tracebacks=args.verbose,
                log_time_format="[%X]",
            )
        ],
        force=True,
    )

    if LLM_PROVIDER == "openai":
        llm_label = f"openai:{OPENAI_MODEL}"
    elif LLM_PROVIDER == "groq":
        llm_label = f"groq:{GROQ_MODEL}"
    elif LLM_PROVIDER == "gemini":
        llm_label = f"gemini:{GEMINI_MODEL}"
    elif LLM_PROVIDER == "deepseek":
        llm_label = f"deepseek:{DEEPSEEK_MODEL}"
    else:
        llm_label = f"ollama:{OLLAMA_MODEL}"
    _progress.console.print(f"[bold cyan]LLM[/]      : {llm_label}")
    _progress.console.print(f"[bold cyan]Statements[/] : {', '.join(target_statements)}")

    if args.dry_run:
        total = len(companies)
        with _progress as progress:
            overall_task = progress.add_task("[bold]Dry-run[/]", total=total)
            worker_args = [
                (c, args.skip_existing, progress, overall_task, target_statements)
                for c in companies
            ]
            results = []
            if total == 1:
                results.append(_dry_run_company_parallel(worker_args[0]))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                    for future in concurrent.futures.as_completed(
                        pool.submit(_dry_run_company_parallel, wa) for wa in worker_args
                    ):
                        results.append(future.result())

        blocked = sum(1 for r in results if r.get("_dry_run_verdict") == "blocked")
        warnings = sum(1 for r in results if r.get("_dry_run_verdict") == "warning")
        skipped = sum(1 for r in results if r.get("_dry_run_verdict") == "skipped")
        ready = total - blocked - warnings - skipped
        parts = [f"{ready} ready"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if warnings:
            parts.append(f"{warnings} warning")
        if blocked:
            parts.append(f"{blocked} blocked")
        print(f"Dry-run: {total} companies \u2014 {', '.join(parts)}.")
        sys.exit(1 if blocked else 0)

    # Live run
    graph = build_graph()
    total = len(companies)
    with _progress as progress:
        overall_task = progress.add_task("[bold]Companies[/]", total=total)
        worker_args = [
            (graph, c, args.skip_existing, progress, overall_task, target_statements)
            for c in companies
        ]
        results = []
        if total == 1:
            results.append(_run_company_parallel(worker_args[0]))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                for future in concurrent.futures.as_completed(
                    pool.submit(_run_company_parallel, wa) for wa in worker_args
                ):
                    results.append(future.result())

    saved = [r for r in results if r.get("status") == "saved"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    failed = [r for r in results if r.get("status") not in ("saved", "skipped")]
    summary = f"Done: {len(saved)}/{total} saved"
    if skipped:
        summary += f", {len(skipped)} skipped"
    if failed:
        summary += f", {len(failed)} failed"
    print(summary + ".")
    if failed:
        print("Failed: " + ", ".join(r.get("ticker", "?") for r in failed))
        for r in failed:
            if r.get("error"):
                print(f"  {r.get('ticker', '?')}: {r['error']}")
    # Silence MongoDB driver teardown noise (heartbeat/connection-pool debug
    # messages fired by atexit handlers) so the summary is always the last
    # thing printed in the terminal.
    for _log_name in ("pymongo", "mongodb", "bson"):
        logging.getLogger(_log_name).setLevel(logging.WARNING)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

