"""``earnings-backfill`` — bounded, resumable historical quarterly earnings backfills.

Previews a reviewable manifest of historical earnings filings (EDGAR metadata
only), then processes the selected filings through the SAME pipeline the CLI and
Redis worker use — one filing per graph invocation. Backfill mode is
QUARTERLY-ONLY: the period agent classifies each document, annual periods are
skipped before extraction/save, and post-save Q4 derivation is disabled.

    uv run earnings-backfill --ticker ASML --form 6-K --from-year 2023 --to-year 2026
    uv run earnings-backfill --ticker TSM --form 6-K --fiscal-year 2025 --dry-run
    uv run earnings-backfill --ticker MSFT --from-year 2024 --reload --no-resume

Period identity is never inferred here: the year/quarter options only SELECT
which filings to consider; the period agent decides the reporting period from
each document.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid

from earnings_agents.cli.earnings import _resolve_companies
from earnings_agents.config import TARGET_STATEMENTS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="earnings-backfill",
        description=(
            "Backfill historical earnings filings (8-K / 6-K) into "
            "normalize_data. Previews a manifest, then runs each selected "
            "filing through the standard pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run earnings-backfill --ticker ASML --form 6-K --from-year 2023\n"
            "  uv run earnings-backfill --ticker TSM --accession 0001046179-26-000451\n"
            "  uv run earnings-backfill --ticker MSFT --from-year 2024 --dry-run\n"
            "  uv run earnings-backfill --ticker MSFT --from-year 2024 --reload\n"
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", nargs="+", metavar="TICKER", default=[])
    src.add_argument("--cik", nargs="+", metavar="CIK", default=[])

    parser.add_argument(
        "--form",
        dest="forms",
        action="append",
        choices=["8-K", "6-K"],
        help="Source form to include (repeatable). Default: 8-K.",
    )
    parser.add_argument("--from-year", type=int, default=None, metavar="YEAR",
                        help="Earliest FISCAL year to include (the period the filing "
                             "reports on, from SEC reportDate). Selection only — the "
                             "period agent still decides the reporting period.")
    parser.add_argument("--to-year", type=int, default=None, metavar="YEAR",
                        help="Latest FISCAL year to include (default: no upper bound).")
    parser.add_argument("--fiscal-year", type=int, default=None, metavar="YEAR",
                        help="Only this exact fiscal year (stronger than --from/--to-year).")
    parser.add_argument("--fiscal-quarter", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Only this fiscal quarter (approximate: uses the reportDate "
                             "month; requires --fiscal-year).")
    parser.add_argument("--latest", action="store_true", default=False,
                        help="Process only the single newest matching filing per company.")
    parser.add_argument("--oldest-first", action="store_true", default=False,
                        help="Process oldest → newest. Default is newest → oldest (like a "
                             "live run); this only changes processing order, since annual "
                             "periods are skipped in backfill mode.")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Print only per-filing result lines (suppress pipeline "
                             "[llm]/[tool]/[db] progress lines).")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Cap the number of filings processed per company.")
    parser.add_argument("--accession", nargs="+", default=None, metavar="ACCESSION",
                        help="Pin exact accession numbers (skips discovery) — the "
                             "reliable path for 6-K filings.")
    parser.add_argument("--include-unverified", action="store_true", default=False,
                        help="Also include low-confidence candidates (8-K without "
                             "Item 2.02, 6-K with EX-99 exhibits that do not look like "
                             "earnings). Off by default so unrelated filings are not "
                             "silently processed.")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print the manifest and exit — no LLM calls, no writes.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Emit the manifest as JSON (with --dry-run).")
    parser.add_argument("--reload", action="store_true", default=False,
                        help="Re-run filings whose values are already saved (ignore resume).")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="Process every candidate, ignoring the resume check.")
    parser.add_argument("--statements", metavar="STATEMENTS", default=None,
                        help="Comma-separated statements (default: TARGET_STATEMENTS).")
    parser.add_argument("--allow-inconsistent", action="store_true", default=False,
                        help="Save despite unresolved high-severity findings (currency "
                             "values are still omitted when unconfirmed).")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.fiscal_quarter is not None and args.fiscal_year is None:
        print("error: --fiscal-quarter requires --fiscal-year")
        sys.exit(2)
    if args.from_year and args.to_year and args.to_year < args.from_year:
        print(f"error: --to-year ({args.to_year}) is earlier than --from-year ({args.from_year})")
        sys.exit(2)
    if args.json and not args.dry_run:
        print("error: --json is only meaningful with --dry-run")
        sys.exit(2)
    if args.accession and len(args.ticker or []) + len(args.cik or []) > 1:
        print("error: --accession pins specific filings — pass a single --ticker/--cik")
        sys.exit(2)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.allow_inconsistent:
        import earnings_agents.config as _cfg
        _cfg.STRICT_ACCURACY = False
        print("  [config]  --allow-inconsistent: STRICT_ACCURACY disabled")

    target_statements: list[str] = (
        [s.strip().lower() for s in args.statements.split(",") if s.strip()]
        if args.statements
        else TARGET_STATEMENTS
    )

    companies = _resolve_companies(args.cik, args.ticker)
    if not companies:
        print("No valid companies resolved. Exiting.")
        sys.exit(1)

    forms = tuple(args.forms) if args.forms else ("8-K",)

    from earnings_agents.integrations.edgar import build_earnings_manifest

    manifests: list[tuple[dict, list[dict]]] = []
    for info in companies:
        ticker = info.get("ticker") or ""
        cik = info["cik"]
        if args.accession:
            # Pinned accessions skip discovery entirely — the reliable 6-K path.
            candidates = [
                {
                    "form": forms[0],
                    "accession": acc,
                    "filing_date": "",
                    "report_date": "",
                    "primary_document": "",
                    "items": "",
                    "confidence": "pinned",
                    "reason": "pinned_accession",
                }
                for acc in args.accession
            ]
            if args.latest:
                candidates = candidates[:1]
        else:
            if args.latest:
                manifest_args = dict(
                    forms=forms,
                    start_year=args.from_year,
                    end_year=args.to_year,
                    fiscal_year=args.fiscal_year,
                    fiscal_quarter=args.fiscal_quarter,
                    limit=1,
                    include_low_confidence=args.include_unverified,
                )
            else:
                manifest_args = dict(
                    forms=forms,
                    start_year=args.from_year,
                    end_year=args.to_year,
                    fiscal_year=args.fiscal_year,
                    fiscal_quarter=args.fiscal_quarter,
                    limit=args.limit,
                    include_low_confidence=args.include_unverified,
                )
            candidates = build_earnings_manifest(cik, **manifest_args)
        manifests.append((info, candidates))

    if args.dry_run:
        payload = []
        for info, candidates in manifests:
            ticker = info.get("ticker") or info["cik"]
            print(f"\n{ticker} ({info['cik']}) — {len(candidates)} candidate(s)")
            for idx, cand in enumerate(candidates, start=1):
                fy = cand.get("fiscal_year")
                print(
                    f"  [{idx}/{len(candidates)}] {cand['form']:<4} "
                    f"FY{fy if fy is not None else '?'}  "
                    f"filed {cand.get('filing_date') or '?':<10}  "
                    f"period-end {cand.get('report_date') or '?':<10}  "
                    f"{cand['accession']}  "
                    f"{cand.get('confidence')}/{cand.get('reason')}"
                )
            payload.append({"ticker": ticker, "cik": info["cik"], "candidates": candidates})
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
        return

    # ── Run the backfill through the real graph ──────────────────────────────
    # Presentation uses the SAME RunPresenter as the single-ticker CLI, so a
    # backfill prints byte-identical stage / step-summary / [llm] / [tool] /
    # [db] lines with the same spinner + progress bar.
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from earnings_agents.backfill import run_backfill
    from earnings_agents.cli.run_ui import RunPresenter
    from earnings_agents.graph import build_graph

    graph = build_graph()
    run_id = uuid.uuid4().hex[:12]
    totals = {"saved": 0, "failed": 0, "skipped": 0, "resumed": 0}
    order_label = "oldest first" if args.oldest_first else "newest first"

    progress = Progress(
        SpinnerColumn(finished_text="[green]\u2713[/]"),
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=36),
        TimeElapsedColumn(),
    )

    def _run_one(info: dict, candidates: list[dict], overall_task) -> dict:
        """Process one company's filings with a live per-filing progress row."""
        ticker = info.get("ticker") or ""
        name = ticker or info.get("company_name", "?")

        # ONE presenter for the whole company — the same class the single-ticker
        # CLI uses, rebound to a fresh progress row per filing.
        presenter = RunPresenter(name)
        presenter.quiet = args.quiet
        bound: list[tuple[object, str]] = []

        def _before_filing(candidate: dict, index: int, total: int) -> None:
            label = f"[cyan]{name}[/]"
            row = (
                f"{label}  [{index}/{total}] {candidate.get('accession') or '?'}"
                f"  {candidate.get('form') or ''} {candidate.get('filing_date') or ''}".rstrip()
            )
            task = progress.add_task(f"{row}  resolving\u2026", total=None)
            presenter.bind(progress, task)
            bound.append((task, row))

        def _after_filing(candidate: dict, result: dict) -> None:
            if not bound:
                return
            task, row = bound.pop()
            desc, _ok = presenter.result_description(result)
            # ``result_description`` starts with the presenter's name; swap in the
            # richer per-filing row so the completed line keeps the [i/N] context.
            desc = desc.replace(f"[green]{name}[/]", row).replace(
                f"[red]{name}[/]", row
            ).replace(f"[yellow]{name}[/]", row)
            progress.update(task, total=1, completed=1, description=desc)

        summary = run_backfill(
            graph=graph,
            cik=info["cik"],
            ticker=ticker,
            company_name=info["company_name"],
            candidates=candidates,
            target_statements=target_statements,
            resume=not args.no_resume,
            reload=args.reload,
            run_id=run_id,
            printer=lambda _msg: None,
            presenter=presenter,
            on_filing_start=_before_filing,
            on_filing_end=_after_filing,
        )
        progress.update(overall_task, advance=1)
        return summary

    with progress:
        for info, candidates in manifests:
            ticker = info.get("ticker") or ""
            if not candidates:
                progress.console.print(
                    f"[yellow]{ticker or info['cik']}[/] \u2014 no matching filings; nothing to do"
                )
                continue
            # The manifest is newest-first (matching a live run).  --oldest-first
            # reverses only the processing order for chronological audit/review;
            # annual periods are skipped and Q4 derivation is disabled in this
            # quarterly-only backfill mode.
            if args.oldest_first and not args.latest:
                candidates = list(reversed(candidates))
            progress.console.print(
                f"\n[bold]=== {info.get('company_name')} ({ticker or info['cik']}) \u2014 "
                f"{len(candidates)} filing(s), {order_label} ===[/]"
            )
            overall_task = progress.add_task(
                f"[bold]{ticker or info['cik']}[/]  backfill", total=len(candidates)
            )
            summary = _run_one(info, candidates, overall_task)
            for key in totals:
                totals[key] += summary.get(key, 0)

    print(
        f"\nBackfill run {run_id} complete — saved {totals['saved']}, "
        f"failed {totals['failed']}, skipped {totals['skipped']}, "
        f"resumed {totals['resumed']}"
    )


if __name__ == "__main__":
    main()
