"""Redis consumer: process 8-K filings published by admin_backend.

Listens to the **dedicated** ``sec:filings:8k`` queue (set via
``REDIS_QUEUE_NAME`` env var, default ``sec:filings:8k``).
admin_backend publishes 8-K messages here and 10-K/10-Q messages to a
separate queue consumed by the filings-extractor worker.  Each worker
only ever sees its own messages — no re-queue loops.

Pipeline mirrors ``uv run earnings --ticker X`` exactly:
  CLI    → ticker arg → EDGAR lookup → Exhibit 99.1 → pipeline
  Worker → Redis msg  → accession  → Exhibit 99.1 → pipeline
  Manual → Redis msg  → filing_url (PDF letter / press HTML) → pipeline
            (worker skips the EDGAR lookup when ``filing_url`` is present)

Progress events are published to the ``sec:worker:events`` Redis pub/sub
channel so admin_backend can stream them to the frontend in real time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from time import perf_counter
from typing import Any

from bson import ObjectId
from pymongo import MongoClient
from redis import Redis

from earnings_agents.config import REDIS_URL
from earnings_agents.hooks import set_call_callback, set_detail_callback, set_node_callback
from earnings_agents.integrations.redis import get_redis_client, serialize_message
from earnings_agents.progress import WorkerProgressPublisher, make_call_callback, make_node_callback, WorkerHeartbeat
from earnings_agents.graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,  # override earnings.py which calls basicConfig(WARNING) on import
)
logger = logging.getLogger(__name__)

# Dedicated queue — admin_backend publishes 8-K messages here only.
_DEFAULT_QUEUE = "sec:filings:8k"


# ── MongoDB helper ─────────────────────────────────────────────────────────────

def _update_load_request_status(
    payload: dict[str, Any],
    status: str,
    period_of_report: str | None = None,
) -> None:
    """Update StockLoadRequest.status (and optionally sec_period_of_report) in MongoDB."""
    load_request_id = payload.get("load_request_id")
    if not load_request_id:
        return
    try:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("DATABASE_NAME", "normalize_data")
        fields: dict[str, Any] = {"status": status}
        if period_of_report:
            fields["sec_period_of_report"] = period_of_report
        with MongoClient(uri, serverSelectionTimeoutMS=5000) as mongo:
            mongo[db_name]["stock_load_requests"].update_one(
                {"_id": ObjectId(load_request_id)},
                {"$set": fields},
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to update load_request status → %s: %s", status, exc)


# ── Core processing — mirrors CLI's _build_initial_state + _run_company ───────

def _process_payload(graph, payload: dict[str, Any]) -> bool:
    """Process one 8-K filing message using the same pipeline as the CLI.

    Steps (identical to ``uv run earnings --ticker X``):
      1. Resolve company (cik, name) from normalize_data — same values the
         CLI derives from the SEC registry.
      2. Build the initial state via the SAME ``_build_initial_state`` the
         CLI uses (EDGAR → Exhibit 99.1 URL + accession + filing date).
      3. Run the SAME graph: period agent → check_period → concepts →
         extraction → save.  Period detection and every pipeline behavior
         are byte-for-byte the same code path as the CLI.

    ``graph`` is injected (built once in ``main``) so tests can substitute a
    recorder.
    """
    ticker = (payload.get("ticker") or "").upper()

    # Create publisher immediately so every exit path can report status.
    redis_url = os.getenv("REDIS_URL", REDIS_URL)
    pub = WorkerProgressPublisher(redis_url, ticker, payload.get("load_request_id"))

    # ── 1. Validate message ───────────────────────────────────────────────
    if not ticker:
        pub.publish("skip", "message missing ticker — cannot look up filing")
        pub.close()
        logger.warning("8-K message missing ticker — skipping")
        return True

    # ── 2. Resolve CIK + company name from DB (mirrors CLI path) ────────
    from earnings_agents.integrations.normalize import get_company_by_ticker

    company = get_company_by_ticker(ticker)
    if company is None:
        pub.publish("skip", f"ticker {ticker} not found in normalize_data — skipping")
        pub.close()
        logger.warning("8-K message for %s — company not in DB, skipping", ticker)
        return True

    cik = company["cik"]
    company_name = company.get("name", ticker)

    # ── 3. Build state via shared function (identical to CLI path) ──────────
    # A manual trigger may carry a direct filing_url (press-release HTML or
    # shareholder-letter PDF from the admin panel).  When present, the EDGAR
    # lookup is skipped inside _build_8k_state and the URL is used directly;
    # the queue's filing_date (queued-day) still anchors the period agent's
    # sanity window.  When absent, the behavior is the classic CLI auto-fetch.
    from earnings_agents.cli.earnings import _build_initial_state

    state = _build_initial_state(
        {
            "ticker": ticker,
            "company_name": company_name,
            "cik": cik,
        },
        filing_url=payload.get("filing_url"),
        filing_date=payload.get("filing_date"),
    )

    # The reporting period is decided by the period agent INSIDE the graph —
    # the pre-graph state carries only URL/accession.
    # _build_8k_state already set status — honour skip/failed.
    if state.get("status") in ("skipped", "failed"):
        reason = state.get("error") or state.get("status")
        pub.publish("skip", f"skipped — {reason}")
        pub.close()
        return True

    # Visible source differentiator: manual triggers carry a direct filing URL
    # (press-release HTML or PDF shareholder letter) and skipped the EDGAR
    # lookup inside _build_8k_state.  Publish an explicit event line so the
    # admin panel's live log shows WHY the URL differs from the EDGAR path.
    manual_url = payload.get("filing_url")
    if manual_url:
        pub.publish(
            "progress",
            "[source]  manual filing URL provided — extracting from it "
            "directly (EDGAR lookup skipped)",
        )

    logger.info("Processing 8-K for %s  url=%s", ticker, state.get("discovered_file_url"))

    # ── Set up progress callbacks using the shared worker_progress module ────────
    # make_node_callback fires on both start (▶ stage) and end (step summary),
    # mirroring the CLI's rich progress output exactly.
    # make_call_callback forwards every LLM/DB/HTTP call event so the UI shows
    # the same intermediate lines the CLI prints.
    llm_call_count: list[int] = [0]
    set_node_callback(make_node_callback(pub, ticker))
    set_call_callback(make_call_callback(pub, ticker, llm_call_count))
    set_detail_callback(None)   # spinner text — not needed in worker

    t0 = perf_counter()
    try:
        with WorkerHeartbeat(pub, ticker, interval_s=60):
            final = graph.invoke(state)
    except BaseException as exc:
        # Catches Exception (LLM errors etc.), KeyboardInterrupt, and SystemExit
        # (SIGTERM converted below) — publishes a visible failure line to the UI
        # so the user sees why the log stopped rather than an empty spinner.
        elapsed_s = perf_counter() - t0
        elapsed_str = (
            f"{elapsed_s:.1f}s"
            if elapsed_s < 60
            else f"{int(elapsed_s // 60)}m {elapsed_s % 60:.0f}s"
        )
        reason = (
            "worker stopped (SIGTERM)"
            if isinstance(exc, SystemExit)
            else "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else str(exc)[:120]
        )
        pub.publish(
            "summary",
            f"✗ {reason}  {elapsed_str}",
            kind="summary",
        )
        raise
    finally:
        set_node_callback(None)
        set_call_callback(None)
        pub.close()

    elapsed_s = perf_counter() - t0
    elapsed_str = (
        f"{elapsed_s:.1f}s"
        if elapsed_s < 60
        else f"{int(elapsed_s // 60)}m {elapsed_s % 60:.0f}s"
    )
    status = final.get("status", "")
    llm_tag = f"  ({llm_call_count[0]} LLM calls)" if llm_call_count[0] else ""

    # Period comes from the graph's canonical period-agent record.  Derive all
    # external status metadata from that record; never read a duplicate date
    # or period field from state.
    from earnings_agents.agent.period import require_detected_period
    try:
        detected_period = require_detected_period(final)
    except Exception:
        detected_period = None
    if final.get("_pending_replace"):
        label = final.get("_replace_period_label", "?")
        if status == "saved":
            # The deferred replace already ran inside mongodb_save (delete +
            # upsert) — report it in the past tense so it doesn't read like a
            # second save/extraction is still pending.
            pub.publish(
                "progress",
                f"replaced {label} — existing data replaced "
                f"(deferred delete + upsert)",
            )
        else:
            pub.publish(
                "progress",
                f"{label} not saved — existing data preserved "
                f"(replace deferred to a successful save)",
            )

    if status == "saved":
        n = len(final.get("concept_metrics") or {})
        period_out = (
            detected_period.period_end.isoformat()
            if detected_period is not None else ""
        )
        year = str(detected_period.fiscal_year) if detected_period else "?"
        summary = f"✓ {ticker}_{year}_latest saved  ({n} concepts){llm_tag}  {elapsed_str}"
        pub.publish("summary", summary, kind="summary")
        # Use the exact canonical period context produced by the period agent.
        # Do not re-query the latest database row: another job could win that
        # race and relabel this filing with a different period.
        _period_label: str | None = None
        if detected_period is not None:
            from earnings_agents.agent.period import format_period_label
            _period_label = format_period_label(detected_period)
        # Store in payload so main() passes it to _update_load_request_status.
        if _period_label:
            payload["_sec_period_label"] = _period_label
        logger.info(
            "8-K saved for %s — period=%s  concepts=%d  llm_calls=%d  elapsed=%s",
            ticker, period_out, n, llm_call_count[0], elapsed_str,
        )
        return True

    if status == "skipped":
        logger.info("8-K skipped for %s — %s", ticker, status)
        return True

    logger.warning(
        "8-K pipeline ended with status=%s  error=%s  for %s",
        status, final.get("error"), ticker,
    )
    return False


# ── CLI argument parsing ───────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="earnings-8k-worker",
        description="Consume 8-K filing jobs from Redis and run the earnings extraction pipeline.",
    )
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", REDIS_URL))
    parser.add_argument("--queue-name", default=os.getenv("REDIS_QUEUE_NAME", _DEFAULT_QUEUE))
    parser.add_argument("--dead-letter-queue", default=os.getenv("REDIS_DEAD_LETTER_QUEUE", "sec:filings:dlq:8k"))
    parser.add_argument("--poll-timeout", type=int, default=5,
                        help="Seconds to block-wait for a Redis message.")
    parser.add_argument("--max-attempts", type=int,
                        default=int(os.getenv("REDIS_MAX_ATTEMPTS", "3")),
                        help="Retry limit before moving a job to the dead-letter queue.")
    parser.add_argument("--retry-delay", type=int,
                        default=int(os.getenv("REDIS_RETRY_DELAY_SECONDS", "5")),
                        help="Seconds to wait between retries.")
    parser.add_argument("--once", action="store_true",
                        help="Process one message then exit (useful for testing).")
    return parser.parse_args(argv)


# ── Main loop ──────────────────────────────────────────────────────────────────

def _make_client(redis_url: str, poll_timeout: int) -> Redis:
    """Create a Redis client suitable for blocking blpop.

    socket_timeout must be None (no socket-level deadline) so the blocking
    BLPOP command can wait the full poll_timeout seconds without the socket
    layer raising TimeoutError.  socket_connect_timeout is kept short so a
    bad URL fails fast at startup.
    """
    return Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=None,  # no socket-level timeout — blpop controls waiting
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    queue_name: str = args.queue_name
    dead_letter_queue: str = args.dead_letter_queue
    redis_url: str = args.redis_url

    # Convert SIGTERM (docker stop / docker-compose down) into SystemExit so it
    # propagates through finally blocks and the except BaseException handler in
    # _process_payload — this lets us publish a failure event and mark the DB
    # record as failed before the process exits.
    def _sigterm(_signum, _frame):  # noqa: ANN001
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    client = _make_client(redis_url, args.poll_timeout)
    graph = build_graph()
    logger.info("8-K worker listening on Redis queue '%s'", queue_name)

    while True:
        try:
            item = client.blpop(queue_name, timeout=args.poll_timeout)
        except Exception as exc:
            logger.warning("Redis blpop error (%s) — reconnecting in 5s", exc)
            time.sleep(5)
            try:
                client = _make_client(redis_url, args.poll_timeout)
            except Exception:
                pass
            continue

        if not item:
            if args.once:
                break
            continue

        _, raw_message = item
        try:
            payload: dict[str, Any] = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.error("Could not decode queue message: %r", raw_message)
            if args.once:
                break
            continue

        form_type = (payload.get("filing_type") or payload.get("form_type") or "").upper()

        if form_type != "8-K":
            # Dedicated queue: only 8-K messages should arrive here.
            # Log and skip anything unexpected without re-queuing.
            logger.warning("Unexpected form_type %r on 8-K queue — skipping", form_type)
            if args.once:
                break
            continue

        # ── Process the 8-K ───────────────────────────────────────────────────
        attempts = int(payload.get("attempts") or 0)
        success = False
        _update_load_request_status(payload, "processing")

        try:
            success = _process_payload(graph, payload)
        except (KeyboardInterrupt, SystemExit):
            # Worker is shutting down (Ctrl-C or docker stop) mid-job.
            # _process_payload already published the ✗ summary event; we just
            # need to persist the failed status before the process exits.
            _update_load_request_status(payload, "failed")
            logger.info(
                "Worker shutdown mid-job — marked %s as failed",
                payload.get("ticker"),
            )
            raise  # let the process exit normally
        except Exception as exc:  # noqa: BLE001
            payload["last_error"] = str(exc)
            logger.exception("Unhandled error processing 8-K job for %s", payload.get("ticker"))

        if success:
            # Write sec_period_of_report so the pipeline table shows the
            # period.  _sec_period_label (e.g. "FY2026 Q1" / "FY2026 Annual")
            # is set by _process_payload after reading the agent-selected
            # collection.
            # Do not fall back to EDGAR/report dates here; a completed job
            # without the agent label must remain unlabeled, not misclassified.
            _period = payload.pop("_sec_period_label", None) or ""
            _update_load_request_status(payload, "completed", period_of_report=_period)
        else:
            payload["attempts"] = attempts + 1
            payload["failed_at"] = time.time()
            if payload["attempts"] < args.max_attempts:
                logger.warning(
                    "8-K job failed; retrying attempt %d/%d for %s",
                    payload["attempts"], args.max_attempts, payload.get("ticker"),
                )
                time.sleep(max(0, args.retry_delay))
                client.rpush(queue_name, serialize_message(payload))
            else:
                logger.error(
                    "8-K job exhausted %d retries; moving to dead-letter queue '%s' for %s",
                    args.max_attempts, dead_letter_queue, payload.get("ticker"),
                )
                _update_load_request_status(payload, "failed")
                client.rpush(dead_letter_queue, serialize_message(payload))

        if args.once:
            break


if __name__ == "__main__":
    main()
