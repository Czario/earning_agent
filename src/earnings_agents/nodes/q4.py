"""Post-save Q4 derivation node — income statement only.

Runs after ``mongodb_save`` when an ANNUAL filing was saved: derives Q4
quarterly values (``Q4 = Annual − (Q1 + Q2 + Q3)``) for the just-saved income
statement concepts and inserts them into ``concept_values_quarterly``
(the same scope as the calculations project's
``--calculate-q4 --statement is``).

Guards (all must hold for the derivation to run):
  • ``CALCULATE_Q4_AFTER_ANNUAL`` is enabled (config, default on)
  • the run just saved (``status == "saved"``)
  • the saved period is ANNUAL (Q4 derives from annual filings only — a
    quarterly filing never triggers it)
  • CIK + saved concepts are present

Never fails the run: Q4 derivation is best-effort.  The summary lands in
``state["q4_calculation"]`` (observability) and every event is surfaced via
``[q4]`` report lines.
"""
from __future__ import annotations

import logging

from earnings_agents.agent.period import format_period_label, require_detected_period
from earnings_agents.state import EarningsAgentState

logger = logging.getLogger(__name__)


def calculate_q4_node(state: EarningsAgentState) -> EarningsAgentState:
    from earnings_agents import config as _config
    from earnings_agents.hooks import report_call
    from earnings_agents.integrations.q4 import calculate_q4_for_period

    if not _config.CALCULATE_Q4_AFTER_ANNUAL:
        report_call("  [q4]  skipped — CALCULATE_Q4_AFTER_ANNUAL is disabled")
        return state

    if state.get("status") != "saved":
        return state

    cik = state.get("cik")
    concept_metrics = state.get("concept_metrics") or {}
    if not cik or not concept_metrics:
        report_call("  [q4]  skipped — no CIK / no saved concepts")
        return state

    try:
        period = require_detected_period(state)
    except Exception as exc:  # noqa: BLE001
        report_call(f"  [q4]  skipped — invalid detected period: {exc}")
        return {**state, "q4_calculation": {"status": "skipped", "error": str(exc)}}

    if period.period_type != "annual":
        report_call(
            f"  [q4]  skipped — {format_period_label(period)} is not annual; "
            "Q4 derives from annual filings only"
        )
        return state

    # When the annual save REPLACED an existing period, stale Q4 values from
    # the previous save must be recomputed (upsert, write-first) — never
    # skipped because they already exist.
    recalculate = bool(state.get("_pending_replace"))
    report_call(
        f"  [q4]  deriving Q4 from annual save — CIK {cik} "
        f"{format_period_label(period)} ({len(concept_metrics)} concept(s)"
        f"{' [replace — recomputing]' if recalculate else ''})"
    )

    try:
        result = calculate_q4_for_period(
            cik=cik,
            period=period,
            annual_concept_ids=list(concept_metrics.keys()),
            statement_type="income",
            allow_incomplete=_config.Q4_ALLOW_INCOMPLETE,
            recalculate=recalculate,
        )
    except Exception as exc:  # noqa: BLE001 — never fails the run
        logger.exception("calculate_q4_node: Q4 derivation failed for CIK %s", cik)
        report_call(f"  [q4]  ✗ derivation failed: {exc}")
        return {
            **state,
            "q4_calculation": {"status": "error", "error": str(exc)},
        }

    report_call(
        f"  [q4]  ✓ {result['calculated']} flow + {result['point_in_time']} "
        f"point-in-time Q4 value(s) inserted; {result['skipped']} skipped"
        f"{f'; {len(result['errors'])} error(s)' if result['errors'] else ''}"
    )
    for reason, n in sorted(result["skipped_reasons"].items()):
        report_call(f"  [q4]    · skipped {n}: {reason}")
    for err in result["errors"][:5]:
        report_call(f"  [q4]    · error: {err}")
    logger.info(
        "calculate_q4_node: CIK %s FY%d — %d calculated, %d point-in-time, "
        "%d skipped (%s)",
        cik, period.fiscal_year,
        result["calculated"], result["point_in_time"], result["skipped"],
        "; ".join(f"{k}={v}" for k, v in sorted(result["skipped_reasons"].items())),
    )
    return {**state, "q4_calculation": result}
