"""Period existence check + deferred replace scheduling.

Runs AFTER the period agent (``detect_period``) — the check has a
trustworthy ``(fiscal_year, quarter)`` derived from the agent-read period end.

No accession checks — only exact-period existence:

  • same fiscal period stored  → schedule ``_pending_replace`` and CONTINUE —
    the delete itself stays deferred to ``mongodb_save``, which removes the
    stale period docs immediately before the upsert (no data-loss window)
  • otherwise                  → proceed

Period typing comes from the agent: an ANNUAL period is checked and replaced
in ``concept_values_annual`` only — a quarterly Q4 record for the same fiscal
year is never checked, deleted, or touched.
"""
from __future__ import annotations

import logging

from earnings_agents.integrations.normalize import fiscal_period_exists
from earnings_agents.state import EarningsAgentState
from earnings_agents.agent.period import (
    format_period_label,
    require_detected_period,
)

logger = logging.getLogger(__name__)


def check_period_node(state: EarningsAgentState) -> EarningsAgentState:
    from earnings_agents.hooks import report_call

    cik = state.get("cik")
    try:
        period = require_detected_period(state)
    except Exception as exc:
        return {**state, "status": "failed", "error": f"check_period: {exc}"}

    if not cik or not period.fiscal_year:
        return {
            **state,
            "status": "failed",
            "error": "check_period: missing cik / agent-detected fiscal year",
        }

    period_label = period.period_label or ""

    if fiscal_period_exists(cik, period):
        label = format_period_label(period)
        report_call(
            f"  [check]  period {label} already stored — will replace after successful save"
        )
        return {
            **state,
            "_pending_replace": {"cik": cik},
            "_replace_period_label": label,
        }

    report_call(
        f"  [check]  new period — {period_label or '?'} "
        f"({format_period_label(period)})"
    )
    return {**state, "_pending_replace": None}
