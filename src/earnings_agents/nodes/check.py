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
from datetime import date

from earnings_agents.integrations.normalize import (
    compute_fiscal_period,
    fiscal_period_exists,
)
from earnings_agents.state import EarningsAgentState

logger = logging.getLogger(__name__)


def check_period_node(state: EarningsAgentState) -> EarningsAgentState:
    from earnings_agents.hooks import report_call

    cik = state.get("cik")
    fy_end_month = state.get("fiscal_year_end_month")
    sec_report_date = state.get("sec_report_date")

    if not cik or not fy_end_month or not sec_report_date:
        return {
            **state,
            "status": "failed",
            "error": "check_period: missing cik / fiscal_year_end_month / period",
        }

    try:
        period_end = date.fromisoformat(sec_report_date)
    except ValueError:
        return {
            **state,
            "status": "failed",
            "error": f"check_period: bad period date {sec_report_date!r}",
        }

    period_label = state.get("period_label") or ""
    period_type = state.get("detected_period_type") or "quarterly"

    # Exact-period existence check ONLY — no accession checks.  When the same
    # fiscal period is already stored, schedule a deferred replace (no delete
    # here).  Period typing comes from the agent: an annual period is checked
    # and replaced in concept_values_annual only — a quarterly Q4 record for
    # the same fiscal year is NEVER checked, deleted, or touched.
    fiscal_year, quarter = compute_fiscal_period(
        period_end, fy_end_month, period_label,
    )
    q_arg = quarter if period_type == "quarterly" else None

    if fiscal_period_exists(cik, fiscal_year, q_arg):
        label = (
            f"FY{fiscal_year} Q{quarter}" if q_arg is not None
            else f"FY{fiscal_year} (annual)"
        )
        report_call(
            f"  [check]  period {label} already stored — will replace after successful save"
        )
        return {
            **state,
            "_pending_replace": {"cik": cik, "fiscal_year": fiscal_year, "quarter": q_arg},
            "_replace_period_label": label,
        }

    report_call(
        f"  [check]  new period — {period_label or sec_report_date} "
        f"(FY{fiscal_year}{f' Q{quarter}' if q_arg is not None else ' annual'})"
    )
    return {**state, "_pending_replace": None}
