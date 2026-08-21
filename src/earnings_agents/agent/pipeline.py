"""Pi-style agent document pipeline node.

The document is already fetched and converted to plain text by
``fetch_filing_node``; this node runs the extraction agent over that text with
navigation tools.  No table extraction, no markdown conversion, no
classification — the agent reads the document like pi reads source code.
"""
from __future__ import annotations

import logging
from typing import Any

from earnings_agents.agent.loop import run_agent_loop
from earnings_agents.agent.period import require_detected_period
from earnings_agents.agent.industry import (
    build_industry_context,
    normalize_company_industry,
)
from earnings_agents.agent.derive import (
    load_prior_values,
    map_concepts,
    semantically_map_unmapped_metrics,
    prescan_document,
    SCALE_MULTIPLIERS,
)
from earnings_agents.agent.prompts import (
    PIPELINE_SYSTEM_PROMPT,
    COMPANY_IDENTITY_RULE,
    build_concept_list,
    build_retry_briefing,
)
from earnings_agents.agent.tools import build_pi_tools
from earnings_agents.config import EXTRACTION_MAX_CHARS
from earnings_agents.state import EarningsAgentState

logger = logging.getLogger(__name__)


# ── Main node ────────────────────────────────────────────────────────────────

def classify_missing_labels(
    missing_ids: set[str],
    target_concepts: list[dict],
) -> tuple[list[str], list[str]]:
    """Split missing concept labels into ``(toplevel, segments)``.

    Dimensionality comes from the DB flags ``dimension`` /
    ``dimension_concept`` — NOT from ``"|"`` in the taxonomy key, which only
    fires for embedded member-tag labels and also appears on path-disambiguated
    non-dimensional rows.
    """
    toplevel = [
        c["label"] for c in target_concepts
        if c["_id"] in missing_ids
        and not (c.get("dimension") or c.get("dimension_concept"))
    ]
    segments = [
        c["label"] for c in target_concepts
        if c["_id"] in missing_ids
        and (c.get("dimension") or c.get("dimension_concept"))
    ]
    return toplevel, segments



def agent_document_pipeline_node(state: EarningsAgentState) -> EarningsAgentState:
    """Pi-style agent pipeline: plain text + navigation tools.

    1. Document text is already in state (fetch_filing_node)
    2. Agent navigates with search/read_lines/verify tools
    3. Post-processing: validate, map, derive (deterministic)
    """
    from earnings_agents.hooks import report_call

    ticker = state["ticker"]
    target_concepts: list[dict] = state.get("target_concepts") or []  # type: ignore[assignment]

    if not target_concepts:
        return {**state, "status": "failed", "error": f"No target concepts for {ticker}"}

    plain_text = state.get("raw_text") or ""
    if not plain_text:
        return {**state, "status": "failed", "error": "No document text in state (fetch_filing missing?)"}

    attempt_num = state.get("extraction_attempts", 0) + 1
    logger.info("Agent pipeline pass %d for %s", attempt_num, ticker)

    from earnings_agents.config import LLM_PROVIDER as _LLM_PROVIDER
    report_call(f"  [pipeline]  🧠 AGENT pipeline  ({_LLM_PROVIDER or 'llm'})")

    # ── 1. Document pre-scan (scale) — deterministic, on text already in state ──
    doc_scale, _ = prescan_document(plain_text)
    n_lines = plain_text.count("\n") + 1
    report_call(f"  [agent doc]  {len(plain_text):,} chars, {n_lines:,} lines → agent")

    # ── 2. Load prior values ─────────────────────────────────────────────
    cik = state.get("cik")
    try:
        period = require_detected_period(state)
    except Exception as exc:
        return {
            **state,
            "extraction_attempts": attempt_num,
            "status": "failed",
            "error": f"Agent pipeline: invalid period-agent result for {ticker}: {exc}",
        }
    prior_values = load_prior_values(target_concepts, cik, period)

    dollar_multiplier = SCALE_MULTIPLIERS.get(doc_scale, 1) if doc_scale else 1

    # ── 3. Build prompt ──────────────────────────────────────────────────
    concept_list_str = build_concept_list(
        target_concepts,
        recent_concept_ids=set(state.get("recent_concept_ids") or []),
        calculated_concepts=state.get("calculated_concepts"),
    )

    retry_briefing = ""
    if attempt_num > 1:
        findings: list[dict] = state.get("findings") or []
        prev_metrics: dict[str, Any] = state.get("metrics") or {}
        retry_briefing = build_retry_briefing(
            findings, prev_metrics,
            missing_toplevel=state.get("missing_toplevel_labels") or [],
            missing_segments=state.get("missing_segment_labels") or [],
        )
        logger.info("Agent retry briefing for %s (pass %d):\n%s", ticker, attempt_num, retry_briefing[:500])

    hints_parts: list[str] = []
    period_type = period.period_type
    fy_code = state.get("fiscal_year_end_code") or ""
    fy_hint = f" (fiscal year ends {fy_code})" if fy_code else ""
    label_hint = (
        f' — column header: "{period.period_label}"'
        if period.period_label else ""
    )
    hints_parts.append(
        f"PERIOD: {period_type} filing, period-end {period.period_end.isoformat()}"
        f"{fy_hint}. Extract from the {period_type} column (most recent/latest)"
        f"{label_hint}. If the document shows both a Q4 and a fiscal-year "
        f"column, extract the fiscal-year (annual) column — never Q4."
    )
    if retry_briefing:
        hints_parts.append(
            f"⚠  RETRY — PASS {attempt_num} — TARGETED FIX ONLY\n\n"
            f"{retry_briefing}\n\n"
            f"SCOPE: You are ONLY fixing the metrics listed above under "
            f"WRONG METRICS and MISSING.  Return ONLY those metrics in your "
            f"finalize_extraction JSON.  All other metrics from the previous "
            f"pass are already correct and will be carried forward automatically.\n"
            f"Do NOT re-extract or re-read the full document for correct metrics."
        )
    elif state.get("extraction_notes"):
        hints_parts.append(f"PRIOR ATTEMPT FAILURES:\n{state['extraction_notes']}")
    hints_block = "\n\n".join(hints_parts) if hints_parts else ""

    # ── Industry context — advisory SIC data injected on EVERY pass ─────
    company_industry = state.get("company_industry")
    industry_context = build_industry_context(company_industry)
    industry_profile = normalize_company_industry(company_industry)
    if industry_profile:
        report_call(
            f"  [industry]  context injected — SIC {industry_profile['sic_code']} "
            f"({industry_profile['sic_description'][:60]})"
        )

    system_prompt = (
        PIPELINE_SYSTEM_PROMPT.format(concept_list=concept_list_str)
        + f"\n\nCOMPANY: {state['company_name']} ({ticker})\nATTEMPT: {attempt_num}\n\n"
        + COMPANY_IDENTITY_RULE.format(
            company_name=state["company_name"], ticker=ticker,
        )
        + "\n\n"
        + industry_context
    )
    if hints_block:
        system_prompt += f"\n\n{hints_block}"

    # ── 4. Build tools and run agent ─────────────────────────────────────
    tools = build_pi_tools(
        plain_text, prior_values, cik=state.get("cik"),
        company_name=state["company_name"],
        company_industry=company_industry,
        document_map=state.get("document_map"),
    )

    initial_msg: str
    if attempt_num > 1 and retry_briefing:
        initial_msg = (
            "TARGETED RETRY — fix ONLY the metrics flagged in the system prompt. "
            "Use search() to find them, read_lines() to read the section, "
            "calculate() to compute any derived values, verify_identity() to check. "
            "Return ONLY the corrected metrics — the rest are carried forward automatically."
        )
    else:
        initial_msg = (
            f"This is a {len(plain_text):,}-character SEC earnings press release "
            f"with {n_lines:,} lines.  Start by searching for the income statement: "
            f'search("Revenue") or search("Net income") to locate it, then '
            f"read_lines() to extract metrics.  Verify before finalizing."
        )

    final_result = run_agent_loop(
        system_prompt=system_prompt,
        initial_message=initial_msg,
        tools=tools,
        ticker=ticker,
        dollar_multiplier=dollar_multiplier,
    )

    if final_result is None:
        return {
            **state,
            "extraction_attempts": attempt_num,
            "status": "failed",
            "error": f"Agent pipeline produced no result for {ticker}",
        }

    # ── 4b. Company-identity gate ──────────────────────────────────────
    # A manual filing URL (or any source) may point at a document that does
    # NOT belong to the ticker (observed live: Netflix shareholder letter fed
    # with ticker ORCL).  The extraction agent is instructed to flag
    # "__company_mismatch__" instead of extracting another company's numbers.
    # The gate HARD-FAILS the run — nothing is mapped, derived, or saved.
    if final_result.pop("__company_mismatch__", False):
        report_call(
            f"  [pipeline]  ✗ document does not belong to {ticker} — "
            f"aborting (company mismatch)"
        )
        logger.warning(
            "Company mismatch for %s — document is for another company; aborting",
            ticker,
        )
        return {
            **state,
            "extraction_attempts": attempt_num,
            "status": "failed",
            "error": (
                f"Document does not match ticker {ticker} — extraction "
                f"aborted (company mismatch)"
            ),
        }

    # ── 5. Merge, validate, map ─────────────────────────────────────────
    # The agent computed derived metrics itself via calculate() — no
    # deterministic derivation step needed.  The agent returns extracted
    # values AND computed values in the same JSON.  We detect which
    # concepts were computed (present in output but in calculated_concepts)
    # and mark them as derived.
    metrics: dict[str, Any] = final_result
    if attempt_num > 1:
        prev_metrics = state.get("metrics")
        if isinstance(prev_metrics, dict):
            metrics = {**prev_metrics, **metrics}

    # Log extracted values for debugging
    extracted_keys = [k for k in metrics if not k.startswith("__")]
    logger.info("Agent extracted %d keys for %s:", len(extracted_keys), ticker)
    for k in extracted_keys:
        v = metrics[k]
        if isinstance(v, (int, float)):
            logger.info("  %s = %s", k, f"{v:,.0f}")
        else:
            logger.info("  %s = %s (non-numeric)", k, str(v)[:80])

    # Pop out the derived-marker field the agent returns (if any)
    metrics.pop("__derived__", None)

    concept_metrics, _reverse_map, mapped_keys = map_concepts(metrics, target_concepts)

    # Exact taxonomy/label mapping is preferred.  Resolve only leftover
    # numeric keys semantically (without changing their values) so filing
    # wording such as "Revenue" vs "Total revenue" does not silently vanish.
    concept_metrics, semantic_mapped_keys = semantically_map_unmapped_metrics(
        metrics, target_concepts, concept_metrics,
    )
    mapped_keys.update(semantic_mapped_keys)

    # ── Deterministic derivation ───────────────────────────────────────
    from earnings_agents.agent.derive import derive_missing_concepts
    concept_metrics, derived_ids = derive_missing_concepts(
        concept_metrics, target_concepts,
    )

    # ── 6. Return state ──────────────────────────────────────────────────
    raw_text = plain_text[:EXTRACTION_MAX_CHARS]

    mapped_ids = set(concept_metrics.keys()) - derived_ids
    all_target_ids = {c["_id"] for c in target_concepts}
    missing_ids = all_target_ids - mapped_ids - derived_ids
    missing_labels = [c["label"] for c in target_concepts if c["_id"] in missing_ids]
    missing_toplevel, missing_segments = classify_missing_labels(missing_ids, target_concepts)

    logger.info(
        "Agent pipeline for %s: %d metrics, %d mapped, %d derived, %d not in filing",
        ticker,
        len([k for k in metrics if not k.startswith("__")]),
        len(concept_metrics) - len(derived_ids),
        len(derived_ids),
        len(missing_labels),
    )

    return {
        **state,
        "raw_text": raw_text,
        "metrics": metrics,
        "concept_metrics": concept_metrics,
        "derived_concept_ids": list(derived_ids),
        "mapped_metric_keys": list(mapped_keys),
        "missing_concept_labels": missing_labels,
        "missing_toplevel_labels": missing_toplevel,
        "missing_segment_labels": missing_segments,
        "extraction_attempts": attempt_num,
        "status": "extracted",
    }
