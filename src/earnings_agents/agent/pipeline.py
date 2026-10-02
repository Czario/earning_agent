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
    build_calc_derivation_block,
    build_extraction_summary,
    build_no_scale_keys,
    extract_financial_statements_section,
    extract_is_section,
    map_concepts,
    semantically_map_unmapped_metrics,
    prescan_document,
    SCALE_MULTIPLIERS,
)
from earnings_agents.agent.currency import usd_metadata, is_usd_safe
from earnings_agents.agent.indexer import build_section_index, format_section_index
from earnings_agents.agent.prompts import (
    PIPELINE_SYSTEM_PROMPT,
    COMPANY_IDENTITY_RULE,
    build_concept_list,
)
from earnings_agents.agent.tools import build_pi_tools
from earnings_agents.config import EXTRACTION_MAX_CHARS, DEACCUMULATE_YTD_CASHFLOW
from earnings_agents.state import EarningsAgentState

logger = logging.getLogger(__name__)


def _hint_covers_targets(hint: dict, target_statements: list[str]) -> bool:
    """True when a prescan routing hint covers every requested statement.

    ``hint`` is one entry from ``prescan_document``'s hints list (keys
    ``income``/``balance_sheet``/``cash_flow`` booleans); ``target_statements``
    is ``["income", "balancesheet", "cashflow"]`` or a subset.  A hint only
    "covers" a statement when its exhibit actually contains that statement's
    table (keyword-strong), so the indexer is skipped only when the best
    exhibit holds all the statements we need.
    """
    wanted = {s.strip().lower() for s in target_statements}
    if "income" in wanted and not hint.get("income"):
        return False
    if "balancesheet" in wanted and not hint.get("balance_sheet"):
        return False
    if "cashflow" in wanted and not hint.get("cash_flow"):
        return False
    return True


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


# ── Completeness findings (agent-driven) ─────────────────────────────────────

def check_company_identity(expected_name: str, reported_name: str) -> bool:
    """Deterministic company-identity cross-check.

    Compares the company name the extraction agent read from the document
    (``__company_name__``) against the expected ticker's company name using
    normalized token overlap.  Tokens shorter than 4 characters (``The``,
    ``Co``, ``Inc``, ``Ltd``) are dropped so suffix/abbreviation variants
    ("Inc." vs "Incorporated", "Holdings Inc" vs "Holdings Limited") cannot
    force a false mismatch; a *disjoint* token set means the document is for a
    different company (wrong-document upload — e.g. a Netflix shareholder
    letter fed with ticker ORCL).

    Returns True when the names plausibly match, or when either side is
    missing/empty (the check is advisory and must never fail a run on its own
    absence).
    """
    expected_tokens = {
        w for w in (expected_name or "").strip().lower().replace(",", " ").split()
        if len(w) >= 4
    }
    reported_tokens = {
        w for w in (reported_name or "").strip().lower().replace(",", " ").split()
        if len(w) >= 4
    }
    if not expected_tokens or not reported_tokens:
        return True  # nothing to compare — not a mismatch
    return not expected_tokens.isdisjoint(reported_tokens)


def _build_completeness_findings(
    currency_meta: dict,
    document_map: list[dict] | None,
    missing_metric_keys: list[str] | None,
) -> list[dict]:
    """Build structured findings from the agent's own reports.

    The tool-calling agent is the authority on what it could and could not
    extract across thousands of heterogeneous filing formats.  Only two
    conditions are promoted to high severity (blocking under STRICT_ACCURACY):
    a confirmed non-USD document currency, and an incomplete/truncated exhibit.
    Concepts the agent searched for but could not locate are recorded as
    medium-severity observability — a rigid hard-coded taxonomy check would
    misfire across custom/IFRS/company-specific labels.
    """
    findings: list[dict] = []

    currency = (currency_meta or {}).get("currency")
    if currency != "USD":
        findings.append({
            "type": "unresolved_currency",
            "severity": "high",
            "message": (
                f"Document currency is {currency}, not USD — "
                "refusing to save non-USD monetary values"
            ),
            "evidence": {"currency_metadata": currency_meta},
        })

    for key in missing_metric_keys or []:
        findings.append({
            "type": "missing_concept",
            "severity": "medium",
            "message": f"Agent could not locate concept: {key}",
        })

    for d in document_map or []:
        reason = d.get("reason") or ""
        blocked = bool(d.get("error")) or bool(d.get("truncated"))
        blocked = blocked or (bool(d.get("skipped")) and reason == "total size budget exhausted")
        if blocked:
            findings.append({
                "type": "incomplete_document",
                "severity": "high",
                "message": (
                    f"Exhibit incomplete: {d.get('exhibit') or d.get('url')} "
                    f"({d.get('error') or reason or 'truncated'})"
                ),
            })

    return findings


def _run_extraction_pass(
    state: EarningsAgentState,
    plain_text: str,
    target_concepts: list[dict],
) -> EarningsAgentState:
    """Run ONE extraction-agent pass and post-process its output.

    Returns the updated state with ``status="extracted"`` on success, or a
    failed/skipped state.  ``metrics``/``concept_metrics``/``value_metadata_by_id``
    are set from this single pass.
    """
    from earnings_agents.hooks import report_call

    ticker = state["ticker"]

    # ── 1. Document pre-scan (scale + statement routing) — deterministic ──
    # Currency is NOT decided here: the tool-calling agent inspects each
    # table/section with detect_currency() and reports __currency__ itself.
    doc_scale, _, is_hints = prescan_document(
        plain_text, document_map=state.get("document_map"),
    )
    n_lines = plain_text.count("\n") + 1
    report_call(f"  [agent doc]  {len(plain_text):,} chars, {n_lines:,} lines → agent")
    if is_hints:
        for h in is_hints:
            report_call(
                f"  [prescan]  statements likely in {h['exhibit']} "
                f"(lines {h['lines'][0]}-{h['lines'][1]}; "
                f"IS {h['primary_count']}+{h['secondary_count']} ctx, "
                f"BS {h['bs_count']}, CF {h['cf_count']})"
            )

    # ── 1b. Section index — build only when prescan hints are insufficient ──
    # When the prescan found an exhibit containing ALL target statements with
    # strong keyword signals, skip the expensive LLM indexer (60K chars →
    # ~90s) and let the agent navigate via the exhibit routing hint instead.
    # Only build the index when the best exhibit is missing a target statement
    # or no hint exists.  (state.document_sections is never populated by the
    # period agent anymore, so this is the only index build site.)
    from earnings_agents.config import TARGET_STATEMENTS
    target_statements = state.get("target_statements") or TARGET_STATEMENTS
    is_multi_statement = any(
        s in target_statements for s in ("balancesheet", "cashflow")
    )

    prebuilt_sections = state.get("document_sections")
    best_hint = is_hints[0] if is_hints else None
    if best_hint and _hint_covers_targets(best_hint, target_statements):
        report_call(
            f"  [prescan]  strong statement signal "
            f"({', '.join(target_statements)}) — skipping LLM indexer "
            f"(agent routed to {best_hint['exhibit']})"
        )
    elif not prebuilt_sections:
        try:
            prebuilt_sections, elapsed = build_section_index(
                plain_text, query="income statement and segment results",
            )
            report_call(
                f"  [index]  section map built — "
                f"{len(prebuilt_sections.get('sections') or [])} "
                f"section(s), coverage {prebuilt_sections.get('coverage')} "
                f"({elapsed:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("section index build failed: %s", exc)
            prebuilt_sections = None

    # ── 2. Extract the best exhibit text for pre-injection ─────────────
    cik = state.get("cik")
    try:
        period = require_detected_period(state)
    except Exception as exc:
        return {
            **state,
            "status": "failed",
            "error": f"Agent pipeline: invalid period-agent result for {ticker}: {exc}",
        }

    dollar_multiplier = SCALE_MULTIPLIERS.get(doc_scale, 1) if doc_scale else 1

    # Pre-read the primary financial statement exhibit so the agent gets the table(s)
    # upfront — eliminates 5+ read_lines navigation calls.
    is_exhibit_text: str | None = None
    is_exhibit_label: str = ""
    if is_hints:
        best = is_hints[0]
        doc_map = state.get("document_map") or []
        lines = plain_text.split("\n")
        for doc in doc_map:
            exhibit = doc.get("exhibit") or ""
            if exhibit == best["exhibit"]:
                ls = doc.get("line_start")
                le = doc.get("line_end")
                if ls and le:
                    exhibit_full = "\n".join(lines[ls - 1 : le])
                    if is_multi_statement:
                        fs_section = extract_financial_statements_section(exhibit_full)
                        if fs_section:
                            is_exhibit_text = fs_section
                            report_call(
                                f"  [prescan]  financial statements section extracted — {len(fs_section):,} chars "
                                f"(from {len(exhibit_full):,} exhibit)"
                            )
                        else:
                            is_exhibit_text = exhibit_full
                            report_call(
                                f"  [prescan]  financial statements section not isolated — using full exhibit "
                                f"({len(exhibit_full):,} chars)"
                            )
                    else:
                        is_section = extract_is_section(exhibit_full)
                        if is_section:
                            is_exhibit_text = is_section
                            report_call(
                                f"  [prescan]  IS section extracted — {len(is_section):,} chars "
                                f"(from {len(exhibit_full):,} exhibit)"
                            )
                        else:
                            is_exhibit_text = exhibit_full
                            report_call(
                                f"  [prescan]  IS section not found — using full exhibit "
                                f"({len(exhibit_full):,} chars)"
                            )
                    is_exhibit_label = exhibit
                    break

    # ── 3. Build prompt ──────────────────────────────────────────────────
    concept_list_str = build_concept_list(
        target_concepts,
        recent_concept_ids=set(state.get("recent_concept_ids") or []),
        calculated_concepts=state.get("calculated_concepts"),
    )
    # CALC (system:/calculated) concepts are computed by the agent in-loop via
    # compute() — never extracted from the filing.  This block also returns the
    # ambiguous hierarchy paths for the observability finding below.
    calc_block, ambiguous_paths = build_calc_derivation_block(target_concepts)

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
    hints_block = "\n\n".join(hints_parts) if hints_parts else ""

    # ── Guidance extraction block (PHASE 3) — injected when enabled ──────
    # The contract overrides the "WHAT TO IGNORE: forward-looking guidance"
    # line in the base system prompt and teaches the agent the
    # __guidance__ JSON contract.  Disabled → the agent ignores guidance
    # exactly as before (no extra LLM work, no contract in context).
    from earnings_agents.config import GUIDANCE_ENABLED
    guidance_block = ""
    if GUIDANCE_ENABLED:
        from earnings_agents.agent.prompts import GUIDANCE_CONTRACT_BLOCK
        guidance_block = GUIDANCE_CONTRACT_BLOCK

    # ── Industry context — advisory SIC data injected on EVERY pass ─────
    company_industry = state.get("company_industry")
    industry_context = build_industry_context(company_industry)
    industry_profile = normalize_company_industry(company_industry)
    if industry_profile:
        report_call(
            f"  [industry]  context injected — SIC {industry_profile['sic_code']} "
            f"({industry_profile['sic_description'][:60]})"
        )

    # ── Long-term memory — advisory hints + write tools ───────────────
    # GOAL: make future extraction FASTER and MORE ACCURATE.  The agent reads
    # stored hints (label aliases + layout) and records new ones via the
    # remember_alias / remember_layout tools.
    memory_block = ""
    memory_tools: list = []
    if cik:
        try:
            from earnings_agents.config import MEMORY_ENABLED
            if MEMORY_ENABLED:
                from earnings_agents.agent.memory import (
                    build_remember_alias_tool,
                    build_remember_currency_tool,
                    build_remember_layout_tool,
                    recall_memory,
                )
                # Extraction cares about aliases (mapping), layout (navigation)
                # and currency (trust USD without re-verifying).
                mem = recall_memory(cik, types={"label_alias", "layout", "currency"})
                memory_block = mem["local_block"]
                if memory_block:
                    report_call(f"  [memory]  advisory hints injected for {ticker}")
                memory_tools = [
                    build_remember_alias_tool(cik),
                    build_remember_layout_tool(cik),
                    build_remember_currency_tool(cik),
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory setup failed for %s: %s", ticker, exc)

    system_prompt = (
        PIPELINE_SYSTEM_PROMPT.format(concept_list=concept_list_str)
        + f"\n\nCOMPANY: {state['company_name']} ({ticker})\n\n"
        + COMPANY_IDENTITY_RULE.format(
            company_name=state["company_name"], ticker=ticker,
        )
        + "\n\n"
        + industry_context
    )
    if memory_block:
        system_prompt += f"\n\n{memory_block}"
    if hints_block:
        system_prompt += f"\n\n{hints_block}"
    if guidance_block:
        # Guidance contract placed EARLY (right after the period hints) so it
        # carries more weight than being buried at the very end of the prompt
        # — the base prompt's "WHAT TO IGNORE" guidance line is reworded to
        # defer to this block.
        system_prompt += f"\n\n{guidance_block}"
    if is_exhibit_text:
        # IS exhibit already in context — suppress navigation instructions
        # that would cause the agent to search for data it already has.
        pass
    else:
        system_prompt += (
            "\n\nMULTI-DOCUMENT FILINGS — the bundle may contain several exhibits "
            "(press release, presentation, supplemental).  Navigate SMARTLY, never "
            "linearly:\n"
            "  • get_document_info() FIRST — it lists each exhibit with its "
            "opening lines; identify the press release vs presentation vs "
            "supplemental without reading them.\n"
            "  • search() is GLOBAL across ALL documents — use it to jump straight "
            "to matches; each result shows which exhibit it is in.\n"
            "  • The income statement is usually in the press release or a "
            "supplemental exhibit, not a slide deck.  When several documents "
            "match, prefer the FULLEST statement (most line items).\n"
            "  • Verify the statement has the row chain (Revenue → Cost of revenue "
            "→ Gross profit → Operating income → Net income) before extracting."
        )
    if memory_tools:
        system_prompt += (
            "\n\nMEMORY GOAL — make future extraction FASTER and MORE "
            "ACCURATE.\n"
            "  • Record memory AS YOU GO — in the SAME step where you observe "
            "the fact: remember_alias in the step you map the concept, "
            "remember_currency in the step you call detect_currency, "
            "remember_layout in the step you locate the statements.  NEVER "
            "batch memory calls into a separate step at the end — trailing "
            "memory-only turns cost LLM steps for zero benefit.\n"
            "  • Call each remember_* tool AT MOST ONCE per distinct fact per "
            "run.  If the COMPANY MEMORY block above already contains the fact, "
            "or you already recorded it this run, do NOT call remember_* for it "
            "again.\n"
            "  • When map_concept() (or your reading) shows a filing label "
            "maps to a known concept whose name differs, call "
            "remember_alias(filing_label=..., canonical=...) — the tool "
            "rejects identical labels and member/segment metadata.\n"
            "  • Record stable NAVIGATIONAL layout facts via "
            "remember_layout(note=...) — which exhibit/section the income "
            "statement lives in (e.g. 'second half of the release', "
            "'Exhibit 99.2 supplemental').  Record AT MOST ONE layout note per "
            "run (the single most useful navigational fact).  NEVER include "
            "per-filing line numbers — they change every filing.\n"
            "  • NEVER record absences, dimension members/segments (handled "
            "by taxonomy keys), or filing-content descriptions (e.g. how "
            "revenue is disaggregated) that a future run re-reads anyway.\n"
            "  • After detect_currency() confirms the document currency, call "
            "remember_currency(currency='USD', other_codes=...) — pass any "
            "OTHER currencies the filing contains (from the 'Codes "
            "detected:' list) so future runs expect them and still extract "
            "USD."
        )
    if calc_block:
        system_prompt += (
            "\n\nCOMPUTE-ONLY CONCEPTS — these are NOT printed in the filing, "
            "so do NOT search the document for them.  After you have extracted "
            "ALL printed rows, compute each one below with compute() and "
            "include the result in finalize_extraction under its EXACT "
            "bracketed key.  List every computed key in __derived__ "
            "(comma-separated bracketed keys).\n"
            + calc_block
        )
    if DEACCUMULATE_YTD_CASHFLOW and "cashflow" in target_statements:
        from earnings_agents.agent.prompts import CF_DEACCUMULATION_BLOCK
        system_prompt += f"\n\n{CF_DEACCUMULATION_BLOCK}"

    # ── 4. Build tools and run agent ─────────────────────────────────────
    # The section map built by the period pass (find_sections) rides in state;
    # find_sections in THIS loop returns it instantly and the initial message
    # points the agent straight at the income-statement range.
    tools = build_pi_tools(
        plain_text, cik=state.get("cik"),
        company_name=state["company_name"],
        company_industry=company_industry,
        document_map=state.get("document_map"),
        target_concepts=target_concepts,
        prebuilt_sections=prebuilt_sections,
        prescan_is_hints=is_hints,
        fiscal_year=getattr(period, "fiscal_year", None),
        quarter=getattr(period, "quarter", None),
        period=period,
    )
    for t in memory_tools:
        tools.append(t)

    if is_exhibit_text:
        # Pre-injected exhibit text — agent extracts directly
        # instead of navigating with read_lines.
        is_section_size = len(is_exhibit_text)
        is_line_start = is_hints[0]["lines"][0] if is_hints else 1
        is_line_end = is_hints[0]["lines"][1] if is_hints else n_lines
        if is_multi_statement:
            target_desc = ", ".join(target_statements)
            # Which statements are actually inside the injected text (the
            # prescan marks them per exhibit; a split multi-exhibit filing may
            # leave one statement in another exhibit).
            present_names: list[str] = []
            missing_names: list[str] = []
            for key, label in (
                ("income", "Income Statement"),
                ("balance_sheet", "Balance Sheet"),
                ("cash_flow", "Cash Flow Statement"),
            ):
                if best_hint and best_hint.get(key):
                    present_names.append(label)
                elif key in ("balance_sheet", "cash_flow"):
                    missing_names.append(label)
            present_txt = ", ".join(present_names) if present_names else "the financial statements"
            missing_txt = ", ".join(missing_names)
            injected_note = (
                f"The text above already contains the {present_txt} tables.  "
                f"Treat them as ALREADY READ: do NOT call read_lines() again on "
                f"those table ranges — extract their rows directly from the text "
                f"above."
            )
            if missing_names:
                injected_note += (
                    f"  The {missing_txt} is NOT in the text above — locate it "
                    f"with search()/read_lines() across the other exhibits in "
                    f"PHASE 2."
                )
            # Cash-flow de-accumulation step (inlined into PHASE 1 so it is
            # followed — the agent ignores late system-prompt blocks).
            if DEACCUMULATE_YTD_CASHFLOW and "cashflow" in target_statements:
                from earnings_agents.agent.deaccumulate import detect_cashflow_statement_basis
                prescan_cf = detect_cashflow_statement_basis(plain_text)
                is_cf_ytd = bool(
                    prescan_cf
                    and prescan_cf.get("basis") in ("ytd_6m", "ytd_9m")
                    and getattr(period, "quarter", None) in (2, 3)
                )
                if is_cf_ytd:
                    m = prescan_cf["months"]
                    b = prescan_cf["basis"]
                    ev = prescan_cf["evidence"]
                    cf_step = (
                        f"     - Cash Flow: ⚠️ MANDATORY DE-ACCUMULATION REQUIRED:\n"
                        f"       The statement is {m}-month YEAR-TO-DATE (\"{ev}\").\n"
                        f"       Printed flow numbers are CUMULATIVE for {m} months, NOT the Q{period.quarter} quarter!\n"
                        f"       For EVERY flow row call: deaccumulate_cashflow(value=<CURRENT-period printed value>,\n"
                        f"       concept_key=<bracketed key>, ytd_months={m}, filing_label=<row label>)\n"
                        f"       and report the RETURNED quarterly value (DO NOT report raw cumulative values).\n"
                        f"       * \"...ending balances\" → use printed value as-is (quarter-end snapshot).\n"
                        f"       * \"...beginning balances\" → SKIP.\n"
                        f"       * List EVERY de-accumulated key in __derived__.\n"
                        f"       * In finalize_extraction, report \"__cashflow_basis__\": \"{b}\".\n"
                    )
                else:
                    cf_step = (
                        f"     - Cash Flow: FIRST call detect_cashflow_period_basis() (no\n"
                        f"       arguments — it reads the cash-flow header for you).  Then:\n"
                        f"       * quarterly (\"Three Months Ended\") → extract as-is.\n"
                        f"       * ytd_6m / ytd_9m (\"Six/Nine Months Ended\", YEAR-TO-DATE) →\n"
                        f"         the printed flow values are CUMULATIVE, NOT the quarter.  For\n"
                        f"         EVERY flow row call deaccumulate_cashflow(value=<printed\n"
                        f"         value>, concept_key=<bracketed key>, ytd_months=6 or 9,\n"
                        f"         filing_label=<row label>) and report the RETURNED quarterly\n"
                        f"         value.  \"...ending balances\" → as-is; \"...beginning\n"
                        f"         balances\" → skip.  List every de-accumulated key in __derived__.\n"
                    )
            else:
                cf_step = f"     - Cash Flow: extract as-is.\n"
            initial_msg = (
                f"This is a {len(plain_text):,}-character earnings document "
                f"with {n_lines:,} lines across {len(state.get('document_map') or []):,} exhibit(s).\n\n"
                f"COMPANY: {state['company_name']} ({ticker})\n\n"
                f"FINANCIAL STATEMENTS ({is_exhibit_label}) — LINES {is_line_start}-{is_line_end}:\n"
                f"The financial statements text is provided below ({is_section_size:,} chars):\n\n"
                f"--- BEGIN {is_exhibit_label} ---\n"
                f"{is_exhibit_text}\n"
                f"--- END {is_exhibit_label} ---\n\n"
                f"{injected_note}\n\n"
                f"MULTI-STATEMENT EXTRACTION:\n\n"
                f"PHASE 1 — Extract from the text above:\n"
                f"  1. Call detect_scale() and detect_currency() on lines "
                f"{is_line_start}-{is_line_end} to confirm units.\n"
                f"  2. Extract ALL target statement concepts ({target_desc}) from the "
                f"tables above:\n"
                f"     - Income Statement: the \"Three Months Ended\" (quarter) column.\n"
                f"     - Balance Sheet: as-of (point-in-time) values, as-is.\n"
                + cf_step +
                f"  3. Reconcile with calculate(), verify equations:\n"
                f"     - Income Statement: verify_identity(revenue, cost_of_revenue, gross_profit)\n"
                f"     - Balance Sheet: verify_balance_sheet_identity(total_assets, total_liabilities, total_equity)\n"
                f"     - Cash Flow: verify_cash_flow_identity(operating_cf, investing_cf, financing_cf, net_change)\n\n"
                f"PHASE 2 — Search ONLY for data NOT in the text above:\n"
                f"  4. Use search()/read_lines() only for concepts NOT present in the "
                f"tables above — segment/geographic/product-line revenue breakdowns, "
                f"and any statement table that is missing from the text above.\n"
                f"  5. Extract any remaining concepts you find.\n\n"
                f"  6. Call finalize_extraction().\n\n"
                f"DO NOT call get_company_info() — you already have the company name above.\n\n"
                f"MEMORY: record remember_* facts in the SAME step you observe them "
                f"(interleaved with extraction), then call finalize_extraction() "
                f"right after your last extraction — do NOT add a separate "
                f"memory-writing step at the end, and do NOT re-read or re-search "
                f"the exhibit text for memory."
            )
        else:
            initial_msg = (
                f"This is a {len(plain_text):,}-character earnings document "
                f"with {n_lines:,} lines across {len(state.get('document_map') or []):,} exhibit(s).\n\n"
                f"COMPANY: {state['company_name']} ({ticker})\n\n"
                f"INCOME STATEMENT ({is_exhibit_label}) — LINES {is_line_start}-{is_line_end}:\n"
                f"The income statement text is provided below ({is_section_size:,} chars).\n\n"
                f"--- BEGIN {is_exhibit_label} ---\n"
                f"{is_exhibit_text}\n"
                f"--- END {is_exhibit_label} ---\n\n"
                f"TWO-PHASE EXTRACTION:\n\n"
                f"PHASE 1 — Extract from the text above (NO search/read_lines needed):\n"
                f"  1. Call detect_scale() and detect_currency() on lines "
                f"{is_line_start}-{is_line_end} to confirm units.\n"
                f"  2. Extract ALL income-statement concepts (Revenue, Operating Income, "
                f"Net Income, EPS, etc.) directly from the text above.\n"
                f"  3. Reconcile with calculate(), verify with verify_identity().\n\n"
                f"PHASE 2 — Search for dimensional/segment data NOT in the text above:\n"
                f"  4. After extracting all IS concepts, use search() to find segment "
                f"breakdowns, geographic revenue, product-line data, or other "
                f"dimensional concepts that may be in OTHER exhibits or sections.\n"
                f"  5. Extract any additional dimensional concepts you find.\n\n"
                f"  6. Call finalize_extraction().\n\n"
                f"DO NOT call get_document_info() or get_company_info() — you already "
                f"have the document structure and company name above.\n\n"
                f"MEMORY: record remember_* facts in the SAME step you observe them "
                f"(interleaved with extraction), then call finalize_extraction() "
                f"right after your last extraction — do NOT add a separate "
                f"memory-writing step at the end, and do NOT re-read or re-search "
                f"the exhibit text for memory."
            )
    elif prebuilt_sections and prebuilt_sections.get("sections"):
        map_text = format_section_index(prebuilt_sections)
        initial_msg = (
            f"This is a {len(plain_text):,}-character earnings document "
            f"with {n_lines:,} lines.\n\n"
            f"DOCUMENT SECTION MAP (1-based line ranges):\n{map_text}\n\n"
            f"Start by reading the income_statement range in ONE read_lines() "
            f"call, then call detect_scale() and detect_currency() on that same "
            f"range.  Extract every concept row from the CURRENT period column, "
            f"reconcile with calculate(), verify with verify_identity(), then "
            f"call finalize_extraction.  Use search() for any concept whose "
            f"section is missing from the map."
        )
    elif is_hints:
        best = is_hints[0]
        map_text = (
            f"EXHIBIT ROUTING HINT (from prescan — the financial statements are "
            f"likely in {best['exhibit']}, lines {best['lines'][0]}-{best['lines'][1]}\n"
            f"IS {best['primary_count']}+{best['secondary_count']} ctx, "
            f"BS {best['bs_count']}, CF {best['cf_count']})."
        )
        initial_msg = (
            f"This is a {len(plain_text):,}-character earnings document "
            f"with {n_lines:,} lines.\n\n"
            f"{map_text}\n\n"
            f"Start by calling read_lines({best['lines'][0]}, {best['lines'][1]}) "
            f"to read the full financial-statements exhibit in ONE call, then call "
            f"detect_scale() and detect_currency() on that same range.  Extract "
            f"every concept row from the CURRENT period column, reconcile with "
            f"calculate(), verify with verify_identity(), then call "
            f"finalize_extraction.  If other exhibits contain additional segment "
            f"or statement data, use search() to locate it."
        )
    else:
        initial_msg = (
            f"This is a {len(plain_text):,}-character earnings document "
            f"with {n_lines:,} lines.\n\n"
            f"If the whole document fits in ONE read_lines() call (read_lines "
            f"returns up to ~60K chars), read the ENTIRE document with "
            f"read_lines(1, {n_lines}) — it contains the period header, the "
            f"income statement, and the segment data, so you can extract "
            f"everything in one pass.  Otherwise, search(\"Revenue\") or "
            f"search(\"Net income\") to locate the income statement, then "
            f"read_lines() the section.  Call detect_scale() and "
            f"detect_currency() on each monetary table you read.  Verify before "
            f"finalizing."
        )

    if guidance_block:
        from earnings_agents.agent.prompts import GUIDANCE_PHASE_NOTICE
        initial_msg += GUIDANCE_PHASE_NOTICE

    if calc_block:
        initial_msg += (
            "\n\nCOMPUTE-ONLY concepts (marked 'not printed in the filing' in the "
            "concept list) — compute each with compute() after you finish extracting "
            "printed rows, and list every computed key in __derived__."
        )

    if DEACCUMULATE_YTD_CASHFLOW and "cashflow" in target_statements:
        from earnings_agents.agent.deaccumulate import detect_cashflow_statement_basis
        prescan_cf = detect_cashflow_statement_basis(plain_text)
        if prescan_cf and prescan_cf.get("basis") in ("ytd_6m", "ytd_9m") and getattr(period, "quarter", None) in (2, 3):
            initial_msg += (
                f"\n\n⚠️ CASH-FLOW DIRECTIVE: The Cash Flow statement is {prescan_cf['months']}-month "
                f"YEAR-TO-DATE (\"{prescan_cf['evidence']}\").  Printed flow values are cumulative "
                f"for {prescan_cf['months']} months, NOT the quarter.  You MUST call deaccumulate_cashflow() for every "
                f"flow row and list every computed key in __derived__.  Report \"__cashflow_basis__\": "
                f"\"{prescan_cf['basis']}\" in finalize_extraction().  Any raw cumulative "
                f"flow rows not de-accumulated will be omitted from the quarterly save."
            )

    final_result = run_agent_loop(
        system_prompt=system_prompt,
        initial_message=initial_msg,
        tools=tools,
        ticker=ticker,
        dollar_multiplier=dollar_multiplier,
        # Per-share/percentage/share-count concepts must never be scaled even
        # when their member-tagged keys carry no "per share" text (the as-is
        # signal lives in the concept LABEL — see build_no_scale_keys).
        no_scale_keys=build_no_scale_keys(target_concepts),
    )

    if final_result is None:
        return {
            **state,
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
            "status": "failed",
            "error": (
                f"Document does not match ticker {ticker} — extraction "
                f"aborted (company mismatch)"
            ),
        }

    # ── 5. Merge, validate, map ─────────────────────────────────────────
    # The agent computes derived metrics itself in-loop via calculate()/
    # compute() and returns them in the same finalize JSON; the deterministic
    # derive pass below remains as a safety net for CALC concepts the agent
    # did not compute.  Concepts the agent computed are marked derived.
    metrics: dict[str, Any] = final_result

    # Log extracted values for debugging
    extracted_keys = [k for k in metrics if not k.startswith("__")]
    logger.info("Agent extracted %d keys for %s:", len(extracted_keys), ticker)
    for k in extracted_keys:
        v = metrics[k]
        if isinstance(v, (int, float)):
            logger.info("  %s = %s", k, f"{v:,.0f}")
        else:
            logger.info("  %s = %s (non-numeric)", k, str(v)[:80])

    # Pop out the agent's own report fields (never mapped/scaled as metrics).
    # __derived__ lists the bracketed keys the agent COMPUTED (via compute()/
    # calculate()) rather than read verbatim — used to mark derived concepts.
    derived_raw = metrics.pop("__derived__", None)
    derived_keys: set[str] = set()
    if isinstance(derived_raw, str):
        derived_keys = {k.strip().strip("[]") for k in derived_raw.split(",") if k.strip()}
    elif isinstance(derived_raw, list):
        derived_keys = {str(k).strip().strip("[]") for k in derived_raw if str(k).strip()}
    agent_currency = str(metrics.pop("__currency__", "") or "").strip().upper()
    agent_currency_evidence = metrics.pop("__currency_evidence__", "") or ""
    # Per-value evidence: {"[us-gaap:Revenues]": {"lines": [s, e], "scale":
    # "millions", "currency": "USD", ...}} — preserved by the parser.
    agent_evidence = metrics.pop("__evidence__", None)
    if not isinstance(agent_evidence, dict):
        agent_evidence = {}
    agent_missing = metrics.pop("__missing__", None)
    if isinstance(agent_missing, str):
        agent_missing = [x.strip() for x in agent_missing.split(",") if x.strip()]
    if not isinstance(agent_missing, list):
        agent_missing = []
    # The cash-flow statement's column basis the agent observed via
    # detect_cashflow_period_basis(): "quarterly" | "ytd_6m" | "ytd_9m" |
    # "annual" (or None when the agent did not report it).
    agent_cashflow_basis = metrics.pop("__cashflow_basis__", None)
    if isinstance(agent_cashflow_basis, str):
        agent_cashflow_basis = agent_cashflow_basis.strip().lower()
    else:
        agent_cashflow_basis = None

    # Currency authority = the tool-calling agent's report.  A deterministic
    # whole-document scan is only a fallback for confirmed foreign/mixed codes
    # when the agent did not report anything; it never blocks "unknown".
    scan = usd_metadata(plain_text)
    if agent_currency:
        currency_meta = {
            "currency": agent_currency,
            "confidence": "agent",
            "detected_codes": [agent_currency],
            "evidence": agent_currency_evidence,
            "requires_review": not is_usd_safe(agent_currency),
        }
    elif scan.get("currency") not in (None, "USD"):
        currency_meta = scan  # confirmed foreign/mixed — will block at save
    else:
        currency_meta = {
            "currency": "unknown",
            "confidence": "unconfirmed",
            "detected_codes": scan.get("detected_codes") or [],
            "evidence": [],
            "requires_review": True,
        }

    # ── 4c. Forward-looking guidance (PHASE 3 output) ─────────────────────
    # The extraction agent reports guidance numbers in finalize_extraction
    # under the metadata key "__guidance__" — a JSON list, one object per
    # guidance number.  Normalize each to the `guidance_values` schema (same
    # collection the admin backend reads/writes).  Absence is NORMAL (many
    # 8-Ks disclose no quantitative guidance) — never a failure, never a
    # blocking finding; dropped records are observability only.
    guidance_records: list[dict] = []
    guidance_issues: list[str] = []
    from earnings_agents.agent.guidance import normalize_guidance_records
    agent_guidance = metrics.pop("__guidance__", None)
    if agent_guidance:
        try:
            guidance_records, guidance_issues = normalize_guidance_records(
                agent_guidance, period, currency_meta.get("currency"),
            )
            if guidance_issues:
                logger.info(
                    "guidance for %s: %d record(s) dropped — %s",
                    ticker, len(guidance_issues),
                    " ".join(guidance_issues[:30]),
                )
        except Exception as exc:  # noqa: BLE001 — guidance must never kill the run
            logger.warning("guidance normalization failed for %s: %s", ticker, exc)
            guidance_records = []
            guidance_issues = [f"normalization error: {exc}"]
    if guidance_records:
        report_call(
            f"  [guidance]  {len(guidance_records)} guidance record(s) extracted "
            f"for {ticker}"
        )
    elif GUIDANCE_ENABLED and agent_guidance:
        # ALWAYS visible: the agent reported guidance but nothing usable
        # survived normalization.  Print the drop reasons — the CLI runs at
        # WARNING level, so logger.info is invisible to operators.
        n_reported = (
            len(agent_guidance)
            if isinstance(agent_guidance, (list, dict))
            else "?"
        )
        brief = ("; ".join(guidance_issues[:5])) or "unknown reason"
        report_call(
            f"  [guidance]  ⚠ {len(guidance_issues)} of {n_reported} reported "
            f"record(s) dropped: {brief}"
        )
    elif GUIDANCE_ENABLED:
        report_call(f"  [guidance]  no guidance reported by agent for {ticker}")

    concept_metrics, _reverse_map, mapped_keys = map_concepts(metrics, target_concepts)

    # Exact taxonomy/label mapping is preferred.  Resolve only leftover
    # numeric keys semantically (without changing their values) so filing
    # wording such as "Revenue" vs "Total revenue" does not silently vanish.
    concept_metrics, semantic_mapped_keys, semantic_reverse = semantically_map_unmapped_metrics(
        metrics, target_concepts, concept_metrics,
    )
    mapped_keys.update(semantic_mapped_keys)
    _reverse_map.update(semantic_reverse)

    # ── Derived (computed) concepts ────────────────────────────────────
    # CALC (system:/calculated) concepts are never verbatim in the filing, so
    # their presence in the mapped results means the agent computed them via
    # compute().  Any other concept the agent listed in __derived__ (e.g. a
    # multi-component sum) is also marked derived.
    concept_by_id = {c["_id"]: c for c in target_concepts}
    derived_ids: set[str] = set()
    for cid in concept_metrics:
        c = concept_by_id.get(cid, {})
        if str(c.get("concept") or c.get("taxonomy_key") or "").lower().startswith("system:") or c.get("calculated"):
            derived_ids.add(cid)
            continue
        mk = (_reverse_map.get(cid) or "").strip().strip("[]")
        mk_base = mk.split("|")[0]
        derived_bases = {dk.split("|")[0] for dk in derived_keys}
        concept_name = (c.get("concept") or "").strip().lower()
        if mk and (
            mk in derived_keys
            or mk_base in derived_keys
            or mk_base.lower() in derived_bases
            or concept_name in derived_bases
        ):
            derived_ids.add(cid)

    # ── Non-blocking cash-flow de-accumulation check ───────────────────
    # The cash-flow statement's basis is checked against both the agent's report
    # and deterministic document scan as ground truth: the agent may omit
    # __cashflow_basis__ or mistakenly claim "quarterly" despite the table header
    # saying "Six/Nine Months Ended".  When the filing or the agent declares YTD
    # for a Q2/Q3 period, any cash-flow flow concept that was not de-accumulated
    # is omitted from concept_metrics so raw YTD values are not saved as
    # quarterly actuals, and a medium-severity finding is recorded.
    # Non-deaccumulated concepts NEVER block the save — all other valid
    # metrics (income statement, balance sheet, de-accumulated flows) are saved.
    from earnings_agents.agent.deaccumulate import (
        classify_cashflow_kind,
        detect_cashflow_statement_basis,
    )
    doc_cf = detect_cashflow_statement_basis(plain_text)
    doc_basis = doc_cf["basis"] if doc_cf else None

    effective_cf_basis = agent_cashflow_basis
    basis_source = "agent"
    if effective_cf_basis not in ("ytd_6m", "ytd_9m") and doc_basis in ("ytd_6m", "ytd_9m"):
        effective_cf_basis = doc_basis
        basis_source = "document"

    cf_deaccum_findings: list[dict] = []
    if (
        effective_cf_basis in ("ytd_6m", "ytd_9m")
        and period.quarter in (2, 3)
    ):
        not_deaccumulated: list[str] = []
        not_deaccumulated_cids: list[str] = []
        for cid in list(concept_metrics.keys()):
            c = concept_by_id.get(cid, {})
            if (c.get("statement_type") or "").strip().lower() != "cashflow":
                continue
            if classify_cashflow_kind(
                c.get("concept") or "", c.get("label") or "",
            ) != "flow":
                continue  # ending/beginning snapshots are as-is / skip
            if cid not in derived_ids:
                not_deaccumulated.append(
                    c.get("label") or c.get("concept") or str(cid)
                )
                not_deaccumulated_cids.append(cid)
        if not_deaccumulated:
            # Omit un-deaccumulated flow concepts so cumulative YTD values are
            # not persisted as quarterly actuals, while saving all valid metrics.
            for cid in not_deaccumulated_cids:
                concept_metrics.pop(cid, None)
            reason = (
                f"Cash-flow statement is {effective_cf_basis} (year-to-date, detected from {basis_source})"
            )
            if basis_source == "document" and agent_cashflow_basis:
                reason += f" although agent reported '{agent_cashflow_basis}'"
            cf_deaccum_findings.append({
                "type": "cashflow_not_deaccumulated",
                "severity": "medium",
                "message": (
                    f"{reason} — {len(not_deaccumulated)} flow concept(s) were not "
                    f"de-accumulated and omitted from save: {', '.join(not_deaccumulated[:10])}."
                ),
                "evidence": {
                    "cashflow_basis": effective_cf_basis,
                    "agent_cashflow_basis": agent_cashflow_basis,
                    "doc_cashflow_basis": doc_basis,
                    "not_deaccumulated": not_deaccumulated[:20],
                },
            })
            report_call(
                f"  [pipeline]  ⚠️ cash-flow is {effective_cf_basis} ({basis_source}) — "
                f"{len(not_deaccumulated)} flow concept(s) not de-accumulated "
                f"(omitted from quarterly save, run continues)"
            )

    # ── 6. Return state ──────────────────────────────────────────────────
    raw_text = plain_text[:EXTRACTION_MAX_CHARS]

    mapped_ids = set(concept_metrics.keys()) - derived_ids
    all_target_ids = {c["_id"] for c in target_concepts}
    missing_ids = all_target_ids - mapped_ids - derived_ids
    missing_labels = [c["label"] for c in target_concepts if c["_id"] in missing_ids]
    missing_toplevel, missing_segments = classify_missing_labels(missing_ids, target_concepts)

    # Hierarchy paths with multiple same-path parent rows — auto-derivation
    # refuses to attach children there; surface them as observability.
    ambiguity_findings: list[dict] = []
    if ambiguous_paths:
        ambiguity_findings.append({
            "type": "hierarchy_ambiguity",
            "severity": "medium",
            "message": (
                f"{len(ambiguous_paths)} hierarchy path(s) have multiple same-path "
                "parent rows — children were not auto-derived for those paths"
            ),
            "evidence": {"paths": sorted(ambiguous_paths)[:20]},
        })

    # Per-concept metadata for persistence (dimension identity, currency,
    # scale/evidence, calculated status) + structured findings for the gate.
    currency_value = currency_meta.get("currency") or "unknown"
    # Resolve evidence keyed by the agent's metric key (bracketed taxonomy key
    # or label) onto concept ids via the mapping reverse-map.
    evidence_by_concept: dict[str, dict] = {}
    for cid, metric_key in _reverse_map.items():
        ev = agent_evidence.get(metric_key)
        if isinstance(ev, dict):
            evidence_by_concept[cid] = ev
    value_metadata_by_id: dict[str, dict] = {}
    for cid in concept_metrics:
        c = concept_by_id.get(cid, {})
        ev = evidence_by_concept.get(cid, {})
        # Per-value currency: agent-declared per-value evidence wins; otherwise
        # the document-level currency decision applies.
        ev_currency = str(ev.get("currency") or "").strip().upper()
        value_currency = ev_currency or currency_value
        value_metadata_by_id[cid] = {
            "statement_type": c.get("statement_type") or "income",
            "dimension": bool(c.get("dimension")),
            "dimension_concept": bool(c.get("dimension_concept")),
            "dimension_member": c.get("dimension_member") or "",
            "dimension_member_label": c.get("dimension_member_label") or "",
            "dimension_axis": c.get("dimension_axis") or "",
            "calculated": cid in derived_ids or bool(c.get("calculated")),
            "currency": "USD" if is_usd_safe(value_currency) else value_currency,
            "status": "derived" if cid in derived_ids else "mapped",
        }
        if isinstance(ev.get("lines"), list) and len(ev["lines"]) == 2:
            value_metadata_by_id[cid]["source_lines"] = [
                int(ev["lines"][0]), int(ev["lines"][1]),
            ]
        if ev.get("scale"):
            value_metadata_by_id[cid]["scale"] = str(ev["scale"]).strip().lower()
        if ev.get("evidence"):
            value_metadata_by_id[cid]["source_snippet"] = str(ev["evidence"])[:500]
        if not is_usd_safe(value_currency) and value_currency not in ("", "unknown"):
            value_metadata_by_id[cid]["requires_review"] = True
    findings = _build_completeness_findings(
        currency_meta, state.get("document_map"), agent_missing
    )
    findings.extend(ambiguity_findings)
    findings.extend(cf_deaccum_findings)

    # Deterministic company-identity cross-check: the agent reports the
    # company name it actually read (__company_name__); compare (normalized)
    # against the expected name.  A mismatch is high severity — it can only be
    # a wrong-document upload (manual URL/PDF path) or an extraction mistake.
    reported_name = str(metrics.pop("__company_name__", "") or "").strip()
    expected_name = str(state.get("company_name") or "").strip()
    if reported_name and expected_name and not check_company_identity(
        expected_name, reported_name
    ):
        report_call(
            f"  [pipeline]  ✗ document company {reported_name!r} does not "
            f"match expected {expected_name!r} — company mismatch"
        )
        logger.warning(
            "Company-name cross-check failed for %s: document says %r, "
            "expected %r", ticker, reported_name, expected_name,
        )
        return {
            **state,
            "status": "failed",
            "error": (
                f"Document company name {reported_name!r} does not match "
                f"{expected_name!r} — company mismatch"
            ),
        }

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
        "guidance_records": guidance_records,
        "derived_concept_ids": list(derived_ids),
        "mapped_metric_keys": list(mapped_keys),
        "missing_concept_labels": missing_labels,
        "missing_toplevel_labels": missing_toplevel,
        "missing_segment_labels": missing_segments,
        "memory_block": memory_block,
        "currency_metadata": currency_meta,
        "value_metadata_by_id": value_metadata_by_id,
        "ambiguous_paths": sorted(ambiguous_paths),
        "findings": findings,
        "status": "extracted",
    }


def agent_document_pipeline_node(state: EarningsAgentState) -> EarningsAgentState:
    """Agent pipeline: a single extraction pass over plain text.

    One extraction-agent pass reads the document with navigation tools and
    post-processes the result (map, derive, company-identity cross-check,
    findings for the save gate).  There is deliberately no verifier agent and
    no retry loop — metrics that are not present in the filing are recorded
    as observability (``missing_concept``) and never re-extracted.
    """
    from earnings_agents.config import LLM_PROVIDER as _LLM_PROVIDER
    from earnings_agents.hooks import report_call

    ticker = state["ticker"]
    target_concepts: list[dict] = state.get("target_concepts") or []  # type: ignore[assignment]

    if not target_concepts:
        return {**state, "status": "failed", "error": f"No target concepts for {ticker}"}

    plain_text = state.get("raw_text") or ""
    if not plain_text:
        return {**state, "status": "failed", "error": "No document text in state (fetch_filing missing?)"}

    report_call(f"  [pipeline]  🧠 AGENT extraction  ({_LLM_PROVIDER or 'llm'})")

    return _run_extraction_pass(state, plain_text, target_concepts)
