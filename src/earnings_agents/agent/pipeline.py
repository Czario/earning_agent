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
    build_no_scale_keys,
    load_prior_values,
    map_concepts,
    semantically_map_unmapped_metrics,
    prescan_document,
    SCALE_MULTIPLIERS,
)
from earnings_agents.agent.currency import usd_metadata, is_usd_safe
from earnings_agents.agent.prompts import (
    PIPELINE_SYSTEM_PROMPT,
    COMPANY_IDENTITY_RULE,
    build_concept_list,
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


def _auditable_missing_labels(
    missing_labels: list[str] | None,
    target_concepts: list[dict],
) -> list[str]:
    """Drop derivation-target labels from the missing list the verifier audits.

    ``calculated``/``system:`` concepts are handled by the derive pass, not by
    extraction — the agent can never emit them.  Showing them to the verifier
    as "missing" makes it flag them every round (Gross Profit churn), so they
    are excluded from the audit.
    """
    calc_labels = {
        (c.get("label") or "").strip() for c in target_concepts
        if c.get("calculated")
        or str(c.get("concept") or c.get("taxonomy_key") or "").lower().startswith("system:")
    }
    return [lbl for lbl in (missing_labels or []) if lbl not in calc_labels]


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


def _build_verifier_retry_briefing(
    issues: list[dict],
    missing_labels: list[str] | None = None,
) -> str:
    """Build a targeted retry briefing from verifier issues.

    The extraction agent's next pass fixes ONLY the flagged items; everything
    else is carried forward by the pipeline (metrics merge across passes).
    """
    sections: list[str] = []
    lines: list[str] = []

    for issue in issues:
        itype = issue.get("type", "")
        concept = issue.get("concept") or "?"
        msg = issue.get("message") or ""
        loc = ""
        if isinstance(issue.get("lines"), list) and len(issue["lines"]) == 2:
            loc = f" (lines {issue['lines'][0]}-{issue['lines'][1]})"
        if itype == "missing_row":
            lines.append(f"  • MISSING — extract: {concept}{loc} — {msg}")
        elif itype == "value_mismatch":
            lines.append(
                f"  • WRONG VALUE — re-read and correct: {concept}{loc} — {msg}"
            )
        elif itype == "wrong_scale":
            lines.append(
                f"  • WRONG SCALE — re-read the scale declaration and correct: "
                f"{concept}{loc} — {msg}"
            )
        elif itype == "wrong_currency":
            lines.append(
                f"  • WRONG CURRENCY — confirm with detect_currency(): "
                f"{concept}{loc} — {msg}"
            )
        elif itype == "wrong_parent":
            lines.append(
                f"  • WRONG PARENT/SEGMENT — re-check the row's position: "
                f"{concept}{loc} — {msg}"
            )
        elif itype == "needs_confirmation":
            lines.append(
                f"  • CONFIRM — verify against the filing: {concept}{loc} — {msg}"
            )

    if lines:
        sections.append("VERIFIER ISSUES — fix these ONLY:\n" + "\n".join(lines))

    if missing_labels:
        sections.append(
            "CONCEPTS STILL UNLOCATED (search the filing again; if genuinely "
            "absent, report them in __missing__):\n"
            + "\n".join(f"  • {lbl}" for lbl in missing_labels[:50])
        )

    return "\n\n".join(sections)


def _resolve_retry_concept_ids(
    issues: list[dict],
    missing_labels: list[str] | None,
    target_concepts: list[dict],
) -> set[str]:
    """Map verifier-flagged concepts + still-missing labels to concept ids.

    The retry pass is scoped to exactly these concepts — the agent searches
    and re-extracts ONLY the ones that were flagged/missing, never the full
    target list.  Everything else is carried forward from the previous pass.

    ``calculated``/``system:`` concepts (derivation targets) are EXCLUDED:
    the extraction agent never emits them (they are filtered out of its
    concept list), so retrying them is pure churn — only the derive pass
    handles them.
    """
    wanted: set[str] = set()
    for issue in issues:
        c = (issue.get("concept") or "").strip().lower()
        if c:
            wanted.add(c)
    for lbl in missing_labels or []:
        lbl = (lbl or "").strip().lower()
        if lbl:
            wanted.add(lbl)
    if not wanted:
        return set()
    ids: set[str] = set()
    for c in target_concepts:
        if c.get("calculated") or str(
            c.get("concept") or c.get("taxonomy_key") or ""
        ).lower().startswith("system:"):
            continue  # derivation target — not extractable by the agent
        key = (c.get("taxonomy_key") or c.get("concept") or "").strip()
        label = (c.get("label") or "").strip()
        if (
            f"[{key}]".lower() in wanted
            or key.lower() in wanted
            or label.lower() in wanted
        ):
            ids.add(c["_id"])
    return ids


# Issue types that can NEVER block the save, no matter what severity the
# verifier LLM assigns.  Absence (missing_row) and uncertainty
# (needs_confirmation) cannot corrupt the values being stored — the
# documented invariant is that only confirmed integrity issues (wrong
# value/scale/currency, wrong segment parent, identity) block.  A phantom
# concept flagged as high-severity missing_row must not drop a full period
# of correct values (observed live: PDD blocked by a polluted `404` row).
_NON_BLOCKING_ISSUE_TYPES = {"missing_row", "needs_confirmation"}


def _issue_signatures(report: dict | None) -> frozenset:
    """Fingerprint a verifier report's actionable issues.

    Used to detect a non-converging retry loop: when the audit after a
    targeted retry reports the SAME (type, concept, values) issues as the
    previous round, the retry changed nothing and further retries cannot
    converge — the caller stops early instead of burning the remaining
    extraction passes.
    """
    sigs: set[tuple] = set()
    for issue in (report or {}).get("issues") or []:
        if issue.get("type") == "absent_ok":
            continue
        sigs.add((
            issue.get("type"),
            str(issue.get("concept") or "").strip().lower(),
            issue.get("reported_value"),
            issue.get("found_value"),
        ))
    return frozenset(sigs)


def _findings_from_verifier(
    report: dict | None,
    target_concepts: list[dict] | None = None,
    missing_labels: list[str] | None = None,
) -> tuple[list[dict], str, set[str]]:
    """Convert a verifier report into gate findings + retry briefing.

    High-severity actionable issues become blocking findings; ``absent_ok``
    and low/medium confirmations are observability only.  Returns
    ``(findings, retry_briefing, retry_concept_ids)`` — *retry_concept_ids*
    scopes the next extraction pass to ONLY the flagged/missing concepts.
    """
    if report is None:
        return [{
            "type": "verification_unavailable",
            "severity": "medium",
            "message": "Verifier produced no report — extraction not independently confirmed",
        }], "", set()

    issues = report.get("issues") or []
    findings: list[dict] = []
    briefing_items: list[dict] = []
    high_count = 0
    for issue in issues:
        itype = issue.get("type", "")
        severity = issue.get("severity", "medium")
        concept = issue.get("concept") or "?"
        message = issue.get("message") or itype
        if itype == "absent_ok":
            findings.append({
                "type": "absent_ok",
                "severity": "low",
                "message": f"Verifier confirmed absent: {concept}",
            })
            continue
        # Absence/uncertainty never blocks (documented invariant) — cap the
        # LLM-assigned severity; missing_row still drives a targeted retry.
        if itype in _NON_BLOCKING_ISSUE_TYPES and severity == "high":
            severity = "medium"
        if severity == "high":
            high_count += 1
            findings.append({
                "type": f"verifier_{itype}",
                "severity": "high",
                "message": f"Verifier: {concept} — {message}",
                "evidence": {"lines": issue.get("lines")},
            })
            briefing_items.append(issue)
        else:
            findings.append({
                "type": f"verifier_{itype}",
                "severity": severity,
                "message": f"Verifier: {concept} — {message}",
                "evidence": {"lines": issue.get("lines")},
            })
            # Only CONFIRMED issues drive a retry round.  needs_confirmation is
            # verifier uncertainty — recorded as a non-blocking finding, but it
            # must not burn another extraction pass on a maybe.
            if itype in ("missing_row", "value_mismatch"):
                briefing_items.append(issue)

    briefing = _build_verifier_retry_briefing(briefing_items, missing_labels)
    retry_ids = _resolve_retry_concept_ids(
        briefing_items, missing_labels, target_concepts or []
    )
    return findings, briefing, retry_ids


def _run_extraction_pass(
    state: EarningsAgentState,
    plain_text: str,
    target_concepts: list[dict],
    attempt_num: int,
    retry_briefing: str = "",
    retry_concept_ids: set[str] | None = None,
) -> EarningsAgentState:
    """Run ONE extraction-agent pass and post-process its output.

    Returns the updated state with ``status="extracted"`` on success, or a
    failed/skipped state.  ``metrics``/``concept_metrics``/``value_metadata_by_id``
    are set (or merged with prior state on retry passes) so the caller can
    verify and re-run.

    On retry passes (*retry_concept_ids*), the pass is SCOPED to the
    flagged/missing concepts only: the concept list, tools, and scale
    guardrails all see just that subset, so the agent searches and
    re-extracts ONLY the missing metrics — never the full target list.  The
    agent's output is merged over the previous pass (no deterministic value
    filtering; the scoped concept list is what keeps the retry targeted).
    """
    from earnings_agents.hooks import report_call

    ticker = state["ticker"]

    # ── 1. Document pre-scan (scale only) — deterministic mechanical step ──
    # Currency is NOT decided here: the tool-calling agent inspects each
    # table/section with detect_currency() and reports __currency__ itself.
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

    # ── 2b. Scope the pass (retry = flagged/missing concepts only) ────────
    # A targeted retry never re-runs the full extraction: the agent sees ONLY
    # the concepts the verifier flagged plus the ones the previous pass could
    # not locate.  Everything else is carried forward by the merge below.
    if attempt_num > 1 and retry_concept_ids:
        pass_concepts = [
            c for c in target_concepts if c["_id"] in retry_concept_ids
        ]
        if not pass_concepts:
            pass_concepts = target_concepts  # defensive fallback
    else:
        pass_concepts = target_concepts

    # ── 3. Build prompt ──────────────────────────────────────────────────
    concept_list_str = build_concept_list(
        pass_concepts,
        recent_concept_ids=set(state.get("recent_concept_ids") or []),
        calculated_concepts=state.get("calculated_concepts"),
    )

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
            f"SCOPE: Your concept list below contains ONLY the metrics to "
            f"retry.  Search for and re-extract JUST those — nothing else.  "
            f"All other metrics from the previous pass are already correct "
            f"and will be carried forward automatically.  Return ONLY the "
            f"metrics you retried in finalize_extraction; do NOT re-extract "
            f"or re-read the full document."
        )
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
        target_concepts=pass_concepts,
    )

    initial_msg: str
    if attempt_num > 1 and retry_briefing:
        initial_msg = (
            "TARGETED RETRY — search ONLY the missing/flagged metrics listed "
            "in the system prompt; your concept list is scoped to them. "
            "Use search() to locate each one, read_lines() to read only that "
            "section, calculate()/compute() for derived values, "
            "map_concept() to confirm the right concept key, "
            "verify_identity() to check. "
            "Return ONLY the metrics you retried — the rest are carried "
            "forward automatically. Do NOT re-extract the whole document."
        )
    else:
        initial_msg = (
            f"This is a {len(plain_text):,}-character earnings document "
            f"with {n_lines:,} lines.  Start by searching for the income statement: "
            f'search("Revenue") or search("Net income") to locate it, then '
            f"read_lines() to extract metrics.  Call detect_scale() and "
            f"detect_currency() on each monetary table you read.  Verify before "
            f"finalizing."
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
        no_scale_keys=build_no_scale_keys(pass_concepts),
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
    # The agent computes derived metrics itself in-loop via calculate()/
    # compute() and returns them in the same finalize JSON; the deterministic
    # derive pass below remains as a safety net for CALC concepts the agent
    # did not compute.  Concepts the agent computed are marked derived.
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

    # Pop out the derived-marker and the agent's own report fields (never
    # mapped/scaled as metrics).  The agent is the authority on currency and on
    # which concepts it searched for but could not locate.
    metrics.pop("__derived__", None)
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
    concept_metrics, derived_ids, ambiguous_paths = derive_missing_concepts(
        concept_metrics, target_concepts,
    )

    # ── 6. Return state ──────────────────────────────────────────────────
    raw_text = plain_text[:EXTRACTION_MAX_CHARS]

    mapped_ids = set(concept_metrics.keys()) - derived_ids
    all_target_ids = {c["_id"] for c in target_concepts}
    missing_ids = all_target_ids - mapped_ids - derived_ids
    missing_labels = [c["label"] for c in target_concepts if c["_id"] in missing_ids]
    missing_toplevel, missing_segments = classify_missing_labels(missing_ids, target_concepts)

    # Hierarchy paths with multiple same-path parent rows — the derivation
    # pass refuses to attach children there; surface them so the verifier
    # confirms child→parent attribution against the filing.
    ambiguity_findings: list[dict] = []
    if ambiguous_paths:
        ambiguity_findings.append({
            "type": "hierarchy_ambiguity",
            "severity": "medium",
            "message": (
                f"{len(ambiguous_paths)} hierarchy path(s) have multiple same-path "
                "parent rows — children were not auto-derived for those paths; "
                "verifier should confirm attribution"
            ),
            "evidence": {"paths": sorted(ambiguous_paths)[:20]},
        })

    # Per-concept metadata for persistence (dimension identity, currency,
    # scale/evidence, calculated status) + structured findings for the gate.
    concept_by_id = {c["_id"]: c for c in target_concepts}
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
            "extraction_attempts": attempt_num,
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
        "derived_concept_ids": list(derived_ids),
        "mapped_metric_keys": list(mapped_keys),
        "missing_concept_labels": missing_labels,
        "missing_toplevel_labels": missing_toplevel,
        "missing_segment_labels": missing_segments,
        "currency_metadata": currency_meta,
        "value_metadata_by_id": value_metadata_by_id,
        "ambiguous_paths": sorted(ambiguous_paths),
        "findings": findings,
        "extraction_attempts": attempt_num,
        "status": "extracted",
    }


def agent_document_pipeline_node(state: EarningsAgentState) -> EarningsAgentState:
    """Agent pipeline: plain text + navigation tools, with verifier loop.

    Extract → verify (independent second-read agent) → targeted retry, bounded
    by ``MAX_EXTRACTION_ATTEMPTS`` / ``VERIFIER_MAX_ROUNDS``.  Verifier issues
    that survive the loop become blocking high-severity findings for the save
    gate; values are never edited by the verifier.
    """
    from earnings_agents.config import LLM_PROVIDER as _LLM_PROVIDER
    from earnings_agents.config import MAX_EXTRACTION_ATTEMPTS, VERIFIER_MAX_ROUNDS, VERIFIER_MAX_STEPS
    from earnings_agents.hooks import report_call
    from earnings_agents.agent.verify import run_verifier

    ticker = state["ticker"]
    target_concepts: list[dict] = state.get("target_concepts") or []  # type: ignore[assignment]

    if not target_concepts:
        return {**state, "status": "failed", "error": f"No target concepts for {ticker}"}

    plain_text = state.get("raw_text") or ""
    if not plain_text:
        return {**state, "status": "failed", "error": "No document text in state (fetch_filing missing?)"}

    report_call(f"  [pipeline]  🧠 AGENT pipeline + verifier  ({_LLM_PROVIDER or 'llm'})")

    max_rounds = max(1, min(VERIFIER_MAX_ROUNDS, MAX_EXTRACTION_ATTEMPTS))

    # ── Pass 1: initial extraction ────────────────────────────────────────
    pass_state = _run_extraction_pass(state, plain_text, target_concepts, 1)
    if pass_state.get("status") != "extracted":
        return pass_state

    round_num = 1
    prev_issue_sigs: frozenset | None = None
    while round_num <= max_rounds:
        report_call(f"  [verify]  round {round_num}/{max_rounds} — independent audit")
        # Derivation targets (calculated/system:) are not extractable — never
        # hand them to the verifier as "missing" (prevents GP-style churn).
        auditable_missing = _auditable_missing_labels(
            pass_state.get("missing_concept_labels") or [], target_concepts,
        )
        try:
            report = run_verifier(
                plain_text,
                ticker=ticker,
                target_concepts=target_concepts,
                concept_metrics=pass_state.get("concept_metrics") or {},
                value_metadata_by_id=pass_state.get("value_metadata_by_id") or {},
                cik=pass_state.get("cik"),
                company_name=pass_state.get("company_name") or ticker,
                company_industry=pass_state.get("company_industry"),
                document_map=pass_state.get("document_map"),
                missing_labels=auditable_missing,
                ambiguous_paths=set(pass_state.get("ambiguous_paths") or []),
                max_steps=VERIFIER_MAX_STEPS,
            )
        except Exception as exc:  # noqa: BLE001 — verifier must never crash the run
            logger.warning("Verifier for %s failed: %s", ticker, exc)
            report = None

        verifier_findings, retry_briefing, retry_concept_ids = _findings_from_verifier(
            report,
            target_concepts=target_concepts,
            missing_labels=auditable_missing,
        )
        if not retry_briefing:
            # Verified (or only non-actionable items) — merge observability
            # findings and finish.
            base_findings = list(pass_state.get("findings") or [])
            merged = base_findings + [
                f for f in verifier_findings
                if f.get("severity") != "low"
                or f.get("type") == "verification_unavailable"
            ]
            return {
                **pass_state,
                "findings": merged,
                "verifier_report": report,
                "status": "extracted",
            }

        # Non-convergence guard: if this audit reports EXACTLY the issues
        # the previous round already flagged (same type/concept/values), the
        # last targeted retry changed nothing, so another retry cannot
        # converge either — stop early with this report as the final audit
        # instead of burning the remaining passes (observed live: the same
        # 5 issues looping through 3 extraction passes on PDD).
        sigs = _issue_signatures(report)
        if prev_issue_sigs is not None and sigs == prev_issue_sigs:
            report_call(
                "  [verify]  ✗ identical issues after targeted retry — "
                "not converging; recording findings"
            )
            merged = list(pass_state.get("findings") or []) + [
                f for f in verifier_findings if f.get("severity") != "low"
            ]
            return {
                **pass_state,
                "findings": merged,
                "verifier_report": report,
                "status": "extracted",
            }
        prev_issue_sigs = sigs

        # Actionable issues — one targeted retry pass scoped to ONLY the
        # flagged/missing concepts (the agent never re-extracts all metrics).
        next_attempt = int(pass_state.get("extraction_attempts", 1)) + 1
        logger.info(
            "Verifier for %s: %d actionable issue(s) — targeted retry pass %d "
            "(%d concept(s) in scope)",
            ticker, len(retry_briefing.splitlines()), next_attempt,
            len(retry_concept_ids),
        )
        report_call(
            f"  [verify]  issues found — targeted retry pass {next_attempt} "
            f"({len(retry_concept_ids)} concept(s))"
        )
        retry_state = _run_extraction_pass(
            pass_state, plain_text, target_concepts, next_attempt,
            retry_briefing=retry_briefing,
            retry_concept_ids=retry_concept_ids,
        )
        if retry_state.get("status") != "extracted":
            return retry_state
        pass_state = retry_state
        round_num += 1

    # ── Rounds exhausted with issues still unresolved — block the save ────
    report = None
    auditable_missing = _auditable_missing_labels(
        pass_state.get("missing_concept_labels") or [], target_concepts,
    )
    try:
        report = run_verifier(
            plain_text,
            ticker=ticker,
            target_concepts=target_concepts,
            concept_metrics=pass_state.get("concept_metrics") or {},
            value_metadata_by_id=pass_state.get("value_metadata_by_id") or {},
            cik=pass_state.get("cik"),
            company_name=pass_state.get("company_name") or ticker,
            company_industry=pass_state.get("company_industry"),
            document_map=pass_state.get("document_map"),
            missing_labels=auditable_missing,
            ambiguous_paths=set(pass_state.get("ambiguous_paths") or []),
            max_steps=VERIFIER_MAX_STEPS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Final verifier for %s failed: %s", ticker, exc)
        report = None

    verifier_findings, _, _ = _findings_from_verifier(
        report,
        target_concepts=target_concepts,
        missing_labels=auditable_missing,
    )
    merged = list(pass_state.get("findings") or []) + [
        f for f in verifier_findings if f.get("severity") != "low"
    ]
    report_call(
        f"  [verify]  ✗ verifier rounds exhausted — unresolved issues recorded"
    )
    return {
        **pass_state,
        "findings": merged,
        "verifier_report": report,
        "status": "extracted",
    }
