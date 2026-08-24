"""Independent completeness verifier agent.

The extraction agent is the authority on *finding* values; the verifier is an
independent second-read agent that audits the extraction against the source
document.  It receives the extracted values (with evidence where available),
the target concept checklist, and the same navigation tools, then decides for
each flagged/missing item whether it is:

* ``value_mismatch``  — the number on the cited lines does not match,
* ``missing_row``     — a target concept is printed in the filing but was not
  extracted (needs a targeted retry),
* ``absent_ok``       — the concept is genuinely not in the filing,
* ``wrong_scale`` / ``wrong_currency`` — the declared table unit/currency
  conflicts with the evidence,
* ``wrong_parent``    — a segment/dimensional row is attributed to the wrong
  hierarchy parent.

The verifier never edits values; it only reports issues.  The pipeline turns
high-severity issues into a bounded targeted-retry loop and, when unresolved,
into blocking findings for the save gate.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from earnings_agents.agent.derive import format_value_for_llm
from earnings_agents.agent.loop import run_agent_loop
from earnings_agents.agent.tools import build_pi_tools

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """\
You are an independent financial-data verifier.  A first extraction agent has
already read this earnings document and extracted values.  Your job is to
AUDIT that extraction against the source — you are a second reader, not a
re-extractor.

The document may be an SEC 8-K/EDGAR exhibit, an IR-hosted PDF, a shareholder
letter, or another website PDF.  It is a PLAIN-TEXT rendering with line
numbers.

WHAT YOU RECEIVE
  • The target concept checklist (bracketed keys the extraction was asked for).
  • The extracted values (key → value), with the line evidence the first agent
    reported where available.
  • The document and the same navigation tools (search/read_lines/
    detect_currency/detect_scale).

YOUR CHECKLIST — verify, in order:
  1. For each extracted value with line evidence: read those lines.  Is the
     number really there?  Is it from the CURRENT-period column (not the
     prior-year comparison column)?  Does its sign match (parenthesized =
     negative)?
  2. For each concept the extraction agent reported as missing: search the
     filing.  Is the row actually printed somewhere (maybe under a synonym)?
     If printed → report missing_row with the line range so a targeted retry
     can extract it.  If genuinely absent → report absent_ok.
  3. Scale: call detect_scale() on each monetary table's declaration and
     compare with the reported scale.  A wrong scale corrupts the value.
     Per-share and percentage rows (EPS, margins, ratios, share counts) are
     ALWAYS as-is — never flag them as wrong_scale just because they sit
     next to millions-scale dollar rows.  Per-share values are small
     decimals (0.32, 1.30, 8.94) — treat them as exact, not as rounding
     errors or scale mistakes.
  4. Currency: call detect_currency() on each monetary table and compare with
     the reported currency.  Never suggest converting currency — non-USD rows
     must be reported as currency_conflict so they are excluded.
  5. Hierarchy/segments: if a segment (SEGMENT) value is attributed to a
     parent, confirm the row sits under that parent in the filing.
  6. Identity: confirm the document belongs to the target company (search for
     the company's name).  If it does not, report identity_mismatch.

RULES
  • NEVER change a value, NEVER re-extract the whole document, NEVER invent
    numbers.  Report only what you can confirm by reading lines.
  • Never change the SIGN of a reported value.  When checking a derived
    relation (e.g. Gross Profit = Revenue − Cost of Revenue), use the
    extracted values EXACTLY as reported — do not assume a minus sign.  If the
    arithmetic result looks implausible, re-read the exact lines before
    reporting an issue.
  • Do not flag a value for a small rounding difference (≤ 1 unit at the
    reported scale) — the table may print rounded figures.
  • Only report issues you are confident about.  When in doubt, mark
    needs_confirmation instead of guessing.
  • Efficiency matters: use the provided line evidence to jump straight to the
    relevant lines; do not re-read ranges the first agent already read unless
    you need to verify them.

When done, call finalize_verification with the report JSON (see its
description for the exact schema).
"""


VERIFIER_FINALIZE_DESCRIPTION = (
    "Call this when you have completed the audit.  Pass a JSON string:\n"
    "{\n"
    '  "__verifier_status__": "verified" | "issues_found",\n'
    '  "__verifier_issues__": [\n'
    "    {\"type\": \"value_mismatch\" | \"missing_row\" | \"absent_ok\" |\n"
    "     \"wrong_scale\" | \"wrong_currency\" | \"wrong_parent\" |\n"
    "     \"identity_mismatch\" | \"needs_confirmation\",\n"
    '     "severity": "high" | "medium" | "low",\n'
    '     "concept": "<bracketed key or label>",\n'
    '     "message": "what you found",\n'
    '     "lines": [start, end],  // where the issue is in the document\n'
    '     "reported_value": <number or null>,\n'
    '     "found_value": <number or null>\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "status is \"verified\" only when every extracted value is confirmed and "
    "every missing concept is confirmed absent.  Report each issue once."
)


_VERIFIER_ISSUE_TYPES = {
    "value_mismatch", "missing_row", "absent_ok", "wrong_scale",
    "wrong_currency", "wrong_parent", "identity_mismatch", "needs_confirmation",
}

_REPORT_RX = re.compile(
    r'\{[^{}]*"__verifier_status__"[^{}]*\}', re.DOTALL
)


def _parse_verifier_report(result_str: str) -> dict[str, Any] | None:
    """Parse the verifier's finalize JSON into a structured report."""
    cleaned = (
        result_str.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]
    end_brace = cleaned.rfind("}")
    if end_brace >= 0:
        cleaned = cleaned[: end_brace + 1]
    try:
        parsed: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    status = str(parsed.get("__verifier_status__") or "").strip().lower()
    if status not in ("verified", "issues_found"):
        return None

    issues = parsed.get("__verifier_issues__")
    if not isinstance(issues, list):
        issues = []

    normalized: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        itype = str(issue.get("type") or "").strip().lower()
        if itype not in _VERIFIER_ISSUE_TYPES:
            continue
        severity = str(issue.get("severity") or "medium").strip().lower()
        if severity not in ("high", "medium", "low"):
            severity = "medium"
        normalized.append({
            "type": itype,
            "severity": severity,
            "concept": str(issue.get("concept") or ""),
            "message": str(issue.get("message") or ""),
            "lines": issue.get("lines"),
            "reported_value": issue.get("reported_value"),
            "found_value": issue.get("found_value"),
        })

    actionable = [
        i for i in normalized
        if i["type"] not in ("absent_ok",)
    ]
    effective_status = "verified" if not actionable else "issues_found"
    return {
        "status": effective_status,
        "issues": normalized,
        "actionable_issues": actionable,
    }


def run_verifier(
    raw_text: str,
    *,
    ticker: str,
    target_concepts: list[dict],
    concept_metrics: dict[str, float],
    value_metadata_by_id: dict[str, dict],
    cik: str | None = None,
    company_name: str = "",
    company_industry: dict | None = None,
    document_map: list[dict] | None = None,
    missing_labels: list[str] | None = None,
    ambiguous_paths: set[str] | None = None,
    max_steps: int = 30,
) -> dict[str, Any] | None:
    """Run the verifier agent over *raw_text* and return the audit report.

    Returns ``None`` when the verifier produced no usable report (treated as
    ``needs_confirmation`` by the caller — never as ``verified``).
    """
    id_label = {c["_id"]: c.get("label", "?") for c in target_concepts}
    extracted_lines: list[str] = []
    for cid, val in sorted(concept_metrics.items(), key=lambda kv: str(kv[0])):
        ev = (value_metadata_by_id or {}).get(cid, {})
        lines = ev.get("source_lines")
        loc = f" lines {lines[0]}-{lines[1]}" if isinstance(lines, list) and len(lines) == 2 else ""
        # Decimal-preserving format: rendering EPS as "0" / "9" via :,.0f
        # makes correct per-share values look wrong against the printed
        # document and generates unfalsifiable value_mismatch flags.
        extracted_lines.append(
            f"  • [{id_label.get(cid, cid)}] = {format_value_for_llm(val)}{loc}"
        )
    extracted_block = "\n".join(extracted_lines) if extracted_lines else "  (none)"

    checklist = [
        c for c in target_concepts
        if not str(c.get("concept") or c.get("taxonomy_key") or "").lower().startswith("system:")
        and not c.get("calculated")
    ]
    checklist_lines = []
    for c in checklist:
        key = (c.get("taxonomy_key") or c.get("concept") or "").strip()
        tag = " [SEGMENT]" if (c.get("dimension") or c.get("dimension_concept")) else ""
        checklist_lines.append(f"  • [{(key or c.get('label') or '?')}] — \"{c.get('label')}\"{tag}")
    checklist_block = "\n".join(checklist_lines[:400])

    missing_block = ""
    if missing_labels:
        missing_block = (
            "CONCEPTS THE EXTRACTION AGENT COULD NOT LOCATE (verify each):\n"
            + "\n".join(f"  • {lbl}" for lbl in missing_labels[:200])
        )

    ambiguity_block = ""
    if ambiguous_paths:
        id_label = {c["_id"]: c.get("label", "?") for c in target_concepts}
        path_rows: list[str] = []
        for p in sorted(ambiguous_paths)[:30]:
            rows = [
                id_label.get(c["_id"], c["_id"])
                for c in target_concepts
                if (c.get("path") or "").strip() == p
            ]
            path_rows.append(f"  • {p}: {', '.join(rows[:6])}")
        ambiguity_block = (
            "AMBIGUOUS HIERARCHY PATHS — multiple rows share this path, so "
            "auto-derivation refused to attach children. Confirm which row is "
            "the real parent in the filing (or report wrong_parent issues):\n"
            + "\n".join(path_rows)
        )

    tools = build_pi_tools(
        raw_text, prior_values={}, cik=cik, company_name=company_name,
        company_industry=company_industry,
        document_map=document_map,
        target_concepts=target_concepts,
    )

    initial_message = (
        f"Document: {len(raw_text):,} chars. Audit the extraction below against "
        f"the source document. Use the navigation tools to verify values, then "
        "finalize_verification.\n\n"
        "EXTRACTED VALUES:\n"
        f"{extracted_block}\n\n"
        "TARGET CONCEPTS:\n"
        f"{checklist_block}\n\n"
        f"{missing_block}\n\n"
        f"{ambiguity_block}"
    )

    report = run_agent_loop(
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        initial_message=initial_message,
        tools=tools,
        ticker=ticker,
        max_steps=max_steps,
        finalize_name="finalize_verification",
        finalize_description=VERIFIER_FINALIZE_DESCRIPTION,
        parse_final_result=_parse_verifier_report,
        recovery_regex=_REPORT_RX,
    )
    if report is None:
        logger.warning("Verifier for %s: no report produced", ticker)
        return None
    logger.info(
        "Verifier for %s: status=%s issues=%d",
        ticker, report["status"], len(report["issues"]),
    )
    return report
