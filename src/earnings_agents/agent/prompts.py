"""Agent prompts — pi-style system prompt for raw document navigation."""
from __future__ import annotations

from typing import Any


PIPELINE_SYSTEM_PROMPT = """\
You are a financial data extraction agent.  Your job is to extract specific
income-statement metrics from an SEC earnings press release (8-K Exhibit 99.1).

The document is a PLAIN-TEXT rendering of the original filing(s) — HTML
press releases (tags stripped, line breaks preserved) or PDF shareholder
letters (each page marked with a "PDF page n of m" separator).

The text may be a BUNDLE of several exhibits separated by ══ DOCUMENT n OF m
headers — e.g. Exhibit 99.1 (press release), 99.2 (presentation),
99.3 (supplemental information).  The income-statement detail may live in the
supplemental exhibits, not the press release: call get_document_info() to see
the exhibit map, then read_lines() into the document that holds the rows you
need.  The reporting-period header is in the FIRST document (press release).

YOUR TOOLS
  • get_document_info() — overview: total lines, chars, first lines preview
  • get_company_info() — industry, fiscal year end, market info
  • read_lines(start, end) — read any line range (e.g. read_lines(120, 200))
  • search(query) — find lines containing a term, with context
  • get_prior_value(metric) — look up a prior-period value for reference
  • verify_identity(revenue, cost_of_revenue, gross_profit) — verify column
  • calculate(expression) — evaluate arithmetic for derived metrics

HOW TO WORK — exactly like a coding agent navigating a repo:
  1. Start with search("Revenue") or search("Net income") to locate the
     income statement.  Use search("In thousands") or search("In millions")
     to find the scale declaration.
  2. Use read_lines() to read the income statement section.
  3. Identify the CURRENT period column by reading column headers.
     Columns typically show: "Three Months Ended [Current Date]" vs
     "Three Months Ended [Prior Year Date]".
  4. Extract metrics from the most-recent column ONLY.
  5. Use search("interest") to find small but critical rows.
  6. Call verify_identity() BEFORE finalizing to confirm the column is right.

EFFICIENCY (no step limit — but be deliberate):
  • Locate the CONSOLIDATED income statement FIRST and extract ALL top-level
    metrics from it before exploring anything else.
  • Only then search segments / supplemental exhibits for the remaining
    concepts.  Do NOT re-read ranges you have already read.
  • Finish with verify_identity() and then finalize_extraction().

COMPLETENESS — CRITICAL (a missed row blocks cost derivation):
  • The concept list mirrors the income statement.  Extract EVERY printed row
    that maps to a concept — including the individual COST and expense lines
    inside the operating-expenses block (e.g. "Cloud and software",
    "Hardware", "Services", "Sales and marketing", "Research and
    development", "Amortization of intangible assets", "Restructuring"),
    NOT just the subtotals.  A line that maps to a concept must be reported
    even when it is not a subtotal.
  • Match rows by MEANING, not wording: filings often name a row differently
    from its concept label.  Examples: filing "Cloud and software" = concept
    "Cloud Services And License Support Expenses"; filing "Software license
    updates and product support" = concept "Software Support".  When a cost
    row has no obvious label match, search the concept list for the closest
    concept rather than skipping it.
  • RECONCILE before finalizing (GAAP rows only — never Non-GAAP/Adjusted):
    sum the extracted cost rows with calculate() and compare with the printed
    total ("Total operating expenses" / "Costs and Expenses").  Sum the
    extracted revenue rows and compare with the printed total revenues.  If
    the sums disagree, a row is missing — search("Cost") / search("Cloud") /
    read_lines() the operating-expenses block again, extract the missing
    line, and re-check.  Also use verify_identity() after reconciling.
  • Treat the concept list as a checklist: for each concept, either find its
    row in the filing or confirm the row is genuinely absent — do not drop a
    concept because its label was hard to match.
  • Variant rows are DISTINCT concepts — extract them when printed (GAAP,
    current column only): e.g. "Net income available to common shareholders"
    (vs "Net income"), "EPS attributable to common shareholders" (basic vs
    diluted), "Income (loss) from continuing operations before income taxes",
    "Preferred stock dividends".  If a statement prints both the headline
    total and its "available to common" / "continuing operations" variant,
    report BOTH under their respective concepts.

WHAT TO IGNORE
  • Any section labeled "Non-GAAP", "Adjusted", "Reconciliation of GAAP"
  • Forward-looking guidance, outlook, or forecast tables
  • Balance sheet data (unless you need share counts for EPS)
  • Cash flow statement data
  • Footnote detail below the main income statement table
  • Anything from the prior-year comparison column

SIGNS — CRITICAL (SEC/IR convention, applies to every source: EDGAR HTML,
PDF letters, presentations):
  • A parenthesized amount or a leading minus in the source means NEGATIVE.
    "(1,234)" and "-1,234" both become -1234 — include the minus sign in the
    number you report.
  • Expense/loss rows are typically shown parenthesized or as negatives:
    "Interest expense (175,685)", "Provision for income taxes (667,172)",
    "(Loss) from operations", "Net cash used in operating activities".
  • Report the sign EXACTLY as printed in the CURRENT column.  If the current
    column shows (1,234) but the prior-year column shows 1,234, the current
    value is STILL -1234 — never copy a sign from another column.
  • Revenue, income, and profit rows are usually unsigned — keep them positive.
  • Negative per-share values (e.g. diluted EPS in a loss quarter) keep their
    sign and are still as-is.
  • Never invent a sign the document does not show, and never drop a minus.

EXTRACTION RULES
  • Use the EXACT bracketed key from the concept list — e.g. [us-gaap:Revenue]
  • Report raw table numbers exactly as printed — do NOT multiply
  • Percentages and per-share values are always as-is (never scaled)
  • "(1,234)" means negative: -1234
  • Extract the individual cost/expense rows too — any printed line that maps
    to a concept is reported, not just subtotals
  • OMIT concepts you cannot find — never return nulls or zeros
  • Set __scale__ to the declared unit: "thousands", "millions", "billions", or "as-is"
  • If different exhibits declare different units (one "in thousands", another
    "in millions"), report ALL dollar values in ONE scale: convert with
    calculate() and set __scale__ to the scale you used.  Percentages and
    per-share values are always as-is (never converted).
  • PERIOD AGENT CONTRACT: the period value in the system context is
    authoritative.  Do not independently decide annual/quarterly or quarter;
    extract the column selected by the period agent.
  • Q4 IS THE FISCAL YEAR-END — never extract a fourth-quarter column.  If the
    document shows both a Q4 and a fiscal-year column, extract the
    FISCAL-YEAR (annual) column selected by the period agent.

MULTI-COMPONENT METRICS — CRITICAL:
  Some metrics are SUMS of multiple line items on the income statement.
  If you see a subtotal row (e.g. "Cost of sales") AND additional cost rows
  below it (e.g. "Amortization of acquired developed intangibles",
  "Depreciation", "Impairment charges"), you MUST SUM them and report the
  TOTAL as the metric value.

  Example: if the filing shows:
    Cost of sales: 509.8
    Amortization of acquired developed intangibles: 19.2
    Gross profit: 477.3

  Then Cost of Revenue = 509.8 + 19.2 = 529.0
  Use calculate("509.8 + 19.2") to sum them.
  Verify: Revenue (1006.3) − CoR (529.0) = GP (477.3) ✓

{concept_list}

WORKFLOW
  search("Revenue") → read_lines around match → extract → calculate() → verify → finalize
"""


COMPANY_IDENTITY_RULE = """\
COMPANY IDENTITY — verify BEFORE extracting (applies to every source: EDGAR
HTML, PDF shareholder letters, presentations):
  • This task is for {company_name} ({ticker}).  The document MUST belong to
    this company — a readable period header alone is NOT proof of identity.
  • Verify identity early: call get_company_info(), search() for the
    company's unique name and its well-known business terms, and sanity-check
    the magnitude of headline figures against get_prior_value().
  • If the document is clearly for a DIFFERENT company — the other company's
    name appears throughout, its terminology does not match, or the figures
    are inconsistent with this company's history — call finalize_extraction
    with "__company_mismatch__": true and return NO metric values.
  • NEVER extract one company's numbers under another company's ticker.
"""


FINALIZE_DESCRIPTION = (
    "Call this when you have extracted ALL metrics.  Pass a JSON string with:\n"
    "  - __scale__: \"millions\", \"thousands\", \"billions\", or \"as-is\"\n"
    "  - Each concept's value, keyed by its EXACT bracketed key (e.g. [us-gaap:Revenue])\n"
    'Negative amounts ("(1,234)" in the filing) must carry the minus sign: -1234.\n'
    "If the document does NOT belong to the target company, set "
    '\"__company_mismatch__\": true and return NO metric values.\n'
    "OMIT any concept you cannot find."
)


def build_concept_list(
    target_concepts: list[dict],
    recent_concept_ids: set[str] | None = None,
    calculated_concepts: list[dict] | None = None,
) -> str:
    """Render the extraction concept list for the agent prompt.

    Filtering is NOT done here: ``target_concepts`` is already hard-filtered
    upstream to concepts valued in the last ``PROMPT_HISTORY_PERIODS`` periods
    (quarterly → quarterly periods, annual → annual periods).  Only
    system-calculated concepts (``system:`` prefix) are excluded — those are
    derived deterministically after extraction, not extracted by the agent.
    *recent_concept_ids* / *calculated_concepts* are kept for API compatibility.
    """
    prompt_concepts = [
        c for c in target_concepts
        if not ((c.get("concept") or c.get("taxonomy_key") or "")).startswith("system:")
    ]

    lines: list[str] = []
    for c in prompt_concepts:
        label = c.get("label", "")
        if not label:
            continue
        taxonomy_key = (c.get("taxonomy_key") or c.get("concept") or "").strip()
        # Dimensional signal comes from the DB flags — the "|" in taxonomy_key
        # only fires for embedded member-tag labels (30 of ~109k docs) and also
        # appears on path-disambiguated non-dimensional rows, so it is neither
        # a necessary nor sufficient proxy.
        has_dim = bool(c.get("dimension") or c.get("dimension_concept"))

        tags: list[str] = []
        if has_dim:
            tags.append("SEGMENT")
        tag_str = f"  [{' | '.join(tags)}]" if tags else ""

        if taxonomy_key:
            lines.append(f"  • [{taxonomy_key}]  — \"{label}\"{tag_str}")
        else:
            lines.append(f"  • \"{label}\"  (no taxonomy key){tag_str}")

    return "\n".join(lines)


def build_retry_briefing(
    findings: list[dict],
    prev_metrics: dict[str, Any],
    missing_toplevel: list[str] | None,
    missing_segments: list[str] | None,
) -> str:
    """Structured retry briefing from previous-pass findings."""
    sections: list[str] = []

    flagged_keys: set[str] = set()
    for f in findings:
        for k in (f.get("keys") or []):
            flagged_keys.add(k)

    carry_forward = {
        k: v for k, v in prev_metrics.items()
        if k not in flagged_keys
        and not k.startswith("__")
        and isinstance(v, (int, float))
    }
    if carry_forward:
        lines = ["CARRY FORWARD (correct — do NOT re-extract):"]
        for k in sorted(carry_forward.keys()):
            lines.append(f"  • {k}: {carry_forward[k]:,.0f}")
        sections.append("\n".join(lines))

    high_findings = [f for f in findings if f.get("severity") == "high"]
    fix_lines: list[str] = []
    for f in high_findings:
        ftype = f.get("type", "")
        msg = f.get("message", "")
        ev = f.get("evidence") or {}
        fix_lines.append(f"  • {msg}")
        if ftype == "identity_violation":
            rev = ev.get("revenue")
            cor = ev.get("cost_of_revenue")
            gp = ev.get("gross_profit")
            if isinstance(rev, (int, float)) and rev and isinstance(gp, (int, float)):
                implied_cor = rev - gp
                gap = implied_cor - cor if isinstance(cor, (int, float)) else 0
                fix_lines.append("    DIAGNOSIS:")
                fix_lines.append(f"      Revenue         = {rev:>22,.0f}  (correct)")
                if isinstance(cor, (int, float)):
                    fix_lines.append(f"      Your extraction = {cor:>22,.0f}  ← WRONG")
                    fix_lines.append(f"      Expected CoR    = {implied_cor:>22,.0f}  (= Revenue − Gross Profit)")
                    if abs(gap) > 1000:
                        fix_lines.append(f"      GAP             = {gap:>22,.0f}")
                        fix_lines.append(
                            f"      ⚠ The gap of ~{abs(gap):,.0f} suggests you extracted a "
                            f"SUBTOTAL (e.g. 'Cost of sales') but missed additional cost "
                            f"components like 'Amortization', 'Depreciation', or other cost "
                            f"lines near the CoR row.  Search for 'Amortization', 'Depreciation', "
                            f"'impairment' near the income statement and SUM all cost "
                            f"components before finalizing."
                        )
                else:
                    fix_lines.append(f"      → Cost of Revenue should be ~{implied_cor:,.0f}")

    if fix_lines:
        sections.append(
            "WRONG METRICS — re-extract from CURRENT-PERIOD column only:\n"
            + "\n".join(fix_lines)
        )

    missing_lines: list[str] = []
    if missing_toplevel:
        missing_lines.append("MISSING from primary income statement:")
        for lbl in missing_toplevel[:10]:
            missing_lines.append(f"  • {lbl}")
    if missing_segments:
        missing_lines.append("MISSING segment/dimensional:")
        for lbl in missing_segments[:20]:
            missing_lines.append(f"  • {lbl}")
    if missing_lines:
        sections.append("\n".join(missing_lines))

    # Final instruction: only return flagged metrics
    if carry_forward:
        sections.append(
            "IMPORTANT: Do NOT include CARRY FORWARD metrics in your output. "
            "They are already correct.  Your finalize_extraction JSON should "
            "contain ONLY the WRONG and MISSING metrics listed above."
        )

    return "\n\n".join(sections)
