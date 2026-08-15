"""Agent tools — document navigation + calculation for the pi-style agent.

The agent receives a plain-text document (HTML tags stripped) and navigates
it with three navigation tools, exactly like a coding agent navigates a
codebase:

    get_document_info()           — size overview
    read_lines(start, end)        — read a line range
    search(query, context_lines)  — find matching lines with context

Plus three utility tools:

    get_prior_value(metric)    — DB lookup for sanity checking
    verify_identity(r,c,g)     — GP = Rev − CoR check
    calculate(expression)      — arithmetic evaluator for derived metrics

The calculate tool lets the agent compute derived values itself (Gross Profit,
Operating Income, Net Income, EPS, margins) instead of relying on external
deterministic derivation code.
"""
from __future__ import annotations

import re

from langchain_core.tools import tool as _lc_tool


def build_pi_tools(
    document_text: str,
    prior_values: dict[str, float],
    cik: str | None = None,
    company_name: str = "",
    company_industry: dict | None = None,
    document_map: list[dict] | None = None,
) -> list:
    """Build the pi-style tool set for raw document navigation.

    Args:
        document_text: Full plain-text of the filing (possibly a concatenated
            multi-exhibit bundle).
        prior_values: Prior-period values for sanity checking.
        cik: Company CIK for company info lookup.
        company_name: Company name for display.
        company_industry: Cached ``{sic_code, sic_description}`` profile from
            ``normalize_data.companies`` — ``get_company_info`` returns it
            directly instead of re-querying MongoDB (falls back to a DB
            lookup when absent).
        document_map: Exhibit boundaries inside *document_text*
            (``[{exhibit, url, line_start, line_end, truncated, skipped}]``).
            When provided, ``get_document_info`` lists it so the agent can
            jump straight to the exhibit holding the data it needs.
    """
    lines = document_text.split("\n")
    total_lines = len(lines)
    total_chars = len(document_text)
    document_map = document_map or []

    # ── Search index: word → line numbers ──────────────────────────────
    _line_index: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        for word in re.findall(r"[a-zA-Z]{4,}", line.lower()):
            _line_index.setdefault(word, []).append(i)

    # ── 1. Document info ───────────────────────────────────────────────
    @_lc_tool
    def get_document_info() -> str:
        """Get an overview of the document bundle's size and structure.

        Returns total line/character counts, the exhibit map (which document
        holds which line ranges), and a preview of the first lines.  Use this
        FIRST to understand the document before navigating — income-statement
        detail may live in a supplemental exhibit.
        """
        # Show first 10 non-empty lines as a "table of contents" preview
        preview_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 3 and not stripped.startswith(("*", "(", "|", "§", "═")):
                preview_lines.append(stripped[:120])
                if len(preview_lines) >= 15:
                    break
        toc = "\n".join(f"  {i+1:4d}: {l}" for i, l in enumerate(preview_lines[:15]))

        if document_map:
            map_lines = []
            for i, d in enumerate(document_map):
                status: list[str] = []
                if d.get("skipped"):
                    status.append(f"SKIPPED — {d.get('reason') or 'unavailable'}")
                elif d.get("error"):
                    status.append(f"ERROR: {str(d['error'])[:60]}")
                elif d.get("truncated"):
                    status.append("TRUNCATED")
                rng = (
                    f"lines {d['line_start']}-{d['line_end']}"
                    if d.get("line_start") else "not loaded"
                )
                suffix = f"  [{'; '.join(status)}]" if status else ""
                map_lines.append(
                    f"  {i+1}. {d.get('exhibit') or d.get('url') or '?'}: {rng}{suffix}"
                )
            docs_block = "\n".join(map_lines)
        else:
            docs_block = f"  single document — lines 1-{total_lines}"

        return (
            f"Document bundle: {total_chars:,} characters, {total_lines:,} lines, "
            f"{len(document_map) or 1} exhibit(s)\n"
            f"Exhibit map:\n{docs_block}\n"
            f"First lines (preview):\n{toc}"
        )

    # ── 2. Read lines ─────────────────────────────────────────────────
    @_lc_tool
    def read_lines(start: int, end: int) -> str:
        """Read a range of lines from the document.

        Args:
            start: Starting line number (1-based, inclusive).
            end: Ending line number (1-based, inclusive).

        Returns the specified lines with line numbers for reference.
        Use this to read a section you've located via search().
        """
        if start < 1 or end > total_lines or start > end:
            return (
                f"Invalid range. Document has {total_lines:,} lines. "
                f"Request read_lines(1, min(500, {total_lines})) for the start."
            )

        result_lines: list[str] = []
        for i in range(start - 1, min(end, total_lines)):
            line_text = lines[i][:300]
            result_lines.append(f"{i + 1:5d}: {line_text}")

        header = f"Lines {start}-{min(end, total_lines)} of {total_lines:,}:"
        return header + "\n" + "\n".join(result_lines)

    # ── 3. Search ──────────────────────────────────────────────────────
    @_lc_tool
    def search(query: str, context_lines: int = 3) -> str:
        """Search the document for lines containing a term or phrase.

        Args:
            query: Text to search for (e.g. "Revenue", "Cost of", "interest").
            context_lines: Lines of context before/after each match (default 3).

        Returns matching lines with line numbers.  Use read_lines() to
        read the surrounding section in full.

        TIP: Search for metric names, dollar amounts, or date headers
        to locate the income statement quickly.
        """
        if not query.strip():
            return "Empty query — provide a search term."

        query_lower = query.lower()
        words = re.findall(r"[a-zA-Z]{4,}", query_lower)

        candidates: set[int] = set()
        if words:
            candidates = set(_line_index.get(words[0], []))
            for w in words[1:]:
                candidates &= set(_line_index.get(w, []))
            if not candidates:
                for w in words:
                    candidates |= set(_line_index.get(w, []))

        # Also direct substring match
        direct_matches: set[int] = set()
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                direct_matches.add(i)

        all_matches = candidates | direct_matches
        if not all_matches:
            return f"No lines found matching '{query}'."

        # Group into blocks with context
        ctx = context_lines
        sorted_matches = sorted(all_matches)
        blocks: list[list[int]] = []
        current_block = [sorted_matches[0]]
        for m in sorted_matches[1:]:
            if m - current_block[-1] <= ctx * 2:
                current_block.append(m)
            else:
                blocks.append(current_block)
                current_block = [m]
        blocks.append(current_block)

        out_lines: list[str] = []
        for block in blocks[:15]:
            block_start = max(0, block[0] - ctx)
            block_end = min(total_lines, block[-1] + ctx + 1)
            out_lines.append(f"── lines {block_start + 1}-{block_end} ──")
            for i in range(block_start, block_end):
                marker = ">>>" if i in all_matches else "   "
                line_text = lines[i][:200]
                out_lines.append(f"{marker} {i + 1:5d}: {line_text}")

        if len(blocks) > 15:
            out_lines.append(f"... ({len(blocks) - 15} more blocks)")
        return "\n".join(out_lines)

    # ── 4. Prior value lookup ──────────────────────────────────────────
    @_lc_tool
    def get_prior_value(metric_description: str) -> str:
        """Look up the prior reporting period's value for a metric.

        Args:
            metric_description: A short description, e.g. "Revenue".

        Returns the prior period's value for reference.  Use ONLY to
        confirm the correct column — do NOT return it as your answer.
        """
        desc_lower = metric_description.lower().strip()
        for label, value in prior_values.items():
            if label.lower() == desc_lower:
                return f"Prior period '{label}': {value:,.0f}"
        for label, value in prior_values.items():
            label_lower = label.lower()
            if desc_lower in label_lower or label_lower in desc_lower:
                return f"Prior period '{label}': {value:,.0f}"
        if prior_values:
            available = ", ".join(list(prior_values.keys())[:10])
            return f"No prior value for '{metric_description}'. Available: {available}"
        return "No prior-period reference values available."

    # ── 5. Identity verification ──────────────────────────────────────
    @_lc_tool
    def verify_identity(
        revenue: float,
        cost_of_revenue: float,
        gross_profit: float,
    ) -> str:
        """Verify that Gross Profit = Revenue − Cost of Revenue.

        Call before finalizing to confirm correct column selection.
        """
        expected = revenue - cost_of_revenue
        diff = abs(gross_profit - expected)
        pct = (diff / max(abs(revenue), 1)) * 100
        if pct < 1.0:
            return (
                f"✓ VERIFIED: Revenue ({revenue:,.0f}) − Cost of Revenue "
                f"({cost_of_revenue:,.0f}) = {expected:,.0f} ≈ Gross Profit "
                f"({gross_profit:,.0f}). Column is correct."
            )
        else:
            return (
                f"✗ FAILED: Revenue ({revenue:,.0f}) − Cost of Revenue "
                f"({cost_of_revenue:,.0f}) = {expected:,.0f}, but Gross Profit "
                f"= {gross_profit:,.0f}. Difference: {diff:,.0f} ({pct:.2f}%). "
                f"Wrong column — re-read the income statement."
            )

    # ── 6. Calculator (for derived metrics) ─────────────────────────────
    @_lc_tool
    def calculate(expression: str) -> str:
        """Evaluate a simple arithmetic expression and return the result.

        Use this to compute derived metrics AFTER extracting raw values from
        the document.  For example:

          calculate("1006300000 - 509800000")        → Gross Profit
          calculate("(753214000 - 272411000) / 753214000 * 100")  → margin %
          calculate("33900000000 / 15700000000")     → EPS

        Supports: +, -, *, /, parentheses, decimal numbers.
        No variables, no functions, no imports — pure arithmetic only.
        """
        import ast
        import operator

        _ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def _eval(node):
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op_fn = _ops.get(type(node.op))
                if op_fn is None:
                    raise ValueError(f"Unsupported operator: {ast.dump(node.op)}")
                return op_fn(left, right)
            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op_fn = _ops.get(type(node.op))
                if op_fn is None:
                    raise ValueError(f"Unsupported unary operator")
                return op_fn(operand)
            if isinstance(node, ast.ParenExpr):
                return _eval(node.value)
            raise ValueError(f"Unsupported expression type: {ast.dump(node)}")

        try:
            # Clean the expression: remove commas, currency symbols, whitespace
            clean = expression.replace(",", "").replace("$", "").strip()
            tree = ast.parse(clean, mode="eval")
            result = _eval(tree.body)
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return str(result)
        except Exception as exc:
            return f"Error evaluating '{expression}': {exc}"

    # ── 7. Company info lookup ──────────────────────────────────────────
    @_lc_tool
    def get_company_info() -> str:
        """Get the company's industry profile (SIC code + description).

        ADVISORY CONTEXT ONLY: use it to understand what kind of company
        you're dealing with (bank, insurer, manufacturer, retailer) so you
        can recognize industry terminology in the filing.  Industry data can
        never supply values — extract numbers from the filing only.
        """
        from earnings_agents.agent.industry import normalize_company_industry

        profile = normalize_company_industry(company_industry)
        if profile:
            return (
                f"Company: {company_name or '?'}\n"
                f"CIK: {cik or 'unknown'}\n"
                f"Industry: {profile['sic_description'] or '?'} "
                f"(SIC {profile['sic_code'] or '?'})"
            )

        # Fallback — callers that built the toolset without a cached profile
        # (or with an empty one) get the full document lookup.
        try:
            from earnings_agents.integrations.normalize import _get_client, _NORMALIZE_DB
            db = _get_client()[_NORMALIZE_DB]
            doc = db["companies"].find_one({"cik": cik}) if cik else None
            if not doc:
                return f"Company not found in database. CIK: {cik or 'unknown'}"

            ind = doc.get("industry") or {}
            corp = doc.get("corporate_info") or {}
            market = doc.get("market_info") or {}
            sic = ind.get("sic_code", "?")
            sic_desc = ind.get("sic_description", "?")
            fy = corp.get("fiscal_year_end", "?")
            exchanges = ", ".join(market.get("exchanges", []))
            entity = corp.get("entity_type", "?")

            return (
                f"Company: {company_name or doc.get('name', '?')}\n"
                f"CIK: {cik}\n"
                f"Industry: {sic_desc} (SIC {sic})\n"
                f"Fiscal Year End: {fy} (MMDD format — e.g. 1231 = Dec 31)\n"
                f"Exchange: {exchanges}\n"
                f"Entity Type: {entity}"
            )
        except Exception as exc:
            return f"Error looking up company info: {exc}"

    return [get_document_info, read_lines, search, get_prior_value, verify_identity, calculate, get_company_info]
