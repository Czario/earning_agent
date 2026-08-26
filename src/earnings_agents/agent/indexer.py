"""LLM-backed document section locator.

``find_sections`` (agent/tools.py) uses this to build a per-document section
map: an LLM reads the numbered plain-text document and reports the line ranges
of the income statement, segment results, balance sheet, cash flow, EPS/share
data, and other sections.  The map is a NAVIGATION AID for the agents — the
extraction agent always re-reads a range with ``read_lines`` before extracting,
so a mislocated range costs a re-read, never a wrong value.

Everything here is model-driven: no regex, no header heuristics, no structure
assumptions — the LLM looks at the actual document and decides what is where.
The map is built lazily on the first ``find_sections`` call (during the period
pass), cached for the run, and persisted to state for the extraction pass.

Latency: the indexer output is a compact FIXED-SCHEMA line-range map (~10
short sections, no notes) so generation stays small, and the call can be routed
to a fast provider via ``INDEX_LLM_PROVIDER`` / ``INDEX_LLM_MODEL`` even when
the agent loops run on a slower default provider (e.g. gemini-2.5-flash for
the indexer, deepseek for the loops).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from earnings_agents.config import LLM_PROVIDER

logger = logging.getLogger(__name__)

# Numbered-text slice handed to the indexer.  Small-context providers (ollama)
# get a smaller slice; the model reports coverage honestly when truncated, so
# the agents know to fall back to search()/read_lines() beyond it.
_DEFAULT_MAX_CHARS = 60_000
_OLLAMA_MAX_CHARS = 12_000

INDEXER_SYSTEM_PROMPT = """\
You are a document section locator for an earnings press release (plain text).

The document text below has every line prefixed with its 5-digit 1-based line
number (e.g. "  123: Revenue ...").  Identify which sections below are present
and report their exact INCLUSIVE line ranges [start, end]:

  - company_header: the opening lines with the company name and the reporting
    period header (e.g. "Three Months Ended June 30, 2026")
  - income_statement: Consolidated Statements of Operations (revenue, cost of
    revenue, gross profit, operating expenses, net income)
  - segment_results: business-segment / product-line / channel revenue
    breakdown (often prose — e.g. a "Key Business Metrics" section with
    Online vs Wholesale revenue, or per-brand revenue)
  - balance_sheet: consolidated balance sheet
  - cash_flow: consolidated cash flow statement
  - eps_data: earnings-per-share / share-count detail
  - notes: footnotes / GAAP reconciliation
  - guidance: outlook / forward-looking guidance

OMIT keys for sections that are not present.  When a section spans several
sub-blocks, report ONE range covering the whole block.  If the document
extends beyond the text you received, set coverage to "partial".  Line
numbers MUST be accurate — downstream readers use them directly.

Return ONLY JSON, no prose:
{"coverage": "full" | "partial", "income_statement": [285, 474], "segment_results": [27, 103]}
"""


def _parse_index_json(raw: str) -> dict[str, Any] | None:
    """Parse the indexer's JSON response into a cleaned section map.

    Accepts the compact flat form (``{"coverage": ..., "income_statement":
    [285, 474], ...}``) and the legacy list form (``{"sections": [{name,
    label, lines}]}``).  Sections without a valid inclusive line range are
    dropped.  Returns ``None`` on unparseable input so callers can fall back
    to a degraded (empty) map.
    """
    cleaned = (
        (raw or "").strip()
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
        parsed: Any = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    coverage = str(parsed.get("coverage") or "").strip().lower()
    coverage = coverage if coverage in ("full", "partial") else "unknown"
    summary = str(parsed.get("summary") or "").strip()[:200]

    def _lines_to_range(val: Any) -> list[int] | None:
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            return None
        if not all(isinstance(x, (int, float)) and x > 0 for x in val):
            return None
        lo, hi = int(val[0]), int(val[1])
        return [lo, hi] if lo <= hi else [hi, lo]

    sections: list[dict[str, Any]] = []
    raw_sections = parsed.get("sections")
    if isinstance(raw_sections, list):
        # legacy list form
        for s in raw_sections:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or "").strip()
            rng = _lines_to_range(s.get("lines"))
            if not name or rng is None:
                continue
            out: dict[str, Any] = {"name": name, "lines": rng}
            if s.get("label"):
                out["label"] = str(s["label"]).strip()[:120]
            if s.get("scale"):
                out["scale"] = str(s["scale"]).strip().lower()
            if s.get("currency"):
                out["currency"] = str(s["currency"]).strip().upper()
            sections.append(out)
    else:
        # compact flat form: any key besides coverage/summary with a range
        for key, val in parsed.items():
            if key in ("coverage", "summary"):
                continue
            rng = _lines_to_range(val)
            if rng is None:
                continue
            sections.append({"name": key, "lines": rng})

    return {"coverage": coverage, "summary": summary, "sections": sections}


def _build_indexer_llm() -> Any:
    """Build the LLM client for the indexer call.

    Routes to ``INDEX_LLM_PROVIDER`` / ``INDEX_LLM_MODEL`` when configured (a
    fast provider can index while the agent loops stay on a slower default),
    falling back to the default provider when the routed provider's API key is
    missing — the indexer must never fail a run on its own.
    """
    from earnings_agents.config import INDEX_LLM_MODEL, INDEX_LLM_PROVIDER
    from earnings_agents.llm import build_llm

    provider = (INDEX_LLM_PROVIDER or "").strip().lower() or None
    model = (INDEX_LLM_MODEL or "").strip() or None
    if provider:
        logger.info(
            "section indexer routed to provider=%s model=%s",
            provider, model or "(provider default)",
        )
    try:
        return build_llm(
            format_json=True, max_retries=0, provider=provider, model=model
        )
    except ValueError as exc:
        logger.warning(
            "section indexer provider unavailable (%s) — using default provider",
            exc,
        )
        return build_llm(format_json=True, max_retries=0)


def build_section_index(
    document_text: str,
    llm: Any | None = None,
    max_chars: int | None = None,
    query: str | None = None,
) -> tuple[dict[str, Any], float]:
    """Build the section map for *document_text* with ONE LLM call.

    Returns ``(index, elapsed_seconds)`` where *index* is
    ``{"coverage", "summary", "sections": [...]}``.  *llm* must expose
    ``invoke(str) -> str`` (default: the indexer-routed provider in JSON mode).
    The text is passed with line numbers so returned ranges match
    ``read_lines`` numbering.  *query* is an optional focus hint that steers
    the indexer (the full map is returned either way).
    """
    if max_chars is None:
        max_chars = (
            _OLLAMA_MAX_CHARS
            if (LLM_PROVIDER or "").strip().lower() == "ollama"
            else _DEFAULT_MAX_CHARS
        )

    lines = (document_text or "").split("\n")
    numbered = "\n".join(f"{i + 1:5d}: {ln[:200]}" for i, ln in enumerate(lines))
    truncated = len(numbered) > max_chars
    if truncated:
        numbered = numbered[:max_chars]
        numbered += "\n... (document continues beyond this text — coverage partial)"

    prompt = INDEXER_SYSTEM_PROMPT
    if query and query.strip():
        prompt += (
            "\nFOCUS QUERY (the reader is specifically looking for this — make "
            "sure the relevant section is included and its line range is "
            f"accurate): {query.strip()[:200]}\n"
        )
    prompt += "\n\nDOCUMENT TEXT (numbered):\n" + numbered

    if llm is None:
        llm = _build_indexer_llm()

    t0 = time.perf_counter()
    raw = llm.invoke(prompt)
    elapsed = time.perf_counter() - t0

    parsed = _parse_index_json(raw)
    if parsed is None:
        logger.warning(
            "section indexer: unparseable response (%.0f chars)", len(raw or "")
        )
        parsed = {"coverage": "unknown", "summary": "", "sections": []}
    if truncated and parsed.get("coverage") != "partial":
        parsed["coverage"] = "partial"
    return parsed, elapsed


def format_section_index(index: dict[str, Any] | None) -> str:
    """Render a section map for the agent (tool result / initial message)."""
    index = index or {}
    sections = index.get("sections") or []
    coverage = str(index.get("coverage") or "unknown")
    lines = [
        f"Document section map — {len(sections)} section(s), coverage: {coverage}"
    ]
    for i, s in enumerate(sections, 1):
        parts = [f"{i}. {s.get('name')}"]
        if s.get("label"):
            parts.append(f'"{s["label"]}"')
        if s.get("lines"):
            parts.append(f"lines {s['lines'][0]}-{s['lines'][1]}")
        if s.get("scale"):
            parts.append(f"scale: {s['scale']}")
        if s.get("currency"):
            parts.append(f"currency: {s['currency']}")
        lines.append("  " + " ".join(parts))
    if not sections:
        lines.append("  (no sections identified)")
    return "\n".join(lines)
