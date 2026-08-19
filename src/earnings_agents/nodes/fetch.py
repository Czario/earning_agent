"""Fetch ALL exhibits of the filing and concatenate them into plain text.

Many filings carry multiple EX-99 exhibits (press release, presentation,
supplemental information) and the income-statement detail often lives in a
supplemental exhibit (e.g. BofA's 99.3).  This node fetches every text
**or PDF** document, converts each to plain text (the ONLY pre-processing
step — HTML via BeautifulSoup, PDFs via pdfplumber), and concatenates them
with loud ``DOCUMENT n OF m`` headers so the agents can navigate the bundle
with the normal search/read tools.  PDFs cover manually-triggered filings
such as shareholder letters hosted on Q4CDN/company sites
(e.g. ``FINAL-Q2-26-Shareholder-Letter.pdf``).  The ``document_map`` in state
records each exhibit's line range, truncation, and fetch status.
"""
from __future__ import annotations

import logging
import os
import re
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup

from earnings_agents.config import FETCH_EXHIBIT_MAX_CHARS, FETCH_TOTAL_MAX_CHARS
from earnings_agents.integrations.html import MIN_CONTENT_CHARS, strip_sgml_wrapper
from earnings_agents.integrations.http import get as _http_get
from earnings_agents.integrations.http import get_binary as _http_get_binary
from earnings_agents.integrations.playwright import (
    fetch_binary,
    fetch_binary_curl,
    fetch_page_js,
)
from earnings_agents.state import EarningsAgentState

logger = logging.getLogger(__name__)

# Documents with these extensions are fetched and converted to plain text.
# PDFs (shareholder letters, Q4CDN-hosted supplements) are extracted with
# pdfplumber; anything else (images, etc.) is recorded as skipped in document_map.
_HTML_EXTENSIONS = (".htm", ".html", ".txt")
_PDF_EXTENSIONS = (".pdf",)
_FETCHABLE_EXTENSIONS = _HTML_EXTENSIONS + _PDF_EXTENSIONS


def _url_extension(url: str) -> str:
    """Lowercased file extension from the URL path (ignores query strings)."""
    return os.path.splitext(urlparse(url).path)[1].lower()


def _base_name(url: str) -> str:
    """Filename portion of a URL (decoded, query string stripped)."""
    name = os.path.basename(urlparse(url).path) or url
    return unquote(name)

_DOC_HEADER = "═" * 72


def _html_to_plain_text(html: str) -> str:
    """Convert HTML to plain text — the ONLY pre-processing step.

    Strips <script> and <style> tags (they contain code, not content).
    Converts all other tags to their text content, preserving line breaks.
    Collapses excessive whitespace but preserves document structure.

    Mechanical format conversion, not intelligent extraction.
    """
    html = strip_sgml_wrapper(html)

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse runs of 3+ blank lines into 2 (readability, not data loss)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs on each line
    text = re.sub(r"[^\S\n]+", " ", text)

    return text.strip()


def _pdf_to_plain_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber.

    Each page is prefixed with a ``PDF page n of m`` separator so the agents
    can locate content by page and multi-page letters navigate exactly like
    the multi-exhibit DOCUMENT bundle.  Mechanical format conversion, not
    intelligent extraction — same contract as ``_html_to_plain_text``.
    """
    import io

    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""
            parts.append(f"\n───── PDF page {i} of {n_pages} ─────\n{page_text}")
    # Collapse runs of spaces/tabs on each line (same as the HTML path).
    text = re.sub(r"[^\S\n]+", " ", "\n".join(parts))
    return text.strip()


def _exhibit_label(index: int, meta: dict | None = None, url: str = "") -> str:
    """Human label for an exhibit, preferring the EDGAR index description.

    EDGAR-sourced exhibits keep their index metadata (EX-99.n + description).
    Documents without metadata (manual PDF URLs, supplemental URLs) get a
    filename-based label so agents can tell documents apart.
    """
    meta = meta or {}
    exhibit = meta.get("exhibit") or f"EX-99.{index + 1}"
    description = (meta.get("description") or "").strip()
    if description:
        return f"Exhibit {exhibit} — {description}"
    if meta.get("exhibit"):
        # EDGAR-sourced exhibit without an index description — keep old labels.
        if index == 0:
            return f"Exhibit {exhibit} — primary press release"
        return f"Exhibit {exhibit} — supplemental exhibit"
    # No EDGAR metadata (manual trigger / supplement) — label from the filename.
    name = _base_name(url) if url else f"document {index + 1}"
    role = "primary filing document" if index == 0 else "supplemental document"
    return f"Document {index + 1} — {name} ({role})"


def fetch_filing_node(state: EarningsAgentState) -> EarningsAgentState:
    """Fetch every text exhibit and concatenate into ``raw_text``."""
    from earnings_agents.hooks import report_call

    url = state.get("discovered_file_url") or ""
    if not url:
        return {**state, "status": "failed", "error": "No filing URL available"}

    # ── Assemble the exhibit target list ─────────────────────────────────
    exhibit_meta: list[dict] = state.get("exhibit_meta") or []  # type: ignore[assignment]
    targets: list[dict] = []  # {"url", "meta"} in processing order
    if exhibit_meta:
        seen = set()
        for meta in exhibit_meta:
            u = meta.get("url")
            if u and u not in seen:
                targets.append({"url": u, "meta": meta})
                seen.add(u)
        if not targets or targets[0]["url"] != url:
            targets.insert(0, {"url": url, "meta": {}})
    else:
        targets.append({"url": url, "meta": {}})
        for u in state.get("supplemental_file_urls") or []:
            if isinstance(u, str) and u != url:
                targets.append({"url": u, "meta": {}})

    # ── Fetch + convert each text exhibit; concatenate with headers ──────
    chunks: list[str] = []
    document_map: list[dict] = []
    total_chars = 0
    line_counter = 1
    budget_exhausted = False
    n_fetchable = sum(
        1 for t in targets if _url_extension(t["url"]) in _FETCHABLE_EXTENSIONS
    )
    doc_no = 0
    any_pdf = False

    for index, target in enumerate(targets):
        u = target["url"]
        meta = target["meta"]
        ext = _url_extension(u)
        label = _exhibit_label(index, meta, u)

        if ext not in _FETCHABLE_EXTENSIONS:
            report_call(f"  [fetch]  skip non-text exhibit — {label}")
            document_map.append({
                "exhibit": label, "url": u,
                "line_start": None, "line_end": None,
                "truncated": False, "skipped": True, "reason": "non-text exhibit",
                "error": None,
            })
            continue
        if budget_exhausted:
            document_map.append({
                "exhibit": label, "url": u,
                "line_start": None, "line_end": None,
                "truncated": False, "skipped": True,
                "reason": "total size budget exhausted", "error": None,
            })
            continue

        report_call(f"  [http]  GET {u[:80]}")
        is_pdf = ext in _PDF_EXTENSIONS
        try:
            if is_pdf:
                is_sec = "sec.gov" in u
                direct_exc: Exception | None = None
                try:
                    # Start with a narrow static-asset request.  In
                    # particular, do not force the HTML browser User-Agent:
                    # Adobe's CDN can stall a requests connection when it sees
                    # Chrome/124 without the matching browser TLS fingerprint.
                    report_call(f"  [http]  PDF download  {u[:80]}")
                    response = _http_get_binary(u, sec=is_sec)
                    pdf_bytes = response.content
                    if not pdf_bytes:
                        raise ValueError("empty PDF response")
                    if not pdf_bytes.lstrip().startswith(b"%PDF"):
                        raise ValueError("response is not a PDF")
                except Exception as exc:
                    direct_exc = exc
                    pdf_bytes = b""

                if not pdf_bytes:
                    # curl is a useful second transport for static files.  It
                    # intentionally keeps curl's native UA for non-SEC hosts;
                    # see fetch_binary_curl for why this is not Chrome/124.
                    report_call(f"  [curl]  PDF fallback  {u[:80]}")
                    pdf_bytes = fetch_binary_curl(u, sec=is_sec)
                    if pdf_bytes and not pdf_bytes.lstrip().startswith(b"%PDF"):
                        pdf_bytes = b""

                if not pdf_bytes:
                    report_call(f"  [playwright]  PDF fallback  {u[:80]}")
                    pdf_bytes = fetch_binary(u)
                    if not pdf_bytes:
                        if direct_exc is not None:
                            raise direct_exc
                        raise ValueError("empty PDF response")
                    if not pdf_bytes.lstrip().startswith(b"%PDF"):
                        raise ValueError("browser response is not a PDF")

                text = _pdf_to_plain_text(pdf_bytes)
                any_pdf = True
            else:
                is_sec = "sec.gov" in u
                response = _http_get(u, sec=is_sec)
                html = response.text
                if "sec.gov" not in u:
                    quick_text = BeautifulSoup(html, "lxml").get_text(strip=True)
                    if len(quick_text) < MIN_CONTENT_CHARS:
                        report_call(f"  [playwright]  JS render  {u[:80]}")
                        html = fetch_page_js(u)
                text = _html_to_plain_text(html)
        except Exception as exc:
            logger.error("fetch exhibit %s (%s): %s", label, u, exc)
            if index == 0:
                kind = "PDF" if is_pdf else "HTML"
                return {**state, "status": "failed", "error": f"{kind} fetch failed: {exc}"}
            document_map.append({
                "exhibit": label, "url": u,
                "line_start": None, "line_end": None,
                "truncated": False, "skipped": True,
                "reason": f"fetch failed: {str(exc)[:120]}", "error": str(exc)[:200],
            })
            continue

        if not text:
            if index == 0:
                return {**state, "status": "failed", "error": "Empty document after text conversion"}
            document_map.append({
                "exhibit": label, "url": u,
                "line_start": None, "line_end": None,
                "truncated": False, "skipped": True,
                "reason": "empty after text conversion", "error": None,
            })
            continue

        truncated = False
        if len(text) > FETCH_EXHIBIT_MAX_CHARS:
            text = text[:FETCH_EXHIBIT_MAX_CHARS]
            truncated = True
        remaining_budget = FETCH_TOTAL_MAX_CHARS - total_chars
        if remaining_budget <= 0:
            budget_exhausted = True
            document_map.append({
                "exhibit": label, "url": u,
                "line_start": None, "line_end": None,
                "truncated": False, "skipped": True,
                "reason": "total size budget exhausted", "error": None,
            })
            continue
        if len(text) > remaining_budget:
            text = text[:remaining_budget]
            truncated = True
            budget_exhausted = True

        doc_no += 1
        trunc_note = "  [TRUNCATED]" if truncated else ""
        header = (
            f"DOCUMENT {doc_no} OF {n_fetchable} — {label}{trunc_note}\n"
            f"URL: {u}"
        )
        block = f"\n\n{_DOC_HEADER}\n{header}\n{_DOC_HEADER}\n{text}"

        line_start = line_counter
        chunks.append(block)
        line_counter += block.count("\n") + 1

        document_map.append({
            "exhibit": label, "url": u,
            "line_start": line_start, "line_end": line_counter - 1,
            "truncated": truncated, "skipped": False, "reason": None,
            "error": None,
        })
        total_chars += len(text)
        report_call(f"  [fetch]  {label}: {len(text):,} chars → lines {line_start}-{line_counter - 1}")

    raw_text = "".join(chunks)
    if not raw_text:
        return {**state, "status": "failed", "error": "No usable exhibit text after fetching"}

    report_call(
        f"  [fetch]  {len(raw_text):,} chars, {line_counter - 1:,} lines "
        f"({len(document_map)} exhibit(s))"
    )
    return {
        **state,
        "raw_text": raw_text,
        "document_map": document_map,
        "file_type": "pdf" if any_pdf else "html",
        "status": "fetched",
    }
