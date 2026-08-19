"""SEC EDGAR client — finds the most recent earnings press release for any public company.

Uses the EDGAR Submissions API (no API key required) to locate the latest 8-K
with Item 2.02 (Results of Operations), then retrieves the filing index to find
the Exhibit 99.1 press release document URL.

EDGAR rate-limit guideline: ≤10 requests/second.
"""
from __future__ import annotations

import logging
import os
import threading as _th
import time as _time
from datetime import date
from typing import Optional

import requests
from bs4 import BeautifulSoup

from earnings_agents.config import HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}"
_EDGAR_INDEX_HTML = _EDGAR_ARCHIVES_BASE + "/{acc}-index.htm"

# SEC requires a descriptive User-Agent with contact info for automated access
_HEADERS = {
    "User-Agent": "earning-agents data-pipeline@truegrids.com",
    "Accept-Encoding": "gzip, deflate",
}

_EX99_TYPES = frozenset({
    "EX-99.1", "EX-99", "EX99.1", "EX-99.01",
    "EX-99.2", "EX99.2", "EX-99.02",
    "EX-99.3", "EX99.3", "EX-99.03",
})


class _TokenBucket:
    """Thread-safe token-bucket rate limiter."""

    __slots__ = ("_rate", "_tokens", "_last", "_lock")

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = rate
        self._last = _time.monotonic()
        self._lock = _th.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = _time.monotonic()
                self._tokens = min(
                    self._rate,
                    self._tokens + (now - self._last) * self._rate,
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            _time.sleep(1.0 / self._rate)


# ≤ 8 req/s against sec.gov (SEC guideline: ≤ 10 req/s per user-agent)
_EDGAR_RATE_LIMITER = _TokenBucket(rate=float(os.getenv("EDGAR_RATE_LIMIT", "8")))

# HTTP status codes that warrant a retry (transient server-side errors).
_EDGAR_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_EDGAR_MAX_RETRIES: int = 3
_EDGAR_RETRY_BASE_DELAY: float = 1.0  # seconds; doubles on each retry


def _edgar_get(url: str, **kwargs) -> requests.Response:
    """Rate-limited GET with exponential-backoff retry for transient EDGAR errors.

    Retries up to ``_EDGAR_MAX_RETRIES`` times on HTTP 429/5xx or connection
    errors, re-acquiring the rate-limit token before each attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(_EDGAR_MAX_RETRIES + 1):
        _EDGAR_RATE_LIMITER.acquire()
        try:
            resp = requests.get(url, headers=_HEADERS, **kwargs)
            if resp.status_code not in _EDGAR_RETRY_STATUSES or attempt == _EDGAR_MAX_RETRIES:
                return resp
            delay = _EDGAR_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "EDGAR %s returned HTTP %d (attempt %d/%d); retrying in %.1f s",
                url, resp.status_code, attempt + 1, _EDGAR_MAX_RETRIES + 1, delay,
            )
            _time.sleep(delay)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == _EDGAR_MAX_RETRIES:
                raise
            delay = _EDGAR_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "EDGAR %s connection error (attempt %d/%d): %s; retrying in %.1f s",
                url, attempt + 1, _EDGAR_MAX_RETRIES + 1, exc, delay,
            )
            _time.sleep(delay)
    # Unreachable in normal operation; satisfies the type checker.
    raise requests.RequestException(f"_edgar_get exhausted retries for {url}") from last_exc


def _find_all_ex_99_exhibits(
    cik_int: str, acc: str, acc_nodash: str,
) -> list[dict]:
    """Parse the EDGAR HTML filing index to find ALL EX-99 exhibit documents.

    Returns a list of dicts in filing-index order —
    ``{exhibit: "EX-99.1", description: "The Press Release", url}`` —
    carrying the index's Description column so downstream agents know WHAT each
    exhibit is (press release, presentation, supplemental information).

    Returns an empty list when no exhibits are found.
    """
    index_url = _EDGAR_INDEX_HTML.format(cik_int=cik_int, acc_nodash=acc_nodash, acc=acc)
    try:
        resp = _edgar_get(index_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("EDGAR HTML index fetch failed for %s: %s", index_url, exc)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    exhibits: list[dict] = []

    # Filing index table: columns are Seq | Description | Document | Type | Size
    for row in soup.select("table.tableFile tr, table tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        # Type is in the 4th column (index 3); link is in the 3rd column (index 2)
        doc_type = cells[3].get_text(strip=True).upper()
        if doc_type in _EX99_TYPES:
            link_tag = cells[2].find("a", href=True)
            if link_tag:
                href: str = link_tag["href"]
                if href.startswith("/"):
                    href = f"https://www.sec.gov{href}"
                description = cells[1].get_text(" ", strip=True) if len(cells) > 1 else ""
                exhibits.append(
                    {"exhibit": doc_type, "description": description, "url": href}
                )
                logger.info(
                    "Found exhibit %s (%s) for %s/%s: %s",
                    doc_type, description[:60], cik_int, acc, href,
                )

    logger.info(
        "Found %d EX-99 exhibit(s) in index for %s/%s",
        len(exhibits), cik_int, acc,
    )
    return exhibits


def _find_all_ex_99_urls(cik_int: str, acc: str, acc_nodash: str) -> list[str]:
    """Return just the EX-99 exhibit URLs (compat wrapper)."""
    return [e["url"] for e in _find_all_ex_99_exhibits(cik_int, acc, acc_nodash)]


def normalize_cik(cik: str) -> str:
    """Return a zero-padded 10-digit CIK string."""
    return str(int(cik)).zfill(10)


def get_latest_earnings_url(
    cik: str,
) -> tuple[Optional[str], list[str], Optional[str], Optional[str], list[dict]]:
    """Return ``(filing_url, supplemental_urls, accession, filing_date,
    exhibits)`` for the most recent earnings press release.

    ``filing_url`` is the URL to the primary earnings release (Exhibit 99.1).
    ``supplemental_urls`` is a list of additional exhibit URLs (EX-99.2, EX-99.3,
    etc.) that contain supplemental financial data.
    ``accession`` / ``filing_date`` identify the filing and the
    period sanity window.
    ``exhibits`` is the full exhibit list in filing-index order:
    ``[{exhibit, description, url}]`` — e.g. EX-99.1 "The Press Release",
    EX-99.2 "The Presentation Materials", EX-99.3 "The Supplemental Information".

    Falls back to the most recent 8-K primary document if no Exhibit 99.1 is found.
    Returns ``(None, [], None, None, [])`` if no 8-K filing is available.
    """
    cik_padded = normalize_cik(cik)
    cik_int = str(int(cik_padded))  # no leading zeros for archive paths

    # ── 1. Fetch submissions ─────────────────────────────────────────────────
    sub_url = _EDGAR_SUBMISSIONS.format(cik=cik_padded)
    try:
        resp = _edgar_get(sub_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("EDGAR submissions fetch failed for CIK %s: %s", cik_padded, exc)
        return None, [], None, None, []

    recent = data.get("filings", {}).get("recent", {})
    forms: list[str] = recent.get("form", [])
    items_list: list[str] = recent.get("items", [])
    accessions: list[str] = recent.get("accessionNumber", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])
    filing_dates: list[str] = recent.get("filingDate", [])

    # ── 2. Find latest 8-K with Item 2.02 (earnings results) ─────────────────
    target_idx: Optional[int] = None
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        item_str = items_list[i] if i < len(items_list) else ""
        if "2.02" in item_str:
            target_idx = i
            break

    # Fallback: use first available 8-K of any item type
    if target_idx is None:
        for i, form in enumerate(forms):
            if form == "8-K":
                target_idx = i
                logger.info(
                    "No Item 2.02 8-K found for CIK %s — using first available 8-K",
                    cik_padded,
                )
                break

    if target_idx is None:
        logger.warning("No 8-K filings found for CIK %s", cik_padded)
        return None, [], None, None, []

    acc = accessions[target_idx]       # e.g. "0000320193-26-000011"
    acc_nodash = acc.replace("-", "")  # e.g. "000032019326000011"
    filing_date_str: Optional[str] = (
        filing_dates[target_idx] if target_idx < len(filing_dates) else None
    )

    # ── 3. Parse HTML filing index to find ALL EX-99 exhibits ────────────────
    exhibits = _find_all_ex_99_exhibits(cik_int, acc, acc_nodash)

    if exhibits:
        primary_url = exhibits[0]["url"]
        supplemental_urls = [e["url"] for e in exhibits[1:]]
        if supplemental_urls:
            logger.info(
                "Found %d supplemental exhibit(s) for CIK %s: %s",
                len(supplemental_urls), cik_padded, supplemental_urls,
            )
        return primary_url, supplemental_urls, acc, filing_date_str, exhibits

    # ── 4. Last resort: primary document from submissions metadata ────────────
    primary_doc = primary_docs[target_idx] if target_idx < len(primary_docs) else ""
    if primary_doc:
        url = f"{_EDGAR_ARCHIVES_BASE.format(cik_int=cik_int, acc_nodash=acc_nodash)}/{primary_doc}"
        logger.info("EDGAR primary doc fallback for CIK %s: %s", cik_padded, url)
        return url, [], acc, filing_date_str, []

    logger.warning("Could not resolve document URL for CIK %s accession %s", cik_padded, acc)
    return None, [], None, None, []

