"""SEC EDGAR client — find a company's most recent earnings press release.

Uses the EDGAR Submissions API (no API key required) to locate an earnings
filing, then retrieves the filing index to find its Exhibit 99.x documents:

* ``8-K`` — US issuers, preferably Item 2.02 (Results of Operations).
* ``6-K`` — foreign private issuers (ASML, TSM, ...).  A 6-K is a
  general-purpose "furnish" form, so auto-discovery only accepts a 6-K whose
  filing-index exhibit metadata looks like an earnings release; the reliable
  path is a pinned accession (RSS poller / admin trigger / manual URL).

EDGAR rate-limit guideline: ≤10 requests/second.
"""
from __future__ import annotations

import logging
import os
import re
import threading as _th
import time as _time
from datetime import date
from typing import Optional

import requests
from bs4 import BeautifulSoup

from earnings_agents.config import HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
# Older filings live in per-CIK pagination files referenced by
# ``filings.files[].name`` (e.g. CIK0000320193-submissions-001.json).
_EDGAR_SUBMISSIONS_PAGE = "https://data.sec.gov/submissions/{name}"
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

# Source forms this pipeline can ingest for an earnings release: 8-K Item 2.02
# (US issuers) and 6-K (foreign private issuers, e.g. ASML / TSM).  A 6-K is a
# general-purpose "furnish" form, so it is never selected blindly — see
# ``get_latest_filing_url``.
EARNINGS_FORMS: tuple[str, ...] = ("8-K", "6-K")

# Filing-index metadata hints that a 6-K actually carries an earnings release.
# Used ONLY to choose which 6-K to fetch (filing selection), never to navigate
# or extract from the document.
# Deliberately specific: bare "annual" would match "annual general meeting"
# and "financial" would match "financial calendar", both of which are NOT
# earnings releases.  Evaluated against the EX-99 description + filename.
_6K_EARNINGS_HINT_RX = re.compile(
    r"earnings|\bresults\b|press\s+release"
    r"|interim\s+(?:financial|report|statements?)"
    r"|quarter(?:ly|s)?\s+(?:financial|results|report)"
    r"|half[-\s]?year\s+(?:financial|results|report)"
    # Periodic-report phrasing: "Q2 report", "Q2 2026 report", "2Q report",
    # "second quarter report".
    r"|q\s*[1-4][^a-z]{0,12}report"
    r"|(?:first|second|third|fourth)\s+quarter\s+report"
    r"|net\s+(?:sales|revenue|income|profit)|\brevenues?\b|\bprofit\b",
    re.IGNORECASE,
)

# EDGAR filenames often concatenate words without separators (ASML's earnings
# exhibit is ``pressreleasefinancialresul.htm``, Ericsson's is
# ``a26q2report.htm``), which the word-bounded regex above cannot see.  These
# compact stems are checked against the haystack with all non-alphanumerics
# removed.  Deliberately narrow: "agmresults" must not match, so bare
# "results" is NOT a stem.
_6K_COMPACT_STEMS = (
    "earnings",
    "financialresults",
    "quarterlyresults",
    "interimfinancial",
    "financialstatements",
    "financialreport",
    "earningsreport",
    "interimreport",
    "quarterlyreport",
    "halfyearreport",
    "sixmonthreport",
    "ninemonthreport",
    # Periodic-report filenames: "a26q2report.htm", "q3-report", "reportq1".
    "q1report",
    "q2report",
    "q3report",
    "q4report",
    "reportq1",
    "reportq2",
    "reportq3",
    "reportq4",
)
_6K_PRESS_RELEASE_SUBJECTS = (
    "financial", "result", "earning", "quarter", "interim",
)


def _looks_like_earnings_6k(haystack: str) -> bool:
    """True when 6-K exhibit metadata suggests an earnings release.

    Metadata-only filing SELECTION (never document navigation or extraction).
    """
    if _6K_EARNINGS_HINT_RX.search(haystack):
        return True
    compact = re.sub(r"[^a-z0-9]+", "", haystack.lower())
    if any(stem in compact for stem in _6K_COMPACT_STEMS):
        return True
    return "pressrelease" in compact and any(
        subject in compact for subject in _6K_PRESS_RELEASE_SUBJECTS
    )


# Types EDGAR uses for a filing's PRIMARY document (``Type`` column).  The
# pipeline previously recognized only "8-K", which left every 6-K without an
# EX-99 exhibit (e.g. TSM's monthly-revenue 6-Ks) looking like an empty index.
_PRIMARY_DOC_TYPES = frozenset({
    "8-K", "8-K/A", "6-K", "6-K/A",
    "10-K", "10-K/A", "10-Q", "10-Q/A",
    "20-F", "20-F/A", "40-F", "40-F/A",
})

# Cap on how many recent 6-K filing indexes we inspect while looking for an
# earnings release (each costs one EDGAR index request).
_6K_CANDIDATE_LIMIT = 5

# A periodic report (interim/annual) carries full statements, so the filing
# payload is orders of magnitude larger than a notice-style 6-K (buyback
# update, corporate announcement).  Measured on Ericsson: reports 1.8-5.1 MB vs
# notices 19-49 KB.  ``size`` comes from the submissions API we already fetch,
# so this costs no extra request and reads no document content — the report
# lives in the 6-K BODY (primary document) for such filers, which carries no
# filename/description signal at all.
_6K_REPORT_BODY_MIN_BYTES = int(
    os.getenv("SIXK_REPORT_BODY_MIN_KB", "512")
) * 1024


def _looks_like_report_body_6k(size_bytes: int | None) -> bool:
    """True when a 6-K payload is large enough to be a full periodic report."""
    try:
        return int(size_bytes or 0) >= _6K_REPORT_BODY_MIN_BYTES
    except (TypeError, ValueError):
        return False



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


def _parse_filing_index(
    cik_int: str, acc: str, acc_nodash: str,
) -> tuple[list[dict], dict | None]:
    """Fetch a filing index and return ``(exhibits, primary_doc)``.

    ``exhibits`` — ALL EX-99 exhibit documents in filing-index order, each
    ``{exhibit: "EX-99.1", description: "The Press Release", url}`` (the
    Description column lets downstream agents know WHAT each exhibit is).

    ``primary_doc`` — the filing's primary document row (Type is the source
    form: ``8-K``, ``6-K``, ``10-Q``, ...) as a single
    ``{exhibit: <form>, description, url}`` dict, or ``None`` when absent.
    Inline-XBRL viewer links (``/ix?doc=...``) are unwrapped to the real
    document path.

    Returns ``([], None)`` when the index cannot be fetched/parsed.
    """
    index_url = _EDGAR_INDEX_HTML.format(cik_int=cik_int, acc_nodash=acc_nodash, acc=acc)
    try:
        resp = _edgar_get(index_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("EDGAR HTML index fetch failed for %s: %s", index_url, exc)
        return [], None

    soup = BeautifulSoup(resp.text, "lxml")
    exhibits: list[dict] = []
    primary_doc: dict | None = None

    # Filing index table: columns are Seq | Description | Document | Type | Size
    for row in soup.select("table.tableFile tr, table tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        # Type is in the 4th column (index 3); link is in the 3rd column (index 2)
        doc_type = cells[3].get_text(strip=True).upper()
        link_tag = cells[2].find("a", href=True)
        if not link_tag:
            continue
        href: str = link_tag["href"]
        if href.startswith("/ix?doc="):
            # Inline-XBRL viewer link — unwrap to the real document path.
            href = href.split("doc=", 1)[1]
        if href.startswith("/"):
            href = f"https://www.sec.gov{href}"
        description = cells[1].get_text(" ", strip=True) if len(cells) > 1 else ""
        if doc_type in _EX99_TYPES:
            exhibits.append(
                {"exhibit": doc_type, "description": description, "url": href}
            )
            logger.info(
                "Found exhibit %s (%s) for %s/%s: %s",
                doc_type, description[:60], cik_int, acc, href,
            )
        elif doc_type in _PRIMARY_DOC_TYPES and primary_doc is None:
            # Any supported source form (8-K / 6-K / 10-Q / ...) — not just
            # 8-K, so a pinned 6-K filing with no EX-99 exhibit still resolves
            # its primary document instead of looking empty.
            primary_doc = {"exhibit": doc_type, "description": description, "url": href}

    logger.info(
        "Found %d EX-99 exhibit(s) in index for %s/%s",
        len(exhibits), cik_int, acc,
    )
    return exhibits, primary_doc


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
    exhibits, _ = _parse_filing_index(cik_int, acc, acc_nodash)
    return exhibits


def _find_all_ex_99_urls(cik_int: str, acc: str, acc_nodash: str) -> list[str]:
    """Return just the EX-99 exhibit URLs (compat wrapper)."""
    return [e["url"] for e in _find_all_ex_99_exhibits(cik_int, acc, acc_nodash)]


def normalize_cik(cik: str) -> str:
    """Return a zero-padded 10-digit CIK string."""
    return str(int(cik)).zfill(10)


def get_exhibits_for_accession(cik: str, acc: str) -> list[dict]:
    """Resolve the exhibit documents for ONE specific filing accession.

    Unlike :func:`get_latest_earnings_url` — which scans the EDGAR submissions
    API for "the most recent Item 2.02 8-K" (subject to submissions-API lag and
    empty ``items`` metadata on brand-new filings) — this resolves the exhibits
    from the filing index of the exact accession the caller already knows about
    (e.g. the RSS-feed entry the poller queued).  Deterministic, no
    submissions-API race: the reporting period is then decided by the period
    agent after the pinned documents are fetched.

    Returns EX-99 exhibits in filing-index order
    ``[{exhibit, description, url}]``; falls back to the filing's primary
    document when the filing has no EX-99 exhibit.  Returns ``[]`` when the
    index cannot be fetched or neither an EX-99 nor a primary document is
    found (callers may fall back to the latest-filing scan).
    """
    cik_padded = normalize_cik(cik)
    cik_int = str(int(cik_padded))
    acc_nodash = acc.replace("-", "")
    exhibits, primary_doc = _parse_filing_index(cik_int, acc, acc_nodash)
    if exhibits:
        return exhibits
    if primary_doc:
        logger.info(
            "No EX-99 exhibits for %s/%s — using primary document %s",
            cik_int, acc, primary_doc["url"],
        )
        return [primary_doc]
    return []


def _fetch_submissions(cik_padded: str) -> Optional[dict]:
    """Fetch the submissions index (``filings.recent`` + ``filings.files``)."""
    try:
        resp = _edgar_get(
            _EDGAR_SUBMISSIONS.format(cik=cik_padded), timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.error("EDGAR submissions fetch failed for CIK %s: %s", cik_padded, exc)
        return None


def _fetch_submission_page(name: str) -> Optional[dict]:
    """Fetch one historical submissions page referenced by ``filings.files``."""
    try:
        resp = _edgar_get(
            _EDGAR_SUBMISSIONS_PAGE.format(name=name), timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.warning("EDGAR submissions page %s failed: %s", name, exc)
        return None


def get_company_filings(
    cik: str,
    *,
    forms: tuple[str, ...] = EARNINGS_FORMS,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[dict]:
    """Return a company's filings of *forms* within an optional year range.

    Reads ``filings.recent`` AND every historical pagination page
    (``filings.files[]``), so year ranges older than the submissions window
    are reachable.  Pages entirely older than *start_year* are skipped without
    an HTTP request.

    Each entry: ``{form, accession, filing_date, report_date,
    primary_document, items}``, newest first.  The year range filters on
    ``filingDate`` — it selects WHICH filings to consider and never decides a
    reporting period (the period agent does).
    """
    cik_padded = normalize_cik(cik)
    wanted = {str(f).strip().upper() for f in forms if str(f or "").strip()}
    if not wanted:
        return []

    data = _fetch_submissions(cik_padded)
    if data is None:
        return []

    entries: list[dict] = []

    def _collect(page: dict) -> None:
        form_list = page.get("form") or []
        items = page.get("items") or []
        accessions = page.get("accessionNumber") or []
        filing_dates = page.get("filingDate") or []
        report_dates = page.get("reportDate") or []
        primary_docs = page.get("primaryDocument") or []
        sizes = page.get("size") or []
        for i, form in enumerate(form_list):
            if str(form).strip().upper() not in wanted:
                continue
            filing_date = str(filing_dates[i] if i < len(filing_dates) else "") or ""
            year = int(filing_date[:4]) if filing_date[:4].isdigit() else None
            if start_year is not None and (year is None or year < start_year):
                continue
            if end_year is not None and (year is None or year > end_year):
                continue
            entries.append({
                "form": str(form).strip().upper(),
                "accession": str(accessions[i] if i < len(accessions) else "") or "",
                "filing_date": filing_date,
                "report_date": str(report_dates[i] if i < len(report_dates) else "") or "",
                "primary_document": str(primary_docs[i] if i < len(primary_docs) else "") or "",
                "items": str(items[i] if i < len(items) else "") or "",
                # Filing payload size (bytes) — used to spot report-body-only
                # 6-Ks that carry no EX-99 exhibit and no descriptive filename.
                "size": int(sizes[i]) if i < len(sizes) and str(sizes[i]).isdigit() else 0,
            })

    _collect((data.get("filings") or {}).get("recent") or {})

    for info in (data.get("filings") or {}).get("files") or []:
        name = info.get("name")
        if not name:
            continue
        # Skip pages that end before the requested range (filingTo is the
        # newest filing date in the page).
        filing_to = str(info.get("filingTo") or "")
        if start_year is not None and filing_to[:4].isdigit() and int(filing_to[:4]) < start_year:
            continue
        page = _fetch_submission_page(name)
        if page:
            _collect(page)

    entries.sort(
        key=lambda e: (e.get("filing_date") or "", e.get("accession") or ""),
        reverse=True,
    )
    logger.info(
        "EDGAR: %d %s filing(s) for CIK %s (years %s-%s)",
        len(entries), "/".join(sorted(wanted)), cik_padded,
        start_year if start_year is not None else "any",
        end_year if end_year is not None else "any",
    )
    return entries


def _quarter_of_month(month: int) -> int:
    return (max(1, min(12, month)) - 1) // 3 + 1


def fiscal_year_of(filing: dict) -> int | None:
    """Return a filing's FISCAL year (the ``reportDate`` year when present).

    SEC ``reportDate`` is the period end the filing reports on, which is what
    users mean by "fiscal year".  Some forms (notably many 8-Ks) leave it
    empty, so the filing year is used as a fallback — never as period identity;
    the period agent still decides the reporting period from the document.
    """
    report_date = str(filing.get("report_date") or "")
    if report_date[:4].isdigit():
        return int(report_date[:4])
    filing_date = str(filing.get("filing_date") or "")
    if filing_date[:4].isdigit():
        return int(filing_date[:4])
    return None


def build_earnings_manifest(
    cik: str,
    *,
    forms: tuple[str, ...] = ("8-K",),
    start_year: int | None = None,
    end_year: int | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
    limit: int | None = None,
    verify_exhibits: bool = True,
    include_low_confidence: bool = False,
    max_exhibit_checks: int = 60,
) -> list[dict]:
    """Build a reviewable earnings-filing manifest for a historical range.

    Each returned entry adds ``confidence`` (``high``/``low``), ``reason`` and
    the resolved ``fiscal_year``, so a backfill can be PREVIEWED before any
    LLM work.

    **Fiscal-year selection.**  *start_year* / *end_year* bound the FISCAL year
    of each filing — taken from the SEC ``reportDate`` (the period the filing
    reports on), falling back to the filing year only when ``reportDate`` is
    absent.  A fiscal year N report may be filed in N or N+1, so the EDGAR
    fetch window is widened by one year on each side and the fiscal-year test
    is applied to the fetched records.  This selects WHICH filings to consider;
    it never sets the reporting period (the period agent does).

    *fiscal_year* / *fiscal_quarter* narrow further to an exact fiscal period
    (``fiscal_quarter`` requires a ``reportDate`` month).

    * 8-K with Item 2.02 → high confidence.  Other 8-Ks are only included when
      *include_low_confidence* is set (most non-2.02 8-Ks are not earnings).
    * 6-K → high confidence only when its filing index has an EX-99 exhibit
      whose metadata looks like an earnings release (metadata-only selection).
      6-Ks without EX-99 exhibits are excluded — that is the safe failure mode.

    Results stay NEWEST-first (the EDGAR ordering) so ``--latest`` is the
    newest match; the caller decides processing order.
    """
    # Widen the EDGAR filing-date window: fiscal year N can be filed in N, N+1
    # (or, for late/amended filings, N+2).
    pre_start = (start_year - 1) if start_year is not None else None
    pre_end = (end_year + 1) if end_year is not None else None
    filings = get_company_filings(
        cik, forms=forms, start_year=pre_start, end_year=pre_end,
    )
    cik_padded = normalize_cik(cik)
    cik_int = str(int(cik_padded))
    out: list[dict] = []
    exhibit_checks = 0

    for filing in filings:
        form = filing["form"]
        report_date = filing.get("report_date") or ""
        filing_fiscal_year = fiscal_year_of(filing)
        if start_year is not None and (
            filing_fiscal_year is None or filing_fiscal_year < start_year
        ):
            continue
        if end_year is not None and (
            filing_fiscal_year is None or filing_fiscal_year > end_year
        ):
            continue
        if fiscal_year is not None and filing_fiscal_year != fiscal_year:
            continue
        if fiscal_quarter is not None:
            if not report_date[5:7].isdigit():
                continue
            if _quarter_of_month(int(report_date[5:7])) != fiscal_quarter:
                continue

        if form == "8-K":
            if "2.02" in (filing.get("items") or ""):
                confidence, reason = "high", "item_2.02"
            elif include_low_confidence:
                confidence, reason = "low", "8k_without_item_2.02"
            else:
                continue
        elif form == "6-K":
            if not verify_exhibits:
                if not include_low_confidence:
                    continue
                confidence, reason = "low", "unverified_6k"
            else:
                if exhibit_checks >= max_exhibit_checks:
                    logger.warning(
                        "6-K exhibit-verification cap (%d) reached for CIK %s — "
                        "remaining 6-Ks are not manifest-verified",
                        max_exhibit_checks, cik_padded,
                    )
                    break
                exhibit_checks += 1
                acc = filing["accession"]
                exhibits, primary_doc = _parse_filing_index(
                    cik_int, acc, acc.replace("-", "")
                )
                haystack = " ".join(
                    f"{e.get('description') or ''} {e.get('url') or ''}"
                    for e in exhibits
                )
                if exhibits and _looks_like_earnings_6k(haystack):
                    confidence, reason = "high", "earnings_exhibit_metadata"
                elif _looks_like_report_body_6k(filing.get("size")):
                    # Report-body-only filer (Ericsson): no EX-99 exhibit and an
                    # opaque primary filename, but the payload is a full periodic
                    # report (orders of magnitude larger than a notice 6-K).
                    # The period agent still classifies it; quarterly-only mode
                    # skips annual ones and non-reports fail period detection.
                    confidence, reason = "high", "large_6k_report_body"
                elif include_low_confidence and exhibits:
                    confidence, reason = "low", "6k_with_exhibits"
                elif include_low_confidence and primary_doc:
                    confidence, reason = "low", "6k_primary_document_only"
                else:
                    continue
        else:
            continue

        entry = dict(filing)
        entry.update({
            "confidence": confidence,
            "reason": reason,
            "fiscal_year": filing_fiscal_year,
        })
        out.append(entry)

    return out[:limit] if limit else out


# SEC archive paths embed the zero-padded accession:
#   /Archives/edgar/data/<cik>/<18-digit-accession>/<file>
# Manual links sometimes carry the dashed form in the filename instead.
_ACCESSION_IN_URL_RX = re.compile(
    r"(?<![\d-])(\d{10}-\d{2}-\d{6})(?![\d-])"      # dashed form
    r"|/(\d{18})(?=[/.])"                               # compact/no-dash form
)


def accession_from_url(url: str | None) -> Optional[str]:
    """Return the SEC accession embedded in *url* (dashed form), or None.

    EDGAR archive URLs look like
    ``https://www.sec.gov/Archives/edgar/data/937966/000162828026048235/press.htm``
    and manually-copied links often keep the dashed accession in the filename
    (``0000937966-26-000012.pdf``).  Used to recover form-type provenance for
    manual-URL runs, where no accession is supplied by the caller.
    """
    if not url:
        return None
    text = str(url)
    m = _ACCESSION_IN_URL_RX.search(text)
    if not m:
        return None
    if m.group(1):
        return m.group(1)
    digits = m.group(2)
    return f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"


def get_filing_form_type(cik: str, accession: str) -> Optional[str]:
    """Best-effort lookup of the SEC form type for a known accession.

    Reads the submissions API's recent filings only; an older accession that
    is not in ``recent`` returns None (callers treat the form as unknown).
    Never raises — this is traceability metadata, not a control flow input.
    """
    if not cik or not accession:
        return None
    cik_padded = normalize_cik(cik)
    try:
        resp = _edgar_get(
            _EDGAR_SUBMISSIONS.format(cik=cik_padded), timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — form lookup is informational
        logger.debug("Could not resolve form type for %s/%s: %s", cik, accession, exc)
        return None
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    for i, acc in enumerate(accessions):
        if acc == accession and i < len(forms):
            return str(forms[i])
    return None


def get_latest_filing_url(
    cik: str,
    forms: tuple[str, ...] = EARNINGS_FORMS,
) -> tuple[Optional[str], list[str], Optional[str], list[dict], Optional[str]]:
    """Return ``(filing_url, supplemental_urls, accession, exhibits, form)``.

    *forms* is the set of source forms to consider.  8-K is preferred (Item
    2.02 first, then any 8-K).  6-K candidates are accepted only when their
    filing-index exhibit metadata suggests an earnings release — a foreign
    private issuer files many unrelated 6-Ks, so a blind "latest 6-K" would
    silently feed the wrong document into the earnings pipeline.  If no 6-K
    candidate qualifies, nothing is returned (use a pinned accession instead).

    Reporting-period identity is never inferred here; the period agent decides
    it after the document is read.
    """
    cik_padded = normalize_cik(cik)
    cik_int = str(int(cik_padded))  # no leading zeros for archive paths
    wanted = tuple(
        str(f).strip().upper() for f in forms if str(f or "").strip()
    )

    sub_url = _EDGAR_SUBMISSIONS.format(cik=cik_padded)
    try:
        resp = _edgar_get(sub_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("EDGAR submissions fetch failed for CIK %s: %s", cik_padded, exc)
        return None, [], None, [], None

    recent = data.get("filings", {}).get("recent", {})
    form_list: list[str] = recent.get("form", [])
    items_list: list[str] = recent.get("items", [])
    accessions: list[str] = recent.get("accessionNumber", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])
    sizes: list = recent.get("size", [])

    def _resolve(acc: str, primary_doc: str, form_name: str):
        acc_nodash = acc.replace("-", "")
        exhibits = _find_all_ex_99_exhibits(cik_int, acc, acc_nodash)
        if exhibits:
            primary_url = exhibits[0]["url"]
            supplemental_urls = [e["url"] for e in exhibits[1:]]
            if supplemental_urls:
                logger.info(
                    "Found %d supplemental exhibit(s) for CIK %s: %s",
                    len(supplemental_urls), cik_padded, supplemental_urls,
                )
            return primary_url, supplemental_urls, acc, exhibits, form_name
        if primary_doc:
            url = (
                f"{_EDGAR_ARCHIVES_BASE.format(cik_int=cik_int, acc_nodash=acc_nodash)}"
                f"/{primary_doc}"
            )
            logger.info("EDGAR primary doc fallback for CIK %s: %s", cik_padded, url)
            return url, [], acc, [], form_name
        logger.warning(
            "Could not resolve document URL for CIK %s accession %s", cik_padded, acc
        )
        return None, [], None, [], form_name

    # ── 1. Preferred: 8-K Item 2.02, then any 8-K (unchanged behavior) ────
    if "8-K" in wanted:
        target_idx: Optional[int] = None
        for i, form in enumerate(form_list):
            if form != "8-K":
                continue
            item_str = items_list[i] if i < len(items_list) else ""
            if "2.02" in item_str:
                target_idx = i
                break
        if target_idx is None:
            for i, form in enumerate(form_list):
                if form == "8-K":
                    target_idx = i
                    logger.info(
                        "No Item 2.02 8-K found for CIK %s — using first available 8-K",
                        cik_padded,
                    )
                    break
        if target_idx is not None:
            acc = accessions[target_idx]
            primary_doc = primary_docs[target_idx] if target_idx < len(primary_docs) else ""
            return _resolve(acc, primary_doc, "8-K")

    # ── 2. Fallback: a 6-K whose index metadata looks like earnings ───────
    if "6-K" in wanted:
        considered = 0
        for i, form in enumerate(form_list):
            if form != "6-K":
                continue
            if considered >= _6K_CANDIDATE_LIMIT:
                break
            considered += 1
            acc = accessions[i]
            acc_nodash = acc.replace("-", "")
            exhibits = _find_all_ex_99_exhibits(cik_int, acc, acc_nodash)
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""
            if exhibits:
                haystack = " ".join(
                    f"{e.get('description') or ''} {e.get('url') or ''}"
                    for e in exhibits
                )
                if not _looks_like_earnings_6k(haystack):
                    logger.info(
                        "6-K %s for CIK %s has EX-99 exhibits but no earnings-like "
                        "description/filename — skipping candidate",
                        acc, cik_padded,
                    )
                    continue
            elif _looks_like_report_body_6k(
                sizes[i] if i < len(sizes) else None
            ):
                logger.info(
                    "6-K %s for CIK %s has no EX-99 exhibit but a report-sized "
                    "payload — using its primary document",
                    acc, cik_padded,
                )
            else:
                continue
            logger.info(
                "Selected 6-K %s for CIK %s (earnings-like exhibit metadata)",
                acc, cik_padded,
            )
            return _resolve(acc, primary_doc, "6-K")
        logger.warning(
            "No earnings-looking 6-K found among the %d most recent 6-K filings "
            "for CIK %s — use a pinned accession or a manual filing URL",
            considered, cik_padded,
        )

    if "8-K" not in wanted and "6-K" not in wanted:
        logger.warning("No supported earnings form requested for CIK %s", cik_padded)
    elif "8-K" in wanted and "6-K" not in wanted:
        logger.warning("No 8-K filings found for CIK %s", cik_padded)
    return None, [], None, [], None


def get_latest_earnings_url(
    cik: str,
) -> tuple[Optional[str], list[str], Optional[str], list[dict]]:
    """Return ``(filing_url, supplemental_urls, accession, exhibits)`` for
    the most recent earnings press release.

    ``filing_url`` is the URL to the primary earnings release (Exhibit 99.1).
    ``supplemental_urls`` is a list of additional exhibit URLs (EX-99.2, EX-99.3,
    etc.) that contain supplemental financial data.
    ``accession`` identifies the filing for traceability.  Reporting-period
    identity is determined only by the period agent after the document is read.
    ``exhibits`` is the full exhibit list in filing-index order:
    ``[{exhibit, description, url}]`` — e.g. EX-99.1 "The Press Release",
    EX-99.2 "The Presentation Materials", EX-99.3 "The Supplemental Information".

    Falls back to the most recent 8-K primary document if no Exhibit 99.1 is found.
    Returns ``(None, [], None, [])`` if no 8-K filing is available.

    Thin 8-K-only wrapper around :func:`get_latest_filing_url`; use that
    function directly for 6-K (foreign private issuer) filings.
    """
    filing_url, supplemental, accession, exhibits, _form = get_latest_filing_url(
        cik, forms=("8-K",),
    )
    return filing_url, supplemental, accession, exhibits

