#!/usr/bin/env python3
"""Simulate SEC RSS feed matching for a given ticker.

Shows exactly what the polling path (pollAllFeeds → matchEntriesWithLoadList)
would publish to Redis, and what the worker (get_latest_earnings_url +
_resolve_8k_skip_guard) would process.

Usage:
    uv run scripts/simulate_rss_match.py --ticker ABBV
    uv run scripts/simulate_rss_match.py --ticker ABBV --cik 0001551152
    uv run scripts/simulate_rss_match.py --ticker AAPL
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Optional

import requests

# ── Config ───────────────────────────────────────────────────────────────────
EDGAR_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=exclude&count=100&output=atom"
)
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}"
HEADERS = {
    "User-Agent": "earning-agents data-pipeline@truegrids.com",
    "Accept-Encoding": "gzip, deflate",
}
CIK_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

NS = {"atom": "http://www.w3.org/2005/Atom"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_ticker_cik_map() -> dict[str, str]:
    """Fetch SEC company_tickers.json → {TICKER: cik_padded}."""
    resp = requests.get(CIK_TICKER_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    out: dict[str, str] = {}
    for _k, v in data.items():
        ticker = v.get("ticker", "").upper()
        cik_str = str(v.get("cik_str", "")).zfill(10)
        if ticker and cik_str:
            out[ticker] = cik_str
    return out


def fetch_atom_entries() -> list[dict]:
    """Fetch and parse the 8-K Atom feed, return list of entry dicts."""
    resp = requests.get(EDGAR_ATOM_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    entries = []
    for entry_el in root.findall("atom:entry", NS):
        title = (entry_el.findtext("atom:title", "") or "").strip()
        cik_match = __import__("re").search(r"\((\d{1,10})\)", title)
        cik = (cik_match.group(1) if cik_match else "").zfill(10)

        summary = (entry_el.findtext("atom:summary", "") or "").strip()
        filed_match = __import__("re").search(r"Filed:\s*(\d{4}-\d{2}-\d{2})", summary)
        filing_date = filed_match.group(1) if filed_match else ""

        accno_match = __import__("re").search(r"AccNo:\s*(\d+-\d+-\d+)", summary)
        accession_number = accno_match.group(1) if accno_match else ""

        id_text = entry_el.findtext("atom:id", "") or ""
        accno_from_id = ""
        id_match = __import__("re").search(r"accession-number=([0-9-]+)", id_text)
        if id_match:
            accno_from_id = id_match.group(1)

        entries.append({
            "title": title,
            "cik": cik,
            "accession_number": accession_number or accno_from_id,
            "filing_date": filing_date,
            "company_name": title.split(" - ")[-1].split(" (")[0].strip() if " - " in title else title,
            "form_type": "8-K",
        })

    return entries


def fetch_submissions(cik: str) -> dict:
    """Fetch SEC submissions JSON for a CIK."""
    url = EDGAR_SUBMISSIONS.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def find_latest_8k_item202(submissions: dict) -> Optional[dict]:
    """Find the latest 8-K with Item 2.02 in the submissions data."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    items_list = recent.get("items", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        item_str = items_list[i] if i < len(items_list) else ""
        if "2.02" in item_str:
            return {
                "index": i,
                "accession": accessions[i] if i < len(accessions) else "",
                "filing_date": filing_dates[i] if i < len(filing_dates) else "",
                "report_date": report_dates[i] if i < len(report_dates) else "",
                "items": item_str,
            }
    return None


def fetch_ex99_url(cik_int: str, accession: str) -> Optional[str]:
    """Parse the filing index HTML to find the EX-99.1 URL."""
    acc_nodash = accession.replace("-", "")
    index_url = f"{EDGAR_ARCHIVES.format(cik_int=cik_int, acc_nodash=acc_nodash)}/{accession}-index.htm"
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    import re
    # Find EX-99.1 href
    for line in resp.text.split("\n"):
        if "EX-99.1" in line.upper() and "href=" in line:
            m = re.search(r'href="([^"]*)"', line)
            if m:
                href = m.group(1)
                if href.startswith("/"):
                    href = f"https://www.sec.gov{href}"
                return href
    return None


def extract_period_from_exhibit(url: str) -> Optional[str]:
    """Extract the first period-end date from an EX-99.1 HTML (mimics parse_period_end_date)."""
    import re
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        text = resp.text[:50000]
    except requests.RequestException:
        return None

    MONTH_RE = re.compile(
        r"(January|February|March|April|May|June|July|August|September|October|"
        r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[.]?\s+(\d{1,2}),?\s+(\d{4})",
        re.IGNORECASE,
    )

    m = MONTH_RE.search(text)
    if m:
        date_str = f"{m.group(1)} {m.group(2)} {m.group(3)}"
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(date_str, fmt).date().isoformat()
            except ValueError:
                continue
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate SEC RSS feed matching for a ticker"
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. ABBV)")
    parser.add_argument("--cik", help="Override CIK (auto-resolved from SEC if omitted)")
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()

    # ── Resolve CIK ───────────────────────────────────────────────────────────
    if args.cik:
        cik_padded = args.cik.strip().zfill(10)
    else:
        print("Fetching ticker→CIK map from SEC …", end=" ", flush=True)
        try:
            ticker_map = fetch_ticker_cik_map()
            cik_padded = ticker_map.get(ticker, "")
            if not cik_padded:
                print(f"\n✗ Ticker '{ticker}' not found in SEC company_tickers.json")
                sys.exit(1)
            print(f"→ {cik_padded}")
        except Exception as exc:
            print(f"\n✗ Failed: {exc}")
            sys.exit(1)

    cik_int = str(int(cik_padded))

    print(f"\n{'='*72}")
    print(f"  Simulating RSS feed match for {ticker}  (CIK {cik_padded})")
    print(f"{'='*72}")

    # ── Step 1: Fetch Atom feed (what pollAllFeeds does) ───────────────────────
    print("\n── 1. EDGAR Atom Feed (8-K) ──────────────────────────────────────")
    try:
        all_entries = fetch_atom_entries()
        print(f"   Fetched {len(all_entries)} entries from Atom feed")
    except Exception as exc:
        print(f"   ✗ Failed to fetch Atom feed: {exc}")
        sys.exit(1)

    # Filter entries by CIK (what pollAllFeeds does)
    matched_entries = [e for e in all_entries if e["cik"] == cik_padded]
    if not matched_entries:
        # Try without leading zeros
        matched_entries = [
            e for e in all_entries
            if e["cik"] == cik_int or str(int(e["cik"] or "0")) == cik_int
        ]

    print(f"   Matched {len(matched_entries)} entry(s) for CIK {cik_padded}:")

    # Sort by filing_date descending (what matchEntriesWithLoadList does)
    matched_entries.sort(key=lambda e: e["filing_date"], reverse=True)

    for i, entry in enumerate(matched_entries):
        flag = " ★ FIRST MATCH (would be queued)" if i == 0 else ""
        print(f"\n   [{i}] {entry['filing_date']}  {entry['accession_number']}{flag}")
        print(f"       Company: {entry['company_name']}")
        print(f"       Title:   {entry['title'][:80]}")

    if not matched_entries:
        print("   ⚠ No entries for this CIK in the current Atom feed!")
        print("   (The feed window may not include recent filings for this ticker)")

    # ── Step 2: Simulate what the worker does (get_latest_earnings_url) ───────
    print("\n── 2. SEC Submissions API (what worker calls) ─────────────────────")

    try:
        submissions = fetch_submissions(cik_padded)
        latest_8k = find_latest_8k_item202(submissions)
    except Exception as exc:
        print(f"   ✗ Failed to fetch submissions: {exc}")
        sys.exit(1)

    if latest_8k is None:
        print("   ⚠ No 8-K with Item 2.02 found in submissions!")
        sys.exit(1)

    print(f"   Latest 8-K Item 2.02:")
    print(f"     Accession:    {latest_8k['accession']}")
    print(f"     Filing Date:  {latest_8k['filing_date']}")
    print(f"     Report Date:  {latest_8k['report_date']}")
    print(f"     Items:        {latest_8k['items']}")

    # Check if Atom feed entry matches submissions entry
    atom_accessions = {e["accession_number"] for e in matched_entries}
    if latest_8k["accession"] in atom_accessions:
        print(f"\n   ✓ Atom feed HAS this accession — polling would queue the correct filing")
    else:
        print(f"\n   ⚠ Atom feed does NOT have this accession!")
        print(f"   Atom feed accessions: {sorted(atom_accessions)}")
        if matched_entries:
            latest_atom_acc = matched_entries[0]["accession_number"]
            print(f"   Atom feed's first ABBV entry: {latest_atom_acc}")
            print(f"   → Worker would still call submissions API and process: {latest_8k['accession']}")
            print(f"   → So worker processes the CORRECT (latest) filing regardless of Atom feed")

    # ── Step 3: Check what period the EX-99.1 would resolve to ────────────────
    print("\n── 3. EX-99.1 Period Detection ────────────────────────────────────")

    ex99_url = fetch_ex99_url(cik_int, latest_8k["accession"])
    if ex99_url:
        print(f"   EX-99.1 URL: {ex99_url}")
        period_date = extract_period_from_exhibit(ex99_url)
        if period_date:
            print(f"   Extracted period end date: {period_date}")
            # Determine quarter for Dec FY end
            d = date.fromisoformat(period_date)
            fiscal_year = d.year if d.month <= 12 else d.year + 1  # assume Dec FY
            quarter_map = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
            q = quarter_map.get(d.month, "?")
            print(f"   → FY{fiscal_year} Q{q}")
        else:
            print("   ⚠ Could not extract period date from EX-99.1")
    else:
        print("   ⚠ Could not find EX-99.1 URL in filing index")

    # ── Step 4: Summary ──────────────────────────────────────────────────────
    print(f"\n── 4. What would be published to Redis ───────────────────────────")

    if matched_entries:
        # Simulate the OLD publishFilings (full data)
        print("\n   OLD (full Atom feed data):")
        old_msg = {
            "event": "new_filing",
            "source": "sec_rss",
            "filing_type": "8-K",
            "cik": matched_entries[0]["cik"],
            "ticker": ticker,
            "company_name": matched_entries[0]["company_name"],
            "filing_date": matched_entries[0]["filing_date"],
            "load_request_id": "<pending_request_id>",
            "queued_at": datetime.now().isoformat(),
        }
        print(f"   {json.dumps(old_msg, indent=6)}")

        # Simulate the NEW publishFilings (minimal, matching manual trigger)
        print("\n   NEW (minimal — same as manual trigger):")
        new_msg = {
            "filing_type": "8-K",
            "ticker": ticker,
            "load_request_id": "<pending_request_id>",
            "queued_at": datetime.now().isoformat(),
        }
        print(f"   {json.dumps(new_msg, indent=6)}")

    # Final verdict
    print(f"\n{'='*72}")
    if latest_8k and period_date:
        print(f"  Worker will process: {latest_8k['accession']}  →  period {period_date}")
        print(f"  Atom feed entry:     {matched_entries[0]['accession_number'] if matched_entries else 'N/A'}")
        if matched_entries and latest_8k["accession"] != matched_entries[0]["accession_number"]:
            print(f"  ⚠ MISMATCH! Atom feed has a different (older) accession than submissions API")
        else:
            print(f"  ✓ Accessions match — both paths resolve the same filing")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
