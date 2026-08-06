#!/usr/bin/env python3
"""Simulate a polling match for a given ticker — publishes the same minimal
Redis message that the NestJS poller would after matching an EDGAR Atom feed
entry to a pending load request.

Usage:
    uv run scripts/simulate_polling_match.py ABBV
    uv run scripts/simulate_polling_match.py ABBV --form-type 8-K
    uv run scripts/simulate_polling_match.py ABBV --dry-run     # just show what would be published
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

# Add src to path so we can import earnings_agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def resolve_cik(ticker: str) -> dict[str, Any] | None:
    """Look up CIK + company name from the live SEC EDGAR submissions API."""
    import requests

    # First try the local DB
    try:
        from earnings_agents.tools.normalize_data_client import get_company_by_ticker
        company = get_company_by_ticker(ticker)
        if company:
            return {
                "cik": company["cik"],
                "company_name": company.get("name", ticker),
            }
    except Exception:
        pass

    # Fallback: SEC company_tickers_exchange.json
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers_exchange.json",
            headers={"User-Agent": "earning-agents data-pipeline@truegrids.com"},
            timeout=10,
        )
        resp.raise_for_status()
        # Format: {"fields": [...], "data": [[cik, name, ticker, exchange], ...]}
        data = resp.json()
        for row in data.get("data", []):
            if row[2].upper() == ticker.upper():
                cik = str(row[0]).zfill(10)
                return {"cik": cik, "company_name": row[1]}
    except Exception as exc:
        print(f"[WARN] SEC ticker lookup failed: {exc}")

    return None


def fetch_atom_feed(form_type: str = "8-K") -> list[dict[str, Any]]:
    """Fetch the EDGAR Atom feed and parse entries (same as NestJS poller)."""
    import requests
    import re
    from xml.etree import ElementTree as ET

    url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcurrent&type={form_type}&dateb=&owner=exclude&count=100&output=atom"
    )
    print(f"[ATOM] Fetching {url}")
    resp = requests.get(
        url,
        headers={
            "User-Agent": "earning-agents data-pipeline@truegrids.com",
            "Accept": "application/atom+xml",
        },
        timeout=15,
    )
    resp.raise_for_status()

    # Parse Atom XML — extract entries like the NestJS XMLParser does
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entries: list[dict[str, Any]] = []

    for entry_el in root.findall("atom:entry", ns):
        try:
            title = (entry_el.findtext("atom:title", "", ns) or "").strip()
            summary = (entry_el.findtext("atom:summary", "", ns) or "").strip()
            entry_id = (entry_el.findtext("atom:id", "", ns) or "").strip()

            # Extract accession from id: "urn:tag:sec.gov,2008:accession-number=0001551152-26-000023"
            acc_match = re.search(r"accession-number=([0-9-]+)", entry_id)
            accession = acc_match.group(1) if acc_match else ""

            # Extract filing-type from category
            category = entry_el.find("atom:category", ns)
            form_type_from_feed = category.get("term", "") if category is not None else ""

            # Extract CIK from title: "8-K - AbbVie Inc. (0001551152) (Filer)"
            cik_match = re.search(r"\((\d{10})\)", title)
            cik = cik_match.group(1) if cik_match else ""

            # Extract company name from title
            name_match = re.match(r"^[^-]+-\s*(.+?)\s*\(\d{10}\)", title)
            company_name = name_match.group(1).strip() if name_match else ""

            # Extract filed date from summary.  EDGAR returns the summary with
            # type="html", so the text contains HTML tags like <b>Filed:</b>.
            # Strip tags first, then match the date.
            summary_plain = re.sub(r"<[^>]+>", " ", summary)
            summary_plain = re.sub(r"\s+", " ", summary_plain).strip()
            filed_match = re.search(r"Filed:\s*(\d{4}-\d{2}-\d{2})", summary_plain)
            filing_date = filed_match.group(1) if filed_match else ""

            # Extract items from summary
            items_match = re.findall(r"Item\s+([\d.]+):", summary_plain)
            items = ", ".join(items_match) if items_match else ""

            # Extract filing href from link
            link_el = entry_el.find("atom:link[@rel='alternate']", ns)
            filing_href = link_el.get("href", "") if link_el is not None else ""

            entries.append({
                "accession_number": accession,
                "form_type": form_type_from_feed,
                "filing_date": filing_date,
                "company_name": company_name,
                "cik": cik,
                "items": items,
                "filing_href": filing_href,
                "title": title,
            })
        except Exception as exc:
            print(f"[WARN] Failed to parse entry: {exc}")
            continue

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a polling match — trigger the 8-K worker the same way the NestJS poller would."
    )
    parser.add_argument("ticker", help="Stock ticker (e.g. ABBV)")
    parser.add_argument(
        "--form-type", default="8-K",
        help="Form type to match (default: 8-K)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be published without actually sending to Redis",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        help="Redis URL",
    )
    parser.add_argument(
        "--queue-name",
        default=os.getenv("REDIS_QUEUE_8K", "sec:filings:8k"),
        help="Redis queue name",
    )
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()

    # ── 1. Resolve CIK ──────────────────────────────────────────────────
    company = resolve_cik(ticker)
    if not company:
        print(f"[FAIL] Could not resolve CIK for ticker {ticker}")
        sys.exit(1)

    cik = company["cik"]
    company_name = company["company_name"]
    print(f"[INFO] {ticker} → CIK {cik} ({company_name})")

    # ── 2. Fetch Atom feed ──────────────────────────────────────────────
    print(f"[INFO] Fetching EDGAR Atom feed for {args.form_type}…")
    try:
        entries = fetch_atom_feed(args.form_type)
    except Exception as exc:
        print(f"[FAIL] Atom feed fetch failed: {exc}")
        sys.exit(1)

    print(f"[INFO] Feed returned {len(entries)} entries")

    # ── 3. Filter by CIK (exactly like pollAllFeeds) ────────────────────
    # Accept both padded (0001551152) and numeric (1551152) forms
    padded_cik = cik.zfill(10)
    numeric_cik = str(int(cik))

    matches = [
        e for e in entries
        if e["cik"] in (padded_cik, numeric_cik)
    ]
    print(f"[INFO] {len(matches)} entry(s) match CIK {cik}:")

    for i, m in enumerate(matches):
        print(f"  [{i}] {m['filing_date']}  {m['form_type']}  "
              f"{m['accession_number']}  Items: {m['items'] or '(none)'}")

    if not matches:
        print("[FAIL] No matching entries in the Atom feed — ticker may not have a recent 8-K")
        sys.exit(1)

    # ── 4. Pick the most recent match (same as matchEntriesWithLoadList sort) ──
    sorted_matches = sorted(
        matches,
        key=lambda e: e.get("filing_date") or "",
        reverse=True,
    )
    best = sorted_matches[0]
    print(f"\n[MATCH] Selected most recent: {best['filing_date']} "
          f"accession={best['accession_number']}")

    # ── 5. Build the minimal Redis message (same as Fix 1) ──────────────
    # This is the exact same payload the polling would now publish.
    message = {
        "filing_type": best["form_type"],
        "ticker": ticker,
        "load_request_id": None,  # No load request ID in standalone simulation
        "queued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Include extra context for debugging
    message["_debug"] = {
        "accession_number": best["accession_number"],
        "filing_date": best["filing_date"],
        "company_name": company_name,
        "cik": padded_cik,
        "simulated_polling_match": True,
    }

    print(f"\n[REDIS MESSAGE] (what polling publishes):")
    print(json.dumps(message, indent=2))

    if args.dry_run:
        print("\n[Dry-run] Not publishing to Redis.")
        return

    # ── 6. Publish to Redis ─────────────────────────────────────────────
    try:
        from redis import Redis
        client = Redis.from_url(args.redis_url, decode_responses=True)
        # Remove _debug before publishing to match production behaviour
        clean_message = {k: v for k, v in message.items() if k != "_debug"}
        count = client.rpush(args.queue_name, json.dumps(clean_message))
        print(f"\n[OK] Published to Redis queue '{args.queue_name}' ({count} item(s))")
        client.close()
    except Exception as exc:
        print(f"\n[FAIL] Redis publish failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
