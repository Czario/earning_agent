One filing: 8-K or 6-K

 ### Existing US issuer flow — 8-K is the default

 ```bash
   uv run earnings --ticker MSFT
   uv run earnings --ticker MSFT -s income
   uv run earnings --ticker MSFT --dry-run
 ```

 ### Foreign issuer: 6-K auto-discovery

 ```bash
   uv run earnings --ticker ASML --form 6-K --dry-run
   uv run earnings --ticker ASML --form 6-K
 ```

 ASML’s descriptive exhibit filenames let the conservative discovery filter identify earnings releases. TSM
 auto-discovery may find no candidate because its earnings exhibit names are opaque; use a pinned accession
 instead.

 ### Pin the exact filing — recommended for TSM

 These were the filings used in our read-only Phase 4 validation; choose the accession you actually want to
 process:

 ```bash
   uv run earnings --ticker ASML --form 6-K \
     --accession 0001628280-26-048235 --dry-run

   uv run earnings --ticker ASML --form 6-K \
     --accession 0001628280-26-048235

   uv run earnings --ticker TSM --form 6-K \
     --accession 0001046179-26-000451 --dry-run

   uv run earnings --ticker TSM --form 6-K \
     --accession 0001046179-26-000451
 ```

 A pinned accession resolves that filing’s exhibits directly; it does not scan for the latest filing. For a
 one-off URL:

 ```bash
   uv run earnings --ticker ASML --form 6-K \
     --filing-url 'https://example.com/earnings-release.html' --dry-run
 ```

 ### Other useful options

 ```bash
   # Process selected financial statements only
   uv run earnings --ticker ASML --form 6-K -s income

   # Use a CIK instead of a ticker
   uv run earnings --cik 0000937966 --form 6-K --dry-run

   # DEBUG logging
   uv run earnings --ticker ASML --form 6-K -v
 ```

 If quarterly concepts don’t yet exist for the company, the first quarterly run bootstraps them from annual
 concepts. No annual values are copied; the values still come from the quarterly filing.

 Historical backfill

 ### Preview the candidate manifest first

 ```bash
   # ASML: historical 6-K candidates, no LLM calls or database writes
   uv run earnings-backfill --ticker ASML --form 6-K \
     --from-year 2025 --to-year 2026 --dry-run

   # Same preview as JSON
   uv run earnings-backfill --ticker ASML --form 6-K \
     --from-year 2025 --to-year 2026 --dry-run --json

   # TSM: review lower-confidence 6-K candidates too
   uv run earnings-backfill --ticker TSM --form 6-K \
     --from-year 2025 --to-year 2026 --include-unverified --dry-run --json
 ```

 The default manifest is conservative. For TSM, review the manifest and then run selected filings by pinned
 accession; don’t automatically process every unverified 6-K.

 ### Run a historical range

 After reviewing the dry run:

 ```bash
   uv run earnings-backfill --ticker ASML --form 6-K \
     --from-year 2025 --to-year 2026

   # 8-K historical range
   uv run earnings-backfill --ticker MSFT --from-year 2024 --to-year 2025

   # Multiple forms (repeat --form)
   uv run earnings-backfill --ticker MSFT \
     --form 8-K --form 6-K --from-year 2025 --to-year 2026
 ```

 Backfills run oldest first within the selected candidates. This lets Q1–Q3 actuals be present before an
 annual filing triggers Q4 calculation.

 ### Target a period or control reruns

 ```bash
   # Approximate selection from SEC reportDate; the period agent still decides period identity
   uv run earnings-backfill --ticker ASML --form 6-K \
     --fiscal-year 2026 --fiscal-quarter 2 --dry-run

   # Only the newest matching candidate
   uv run earnings-backfill --ticker ASML --form 6-K --latest --dry-run

   # Limit candidates
   uv run earnings-backfill --ticker ASML --form 6-K \
     --from-year 2025 --limit 5 --dry-run

   # Pin one or more accessions; use --form 6-K for these examples
   uv run earnings-backfill --ticker TSM --form 6-K \
     --accession 0001046179-26-000451 --dry-run

   # Re-run accessions already marked completed in the backfill ledger
   uv run earnings-backfill --ticker ASML --form 6-K \
     --from-year 2025 --reload

   # Ignore the resume ledger
   uv run earnings-backfill --ticker ASML --form 6-K \
     --from-year 2025 --no-resume
 ```