# Earnings Extraction Pipeline

Agent-based pipeline that extracts quarterly and annual financial metrics (income statement, balance sheet, cash flow, forward guidance) from SEC 8-K and 6-K filings into MongoDB `normalize_data`.

---

## Quick Command Reference

### 1. Setup & Testing

```bash
uv sync                                    # Install dependencies
uv sync --extra dev                        # Install dev dependencies (pytest)
uv run pytest tests/ -q                    # Run test suite
```

---

### 2. Live Filing Extraction (`earnings`)

Extracts the latest earnings release (or a pinned accession/URL). By default, updates and overwrites existing records for that period.

```bash
# US Issuers (Form 8-K is default)
uv run earnings --ticker MSFT                               # Full extraction (income, balancesheet, cashflow)
uv run earnings --ticker MSFT -s income                     # Specific statement only
uv run earnings --ticker MSFT --dry-run                     # Test connectivity & fetch without LLM calls
uv run earnings --ticker MSFT -v                            # Debug verbose logging

# Foreign Issuers (Form 6-K required)
uv run earnings --ticker ASML --form 6-K                    # Discover & extract latest 6-K earnings
uv run earnings --ticker ASML --form 6-K --quarterly-only   # Skip annual filing + skip Q4 derivation

# Pin Specific Filing Accession or Document URL
uv run earnings --ticker ASML --accession 0001628280-26-048235
uv run earnings --ticker TSM --form 6-K --accession 0001046179-26-000451
uv run earnings --ticker MSFT --filing-url https://example.com/release.htm
```

---

### 3. Historical Backfills & Specific Quarters (`earnings-backfill`)

Discovers filings from EDGAR metadata. **Resumes by default** (skips already-saved accessions to conserve LLM tokens). Use `--reload` to force re-extraction.

```bash
# Specific Quarter & Year Selection
uv run earnings-backfill --ticker MSFT --fiscal-year 2024 --fiscal-quarter 1   # Q1 only
uv run earnings-backfill --ticker MSFT --fiscal-year 2024 --fiscal-quarter 2   # Q2 only
uv run earnings-backfill --ticker MSFT --fiscal-year 2024 --fiscal-quarter 3   # Q3 only
uv run earnings-backfill --ticker MSFT --fiscal-year 2024 --fiscal-quarter 4   # Q4 / Annual
uv run earnings-backfill --ticker MSFT --fiscal-year 2024                       # All periods in FY2024

# Year Range
uv run earnings-backfill --ticker ASML --form 6-K --from-year 2023 --to-year 2025

# Dry-run Manifests (preview matching filings without running extraction)
uv run earnings-backfill --ticker MSFT --fiscal-year 2024 --dry-run
uv run earnings-backfill --ticker MSFT --fiscal-year 2024 --dry-run --json

# Resuming vs. Force Reload
uv run earnings-backfill --ticker MSFT --fiscal-year 2024                     # Resumes (skips saved accessions)
uv run earnings-backfill --ticker MSFT --fiscal-year 2024 --reload            # Force re-run & overwrite saved data
uv run earnings-backfill --ticker MSFT --fiscal-year 2024 --no-resume         # Process all candidates ignoring saved state

# Pinned Accession Backfill
uv run earnings-backfill --ticker TSM --form 6-K --accession 0001046179-26-000451
```

---

### 4. Background Redis Worker (`earnings-8k-worker`)

Runs the continuous queue-driven pipeline worker.

```bash
# Run worker locally
uv run earnings-8k-worker

# Run worker with attempt limit
uv run earnings-8k-worker --max-attempts 3

# Run worker via Docker Compose
docker compose up -d --build              # Build and start background worker
docker compose restart worker-8k          # Restart after code changes (src is volume-mounted)
docker compose logs -f worker-8k          # Stream worker logs
```

---

### 5. Failure Inspection & Utilities

```bash
# Browse failed or degraded runs in MongoDB
uv run earnings-failures

# Backfill binary period_type on older guidance records
uv run python scripts/backfill_guidance_period_type.py

# Read-only staging pilot preflight test for 6-K filings
STAGING_MONGODB_URI=... STAGING_NORMALIZE_DB_NAME=normalize_data_staging \
  scripts/phase6_6k_staging_pilot.sh
```
