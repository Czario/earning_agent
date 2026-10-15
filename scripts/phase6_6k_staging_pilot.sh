#!/usr/bin/env bash
# Safe, opt-in Phase 6 6-K staging pilot.
# Default mode is read-only preflight + CLI dry-runs. Add --execute only after
# an isolated staging DB is seeded and inspected; PHASE6_CONFIRM=YES is required.
set -euo pipefail

MODE="${1:-preflight}"
if [[ "$MODE" != "preflight" && "$MODE" != "--execute" ]]; then
  echo "usage: $0 [preflight|--execute]" >&2
  exit 2
fi

: "${STAGING_MONGODB_URI:?Set STAGING_MONGODB_URI to an isolated staging MongoDB URI}"
: "${STAGING_NORMALIZE_DB_NAME:?Set STAGING_NORMALIZE_DB_NAME to a dedicated staging database name}"
STAGING_DB_LOWER="$(printf '%s' "$STAGING_NORMALIZE_DB_NAME" | tr '[:upper:]' '[:lower:]')"
if [[ "$STAGING_DB_LOWER" != *staging* ]]; then
  echo "Refusing to run: STAGING_NORMALIZE_DB_NAME must contain 'staging'" >&2
  exit 2
fi
# Read the normal configured URI (including .env) without printing it, and
# refuse an exact match so a staging pilot cannot accidentally target the
# configured production endpoint.
CONFIGURED_MONGODB_URI="$(uv run python - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
print(os.getenv("MONGODB_URI", ""))
PY
)"
if [[ -n "$CONFIGURED_MONGODB_URI" && "$STAGING_MONGODB_URI" == "$CONFIGURED_MONGODB_URI" ]]; then
  echo "Refusing to run: STAGING_MONGODB_URI matches the configured MONGODB_URI" >&2
  exit 2
fi
if [[ "$MODE" == "--execute" && "${PHASE6_CONFIRM:-}" != "YES" ]]; then
  echo "Refusing live staging writes: set PHASE6_CONFIRM=YES and pass --execute" >&2
  exit 2
fi

export MONGODB_URI="$STAGING_MONGODB_URI"
export NORMALIZE_DB_NAME="$STAGING_NORMALIZE_DB_NAME"
export DATABASE_NAME="$STAGING_NORMALIZE_DB_NAME"

ASML_TICKER="${ASML_TICKER:-ASML}"
ASML_CIK="${ASML_CIK:-0000937966}"
ASML_ACCESSION="${ASML_ACCESSION:-0001628280-26-048235}"
TSM_TICKER="${TSM_TICKER:-TSM}"
TSM_CIK="${TSM_CIK:-0001046179}"
TSM_ACCESSION="${TSM_ACCESSION:-0001046179-26-000451}"

# Verify the staging DB is reachable and contains the company + annual concepts
# needed for the period-1 quarterly concept bootstrap. No documents are written.
uv run python - <<'PY'
import os
from pymongo import MongoClient

uri = os.environ["MONGODB_URI"]
db_name = os.environ["NORMALIZE_DB_NAME"]
client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
try:
    client.admin.command("ping")
    db = client[db_name]
    failures = []
    for ticker, cik in ((os.getenv("ASML_TICKER", "ASML"), os.getenv("ASML_CIK", "0000937966")),
                        (os.getenv("TSM_TICKER", "TSM"), os.getenv("TSM_CIK", "0001046179"))):
        company = db.companies.find_one({"ticker_symbol": ticker}, {"_id": 0, "cik": 1, "name": 1})
        annual_count = db.normalized_concepts_annual.count_documents({"cik": cik})
        quarterly_count = db.normalized_concepts_quarterly.count_documents({"cik": cik})
        quarter_values = db.concept_values_quarterly.count_documents({"cik": cik})
        print(f"{ticker} CIK={cik}: company={'yes' if company else 'MISSING'}, "
              f"annual_concepts={annual_count}, quarterly_concepts={quarterly_count}, "
              f"quarterly_values={quarter_values}")
        if not company or annual_count == 0:
            failures.append(ticker)
    if failures:
        raise SystemExit(
            "Staging seed incomplete for: " + ", ".join(failures) +
            ". Copy the companies and normalized_concepts_annual rows into the "
            "isolated staging DB before running the extraction pilot."
        )
finally:
    client.close()
PY

# Resolve the exact pinned accessions and check the LLM/Mongo services without
# invoking the graph or writing values.
uv run earnings --ticker "$ASML_TICKER" --form 6-K --accession "$ASML_ACCESSION" --dry-run
uv run earnings --ticker "$TSM_TICKER" --form 6-K --accession "$TSM_ACCESSION" --dry-run

if [[ "$MODE" == "preflight" ]]; then
  echo "Preflight complete. No LLM extraction or MongoDB writes were performed."
  echo "Review the staging seed and dry-run results; then rerun with:"
  echo "  PHASE6_CONFIRM=YES $0 --execute"
  exit 0
fi

# Live extraction writes only to STAGING_NORMALIZE_DB_NAME on the explicitly
# supplied STAGING_MONGODB_URI. The normal atomic save path is used.
uv run earnings --ticker "$ASML_TICKER" --form 6-K --accession "$ASML_ACCESSION"
uv run earnings --ticker "$TSM_TICKER" --form 6-K --accession "$TSM_ACCESSION"

echo "Staging extraction complete. Verify values, currency, source_form_type, "
echo "quarterly concepts, guidance, Q4 behavior, and rerun idempotence before rollout."
