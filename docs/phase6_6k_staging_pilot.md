# Phase 6 — ASML / TSM 6-K staging pilot and rollout

**Status:** staging harness and runbook implemented. The write pilot itself is **not run yet** because no `STAGING_MONGODB_URI` / isolated staging database is configured in this environment. Do not point the pilot at production.

## Purpose

Exercise the complete 6-K path — accession resolution, multi-exhibit/PDF fetch, period agent, annual-to-quarter concept bootstrap, extraction, USD-preferred/native-currency persistence, source-form provenance, guidance, and save/Q4 guards — using a staging copy of `normalize_data`.

The pilot uses pinned accessions so the run is deterministic and does not rely on automatic 6-K candidate selection:

| Issuer | Ticker | CIK | Accession | Filing |
|---|---|---|---|---|
| ASML Holding N.V. | ASML | `0000937966` | `0001628280-26-048235` | 2026-07-15 earnings 6-K; EX-99.1 press release, EX-99.2 presentation, EX-99.3 financial statements |
| Taiwan Semiconductor Manufacturing Co. | TSM | TSM | `0001046179` | `0001046179-26-000451` | 2Q26 earnings 6-K; EX-99.1 release and EX-99.2 presentation |

The configured `normalize_data` database was checked read-only: ASML has 305 annual concepts and TSM has 466, with zero quarterly concepts for both. Both therefore exercise the Phase 1 company-level quarterly bootstrap. TSM's auto-discovery is intentionally not relied upon because its earnings exhibit filenames are opaque.

## Staging data prerequisites

Use an **isolated staging MongoDB endpoint** and a dedicated database name containing `staging`. Seed the staging database with:

- `companies` rows for ASML and TSM;
- their `normalized_concepts_annual` rows (including parent/dimensional references);
- `concept_values_annual` and relevant `concept_values_quarterly` history if testing guidance scoring, prior-quarter cash-flow de-accumulation, or Q4 reconciliation.

The pilot must not copy or write anything to the production URI. `NORMALIZE_DB_NAME` now selects the SEC normalization database (default remains `normalize_data`) so a dedicated staging DB can be used without changing production defaults.

## Read-only preflight

```bash
STAGING_MONGODB_URI="$STAGING_MONGODB_URI" \
STAGING_NORMALIZE_DB_NAME="normalize_data_phase6_staging" \
./scripts/phase6_6k_staging_pilot.sh
```

The script checks that the endpoint/database are staging-only, confirms company and annual-concept seeds, and runs pinned `earnings --dry-run` checks for both filings. **No graph invocation and no MongoDB writes occur in preflight mode.**

## Execute the isolated staging pilot

After inspecting the preflight output and confirming that the database is isolated:

```bash
STAGING_MONGODB_URI="$STAGING_MONGODB_URI" \
STAGING_NORMALIZE_DB_NAME="normalize_data_phase6_staging" \
PHASE6_CONFIRM=YES \
./scripts/phase6_6k_staging_pilot.sh --execute
```

`--execute` is intentionally guarded by `PHASE6_CONFIRM=YES`. Each pinned filing still runs through the normal graph and save gate. Re-running is safe: concept creation is idempotent, value replacement is atomic write-first, and the source accession/form are stored for traceability.

## Acceptance checklist

For both issuers, verify in the staging database:

1. **Filing/period:** accession and `source_form_type="6-K"` are present; the period agent selected the correct quarter/end date from the filing, not the filing date.
2. **Concept bootstrap:** all annual concepts were copied only because the quarterly concept collection was empty; copied `_id` and hierarchy references remain valid; `form_type` is `10-Q`; no annual values were copied into quarterly values.
3. **Quarterly values:** values come from the selected quarterly release, use `reporting_period.quarter`/`fiscal_year`, have per-value currency (`USD` if a company-reported USD presentation was selected, otherwise EUR/TWD as reported), and do not mix currencies within a statement.
4. **Evidence/provenance:** accession, source form, scale, source lines, dimension metadata, and calculated status are present as appropriate.
5. **Idempotency:** re-running the same pinned accession leaves one value per concept/period, and does not delete the previous period before replacement values are written.
6. **Unknown/mismatched currency:** an unconfirmed value is omitted, never tagged USD; derived Q4 or guidance scores with mismatched currencies are skipped.
7. **TSM YTD cash flow:** if the 6-K has only a six-month/YTD cash-flow column, verify un-de-accumulated new cash-flow concepts are omitted rather than stored as quarter values.
8. **No production side effects:** all collections inspected/modified belong to the staging database.

Only after both issuers pass should a small production rollout start with pinned accessions and manual review. Broader automatic 6-K discovery and historical backfill should remain disabled until the metadata candidate filter's precision has been reviewed against a wider filing sample.
