# AGENTS.md — Project Context for Coding Agents

## Overview

**LangGraph-based earnings extraction pipeline** that ingests SEC 8-K Exhibit 99.1 press releases and extracts financial metrics into MongoDB (`normalize_data`). Uses LLMs for extraction with deterministic guardrails for accuracy.

- **Language**: Python 3.12+
- **Package manager**: `uv`
- **LLM framework**: LangChain + LangGraph
- **State**: `EarningsAgentState` — a `TypedDict` carrying all pipeline data
- **Deployment**: CLI (`earnings`) or Redis worker (`earnings-8k-worker`) via Docker

## Quick Commands

```bash
uv sync                              # Install deps
uv run pytest -q                     # All tests
uv run pytest -q tests/test_<module>.py  # Single test file
uv run earnings --ticker MSFT        # CLI run (SEC EDGAR path)
uv run earnings --ticker MSFT --dry-run  # Connectivity check, no LLM
uv run earnings --ticker MSFT -v     # Verbose (DEBUG logging)
uv run earnings-skills               # Browse failure-mode skill catalog
uv run earnings-failures             # Browse failed extractions
docker compose up -d --build         # Start 8-K worker
docker compose restart worker-8k     # After code edits (src is mounted)
```

## Architecture

### LangGraph Pipeline (7 nodes)

```
load_company_concepts → detect_document_type → extract_html_text
  → extract_financial_metrics ⇄ analyze_metrics   (agentic loop, ≤3 passes)
    → cleanup_metrics → mongodb_save
```

Each node is a pure function `(EarningsAgentState) → EarningsAgentState`, wrapped by `with_hooks()` for structured logging, timing, and exception-to-failure conversion. Node failures set `status="failed"` and short-circuit to `END`.

### State Machine

Status progression: `discovered → fetched → text_extracted → extracted → saved | failed`

Key state fields:
- `target_concepts` — GAAP concept list driving **targeted extraction** (required — no generic fallback)
- `metrics` — raw extracted key-value pairs (company-native labels)
- `concept_metrics` — `concept_id → float` for normalize_data upsert
- `findings` — structured `Finding` objects from analysis (severity: high/medium/low)
- `needs_reextract` — routing signal triggering the extract↔analyze loop
- `sec_report_date` — authoritative period-end date from SEC EDGAR submissions API
- `chunk_metric_sources` — per-metric chunk provenance for scoped retries
- `mapped_metric_keys` — keys successfully matched to concept_ids (protected from cleanup removal)

### Agentic Loop (extract ↔ analyze)

1. **extract_financial_metrics**: Chunks text (GAAP-section-aware), one LLM call per chunk, merges results. On retries, uses `extraction_notes` from prior analysis. Chunk-level provenance enables scoped retries (only affected chunks re-run).

2. **analyze_metrics**: Runs deterministic checkers (presence, identity violations, suspect rounding, case duplicates, GAAP/Non-GAAP leakage), producing `Finding` objects. High-severity findings trigger `needs_reextract=True` (up to `MAX_EXTRACTION_ATTEMPTS`, default 3). Progress detection prevents infinite loops (same findings on consecutive passes → break early).

### Three-Tier Concept Mapping

- **Tier 0**: Direct taxonomy_key match (LLM returns `[us-gaap:Revenue]` bracket key or bare key)
- **Tier 1**: Deterministic label matching (exact, then normalized)
- **Tier 2**: LLM semantic mapping — matches orphaned keys to unmapped concepts
- **Tier 3**: Pure-Python derivation engine — computes missing values from mapped ones (e.g., Gross Profit = Revenue − COGS)

### Code Organization

```
src/earnings_agents/
  workflow.py              # LangGraph graph builder + routing helpers
  workflow_state.py        # EarningsAgentState TypedDict
  config.py                # All env-var-driven settings
  hooks.py                 # Node lifecycle hooks (logging, timing, error handling)
  llm_factory.py           # Multi-provider LLM client builder
  company_registry.py      # CIK/ticker lookup from SEC company_tickers.json
  worker_progress.py       # Redis pub/sub progress streaming for worker mode
  nodes/                   # LangGraph node functions (one per pipeline stage)
    load_company_concepts_node.py
    detect_document_type.py
    extract_html_text.py
    extract_financial_metrics.py  # Largest node — chunking, extraction, concept mapping
    analyze_metrics.py            # All checkers + re-extract decision
    cleanup_metrics.py            # LLM-based key removal with 3 guardrails
  analysis/                # Deterministic checkers, validators, calculators, skills
    findings.py            # Finding model + all observer checkers (ADR-0003)
    validators.py          # Correctors that null-out implausible values
    calculators.py         # Tier-3 derivation and LLM role identification
    skills.py              # Failure-mode skill catalog
    critical_metrics.py    # TIER1/TIER2/TIER3 metric registries
    metric_patterns.py     # Shared regex patterns
    industry.py            # Industry-specific prompt hints
  extraction/              # Chunking, merging, concept mapping
    chunker.py             # Document prescan, GAAP-section-aware chunking
    merger.py              # Cross-chunk merging, scale resolution, parse helpers
    concept_mapper.py      # LLM concept mapping + prompt list builder
  tools/                   # External integrations
    edgar_client.py        # SEC EDGAR API (submissions, exhibit 99.1 discovery)
    mongodb_client.py      # Raw earnings MongoDB client
    normalize_data_client.py  # normalize_data DB client (companies, concepts, upsert)
    redis_queue.py         # Redis queue helpers
    http_client.py         # HTTP with headers + caching
    playwright_scraper.py  # JS-rendered page fallback
    llm_extractor.py       # Core LLM invoke harness with retry + semaphore
    llm_table_classifier.py   # LLM-based HTML table classification
    llm_concept_mapper.py     # LLM concept mapping implementation
  cli/                     # CLI entry points
    earnings.py            # Main CLI (multi-ticker with ThreadPoolExecutor)
    worker_8k.py           # Redis BLPOP consumer for 8-K filings
    failures.py            # Browse failed extraction records
    skills.py              # Browse failure-mode skill catalog
```

### LLM Provider System

Multi-provider via `llm_factory.py`:
- **Ollama** (default, local) — env vars: `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_NUM_CTX`
- **Groq** — `GROQ_API_KEY`, `GROQ_MODEL`, with RPM/TPM rate limiting
- **Gemini** — `GEMINI_API_KEY`, `GEMINI_MODEL`, via official `google-genai` SDK
- **DeepSeek** — `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`

Uniform interface: `llm.invoke(str) -> str`. JSON mode per-call via `build_llm(format_json=True)`. Optional disk cache (`LLM_CACHE=1`) for dev.

### Key Config Env Vars

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_PROVIDER` | `ollama` | ollama / groq / gemini / deepseek |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for worker queue |
| `REDIS_QUEUE_NAME` | `sec:filings:8k` | Worker queue name |
| `STRICT_ACCURACY` | `1` | Refuse save on identity failures |
| `CLEANUP_METRICS` | `1` | Run LLM cleanup pass |
| `SOURCE_GROUNDING` | `0` | Ask LLM for per-metric source snippets |
| `MAX_EXTRACTION_ATTEMPTS` | `3` | Max extraction passes |
| `CHUNK_SIZE` | `0` | 0 = per-provider default (6K ollama, 80K groq, 25K others) |
| `PROMPT_HISTORY_PERIODS` | `3` | Periods for prompt pruning |

### Two Deployment Modes

1. **CLI** (`cli/earnings.py`): Runs the graph via `ThreadPoolExecutor` for one or more tickers. Uses Rich for live progress. Default `--max-workers 8`.

2. **Redis Worker** (`cli/worker_8k.py`): Long-running `BLPOP` consumer on the `sec:filings:8k` queue. Publishes real-time progress events to Redis pub/sub (`sec:worker:events`). Includes heartbeat, dead-letter queue, and retry logic. Deployed via `docker-compose.yml`.

### Key Patterns & Guardrails

- **Metric keys preserved verbatim** — no normalization at extraction time (ADR-0002)
- **Cleanup is append-only in spirit** — LLM cleanup can only drop keys; three deterministic guardrails reject any mutation, invention, or new-identity-failure
- **Save gate**: `STRICT_ACCURACY` (default on) refuses MongoDB upsert when accounting identity checks fail
- **EDGAR rate limiting**: Token-bucket at ≤10 req/s
- **Incremental guard**: Skips filings whose `sec_report_date` ≤ the most recently stored period
- **No-progress detection**: Breaks the extract↔analyze loop when findings don't change between passes

### Tests

`pytest` with `pytest-asyncio` (auto mode). Golden fixture tests in `tests/fixtures/golden/` validate extraction output against known-good results. Tests are in `tests/` mirroring the source structure.

### Important Project Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies, build config, CLI entry points |
| `docker-compose.yml` | 8-K worker deployment |
| `Dockerfile` | Multi-stage Docker build with uv + Playwright Chromium |
| `.env` | Local environment (gitignored) |
| `.env.example` | Environment template |
| `data/reference/sec_company_tickers.json` | Cached SEC company registry (24h TTL) |
