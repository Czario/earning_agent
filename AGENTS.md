# AGENTS.md — SEC 8-K Earnings Extraction Pipeline

Agent-based pipeline that fetches SEC 8-K Exhibit 99.1 press releases and extracts
income-statement metrics into MongoDB `normalize_data`. Two tool-calling agents run per
filing — a **period agent** (decides the reporting period) and an **extraction agent**
(navigates the plain text, extracts metrics). Deterministic guardrails (scale pre-scan,
Tier 0/1 concept mapping, CALC derivation, strict save gate) keep it accurate.

- **Stack**: Python 3.12+, `uv`, LangChain + LangGraph, MongoDB, Redis
- **Deploy**: CLI (`earnings`) or Redis worker (`earnings-8k-worker`) via Docker

## Quick Commands

```bash
uv sync                                  # install deps
uv run pytest -q                         # all tests
uv run earnings --ticker MSFT            # CLI run (SEC EDGAR path)
uv run earnings --ticker MSFT --dry-run  # connectivity check, no LLM
uv run earnings --ticker MSFT -v         # DEBUG logging
uv run earnings-failures                 # browse degraded/failed runs (raw mongo)
docker compose up -d --build             # start 8-K worker
docker compose restart worker-8k         # after code edits (src is volume-mounted)
```

## Architecture

Linear 6-node LangGraph (`graph.py`). Each node is a pure `(State) → State` function
wrapped by `with_hooks()` (structured logging, timing, exception → `status="failed"`
short-circuit to `END`).

```
fetch_filing → detect_period → check_period → load_company_concepts
    → agent_document_pipeline → mongodb_save → END
```

| # | Node | File | Job |
|---|------|------|-----|
| 1 | `fetch_filing` | `nodes/fetch.py` | Fetch **ALL** EX-99 text exhibits (press release + presentation + supplemental — income statements often live in a supplemental exhibit, e.g. BofA's 99.3), convert each to plain text, concatenate with `DOCUMENT n OF m` headers; records `document_map` (exhibit line ranges, truncation, skips) in state. Non-text exhibits skipped; per-exhibit/total size caps. |
| 2 | `detect_period` | `agent/period.py` | **The period agent** (below). Reads the document header and reports `{period_type, period_end, quarter, period_label}`. Failure = run failure — **no deterministic period inference anywhere**. |
| 3 | `check_period` | `nodes/check.py` | With the agent-detected period, computes `(fiscal_year, quarter)` via `compute_fiscal_period` and checks **exact-period existence only** (no accession checks): same fiscal period stored → schedule `_pending_replace` and **continue** (delete stays deferred to save); else proceed. An annual period is checked/replaced in `concept_values_annual` only — a quarterly Q4 record for the same fiscal year is never checked or touched. |
| 4 | `load_company_concepts` | `nodes/concepts.py` | Consumes `detected_period_type` (no period decision here). **Computes the recent-value window FIRST** (`get_recently_valued_concept_ids`: concepts valued in any of the last `PROMPT_HISTORY_PERIODS` (3) periods — quarterly → `concept_values_quarterly`, annual → `concept_values_annual`), then loads **only those** concept docs (`_id $in` query-level filter) — non-recent concepts are never even loaded. **Exception**: `system:` and `calculated` concepts are ALWAYS loaded (CALC derivation targets — excluding them would make them permanently underivable); they are never in the agent's extraction list. **Skips** when no historical data exists OR no concept was valued recently (nothing to extract). |
| 5 | `agent_document_pipeline` | `agent/pipeline.py` | Prescan → prior values → prompt → extraction agent loop → map → derive (below). |
| 6 | `mongodb_save` | `nodes/save.py` | STRICT_ACCURACY gate → deferred replace (`delete_fiscal_period` immediately before upsert — no data-loss window) → upsert into `concept_values_{quarterly\|annual}`. |

> **Vestigial retry machinery.** `state.py` retry-loop fields (`needs_reextract`,
> `findings`, `extraction_notes`, `skill_effectiveness`) and
> `build_retry_briefing()` in `prompts.py` are leftovers from a retired
> agent⇄analyze retry loop. The graph runs **once per filing**; retries exist only at
> the **job level** in the Redis worker (re-queue with an `attempts` counter,
> `--max-attempts`, then dead-letter queue). **That job-level retry is also the only
> safety net for period-agent failures.**

### Period agent (`agent/period.py`) — the single source of truth

- Runs through the **shared agent loop + tools** (`get_document_info`, `search`,
  `read_lines`, `get_company_info` + terminal `finalize_period`), **open-ended** —
  no step cap.
- Given `fiscal_year_end` (MMDD from `normalize_data.companies`); returns strict JSON
  `{period_type, period_end, quarter, period_label}`. Multi-exhibit bundles are
  surfaced via the `get_document_info` exhibit map — the FIRST document is the
  press release carrying the period header.
- Robust parsing: `period_end` accepts ISO, `M/D/YYYY`, and month-name forms;
  a `period_label` without a parseable date is normalized to a standard header
  (save needs to parse it).
- **Business rules** (`apply_period_business_rules`, a gate — not a fallback):
  - **Q4 == annual**: `quarter=4` or "Fourth Quarter" label → `annual`, `quarter=null`.
    When a release shows both Q4 and fiscal-year columns, the fiscal-year column wins.
  - `period_end` must fall in the sanity window vs the 8-K `filing_date` (+7d / −450d).
  - Quarterly periods must name a quarter (1–3).
- Any failure (LLM error, unparseable output, rule rejection) → `status="failed"`, END.
  Deleted: `_infer_period_type`, cadence `get_next_period_type`, EDGAR
  `_infer_period_end`/`_infer_8k_fiscal_period`/`_extract_period_from_exhibit`,
  filename-date logic, prescan period regex.

### Extraction pipeline internals (`agent/pipeline.py`)

1. `prescan_document` (`agent/derive.py`) — deterministic **scale** detection only
   (thousands/millions/billions); period is the period agent's job
2. `load_prior_values` — prior-period DB values for the agent's `get_prior_value` tool
3. Prompt = `PIPELINE_SYSTEM_PROMPT` + `build_concept_list` (`system:` CALC concepts
   excluded — the agent never extracts them; the list is already hard-filtered
   to recently-valued concepts upstream) + period hints from the detected period
   (incl. the Q4→annual column rule) + advisory industry context
   (`agent/industry.py::build_industry_context` — SIC code/description from
   `companies.industry`, injected on EVERY pass incl. retries)
4. `run_agent_loop` (ReAct; `agent/loop.py`) until `finalize_extraction` —
   **open-ended, no step cap**; fallback recovers a final JSON blob from the last AI message
5. `_parse_llm_response` applies the `__scale__` multiplier — **never** scales keys
   matching percentage/per-share/share-count regexes
6. `map_concepts` — Tier 0: `[taxonomy_key]` bracket key or raw `taxonomy_key`;
   Tier 1: exact or whitespace-normalized label
7. `derive_missing_concepts` — one lightweight LLM pass computes missing `system:`
   (CALC) parents from the path hierarchy; never overrides extracted values.
   Margins/ratios are excluded from the `GP = Rev − CoR` shortcut (bug observed live).
   `[derived]` lines report each computed concept's label + value and each
   omitted one by name (`[llm] derive missing CALC — <names> → calling llm` lists
   the candidates)
8. Observability: `missing_concept_labels` / `missing_toplevel_labels` /
   `missing_segment_labels` — dimensionality from DB flags `dimension` /
   `dimension_concept`, **not** `"|" in taxonomy_key`

### Shared agent loop (`agent/loop.py`)

`run_agent_loop(..., finalize_name, finalize_description, parse_final_result,
recovery_regex)` — one loop implementation shared by both agents. Extraction passes
`finalize_extraction` + the scale-aware parser; period detection passes
`finalize_period` + the period JSON parser. All `report_call` surfacing
(`[llm]`/`[tool]` lines → CLI highlighting/counters, worker `sec:worker:events`) is
reused automatically.

### Agent tools (`agent/tools.py`)

`get_document_info`, `read_lines`, `search` (word-indexed, context blocks, 15-block cap),
`get_prior_value`, `verify_identity` (GP = Rev − CoR), `calculate` (safe AST arithmetic),
`get_company_info` (cached SIC profile from state, DB fallback — advisory only). Tool results truncated to 8000 chars.

### Concept lookup & fiscal math (`integrations/normalize.py`)

- `get_statement_concepts(cik, statement_types, period_type)` — returns `{_id, concept,
  label, path, statement_type, taxonomy_key, dimension, dimension_concept}`, sorted by
  `(path, order_key)`; labels disambiguated on collision; taxonomy keys path-qualified.
- `compute_fiscal_period(end_date, fy_end_month, period_str)` — fiscal year + quarter
  from the **agent-read** period date/label; prefers unambiguous period-str durations,
  calendar-math fallback with 3-day boundary tolerance for Dec FY.
- `upsert_concept_values` — routes quarterly/annual (`period_type_override` = the
  agent's decision → FY-end month → period-str duration); deletes the period's docs
  before bulk-write; `calculated` flag per derived id; `accession_number` stamped for traceability (never used for checks/dedup).

## Code map

```
src/earnings_agents/
  graph.py, hooks.py, state.py, config.py, llm.py, registry.py, progress.py
  agent/        period.py · pipeline.py · loop.py · tools.py · prompts.py · derive.py · industry.py
  nodes/        fetch.py · check.py · concepts.py · detect.py · save.py
  integrations/ edgar.py · normalize.py · mongo.py · redis.py · http.py · html.py · playwright.py
  cli/          earnings.py · worker.py · failures.py
tests/          9 test modules + fixtures/golden/ (scale-parsing JSON cases)
```

- `llm.py` — provider factory (`build_llm` → `invoke(str)->str` for the derive pass;
  `build_chat_llm` → `bind_tools()` for the agent loops)
- `hooks.py` — `with_hooks` + per-thread callbacks (`report_call` drives CLI/worker progress)
- `progress.py` — `WorkerProgressPublisher` (Redis pub/sub `sec:worker:events`), heartbeat
- `registry.py` — CIK/ticker lookup from `data/reference/sec_company_tickers.json` (24 h disk cache)
- `integrations/edgar.py` — submissions API → 8-K Item 2.02 → filing index → EX-99.1 URLs;
  `get_latest_earnings_url` returns `(url, supplemental, report_date, accession,
  filing_date)`; **reportDate is passed through raw/informational** (the period agent
  reads the document); token bucket (`EDGAR_RATE_LIMIT`, default 8 req/s); retry on 429/5xx
- `integrations/mongo.py` — raw earnings collection (`earnings_db.earnings`)

## Guardrails & invariants — do not break

- **Period comes from the period agent or the run fails.** No regex/filename/
  EDGAR-reportDate/cadence inference exists anywhere in the codebase.
- **Extraction target = recently-valued concepts only.** A concept is extracted
  only if it had a stored value in any of the last 3 periods (quarterly filing →
  last 3 quarterly periods in `concept_values_quarterly`; annual → last 3 annual
  periods in `concept_values_annual`). Non-recent concepts are filtered out at
  `load_company_concepts` — never in the agent prompt, mapping, or save.
  **Exception**: `system:`/`calculated` concepts are always loaded so the CALC
  derivation pass can compute them (they are never in the agent's extraction
  list). Empty recent set → run skipped (nothing to extract).
- **Industry context is advisory-only.** SIC data from `companies.industry` is
  injected into the extraction prompt and `get_company_info` as read-only
  context (helps recognize filing terminology); it can never add concepts
  outside the recent-value target, supply or infer values, or override
  anything read from the filing. Missing industry data never fails a run.
- **Q4 is never extracted as quarterly.** Q4 == annual; both-column releases → the
  fiscal-year (annual) column. Enforced in the period agent prompt, the business-rules
  gate, the extraction prompt, and naturally by `quarter=null` annual upserts.
- **Pipeline never edits extracted numbers** — identity checks are detection-only; the
  agent fixes its own mistakes via tools, never the code
- **`calculated` contract** — `false` = verbatim from filing; `true` = computed by the
  derivation pass (`derived_concept_ids` tracked in state)
- **Save gate** — `STRICT_ACCURACY` (default on) refuses the upsert on unresolved
  high-severity findings
- **Deferred replace** — `check_period` only *schedules* `_pending_replace` when the
  exact fiscal period (annual → annual collection, quarterly → quarterly collection)
  already exists; the delete happens in `mongodb_save` immediately before the upsert
  (no data loss on mid-run failure). No accession checks anywhere — re-runs always
  replace the same exact period; a quarterly Q4 record is never checked or deleted
  by annual processing.
- **Scale handling** — deterministic document pre-scan + `__scale__` field; parser
  refuses to scale percentages, per-share values, and share counts — incl.
  CamelCase taxonomy keys (`margin`/`yield`/`growth`/`ratio`/`eps`/`per share`…,
  observed live: `custom:NetInterestMarginCompanyProvided` stored 2,080,000 for
  a 2.08% yield). Ratio matching is word-bounded so "Operating expenses"
  (contains "ration") still scales.
- **`report_call` convention** — every LLM/tool/DB call surfaced as `[llm]`/`[tool]`/
  `[db]`-prefixed lines; a single `[industry]` line per extraction pass shows the
  injected SIC context; the CLI highlights `→ calling llm` yellow and industry
  cyan and counts LLM calls; the worker publishes the same lines to
  `sec:worker:events` (`call_llm` vs `call_industry` vs `call` kinds)
- **Multi-component cost rows** — the system prompt teaches the agent to SUM subtotal +
  additional cost rows (amortization/depreciation/impairment) and verify via
  `verify_identity`

## LLM providers (`llm.py`)

`LLM_PROVIDER`: `ollama` (default) | `groq` (RPM/TPM token-bucket rate limiter) |
`deepseek` | `gemini` (official `google-genai` SDK). Opt-in disk cache `LLM_CACHE=1`
(sha256-keyed, dev only).

⚠ **Gemini cannot run the agent loops** — `build_chat_llm()` raises `ValueError`
(no LangChain chat model for google-genai); with Gemini configured, period detection
and extraction both fail → the run fails (worker retries are the only net). The derive
pass (`build_llm`) still works.

## Config env vars (`config.py`)

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_PROVIDER` | `ollama` | ollama / groq / gemini / deepseek |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` / `OLLAMA_NUM_CTX` | `llama3.1:8b` / localhost:11434 / 4096 | Local provider |
| `GROQ_API_KEY` / `GROQ_MODEL` / `GROQ_RPM` / `GROQ_TPM` | — / `openai/gpt-oss-120b` / 30 / 12000 | Groq + rate budgets |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` | — / `deepseek-chat` | DeepSeek |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-2.5-flash` | Gemini |
| `MONGODB_URI` | `mongodb://localhost:27017` | normalize_data lives here (`_NORMALIZE_DB`) |
| `MONGODB_DB` / `MONGODB_COLLECTION` | `earnings_db` / `earnings` | Raw earnings store |
| `REDIS_URL` / `REDIS_QUEUE_NAME` | `redis://localhost:6379/0` / `sec:filings` | Worker queue (deploy sets `sec:filings:8k`) |
| `STRICT_ACCURACY` | `1` | Refuse save on unresolved high-severity findings |
| `EXTRACTION_MAX_CHARS` | `400000` | Cap on `raw_text` stored in state |
| `PROMPT_HISTORY_PERIODS` | `3` | Window (stored periods) for the hard extraction-target filter — always on |
| `MAX_EXTRACTION_ATTEMPTS` | `3` | Vestigial (see retry note) |
| `LLM_CACHE` | `0` | Dev-only LLM response disk cache |
| `EDGAR_RATE_LIMIT` | `8` | SEC token-bucket req/s (in `edgar.py`, not config) |
| `FETCH_EXHIBIT_MAX_CHARS` / `FETCH_TOTAL_MAX_CHARS` | 400000 / 1200000 | Per-exhibit and total text caps for multi-exhibit fetching |

## Deployment modes

1. **CLI** (`cli/earnings.py`) — multi-ticker `ThreadPoolExecutor` (`--max-workers 8`),
   Rich live progress; `--dry-run` prints ready/warning/blocked verdicts without LLM
   calls. `_build_8k_state` resolves URL + accession + filing date only — no
   pre-guards or dedup; the exact-period check lives in the graph (`check_period`).
2. **Redis worker** (`cli/worker.py`) — long-running `BLPOP` on `sec:filings:8k`.
   Receives accession payloads from admin_backend, resolves Exhibit 99.1, runs the same
   graph. Publishes progress to `sec:worker:events`, heartbeats, job-level re-queue
   retries (`--max-attempts`, dead-letter `sec:filings:dlq:8k`), updates
   `stock_load_requests` status in MongoDB. Docker service `worker-8k` (src volume-
   mounted; external network `backend_true_grids_backend_network`).

Both paths share `_build_8k_state` in `cli/earnings.py` — URL/accession resolution and
state construction are identical by design.

## Tests

`pytest` (asyncio auto). `tests/`: `test_period_detection` (period agent: JSON parsing,
Q4→annual coercion, sanity window, failure→failed, loop terminal-tool parameterization),
`test_company_registry`, `test_concept_flags` (concept-audit regression: dimension
flags, path collisions, margin safety), `test_edgar_client`, `test_failures_cli`,
`test_llm_factory`, `test_mongodb_client`, `test_normalize_data_client` (fiscal math,
concept collection selection, `check_period` node, save filters), `test_save_gate`.
`tests/fixtures/golden/` holds scale-parsing JSON cases.

## Design decisions & known issues

- **Period detection is agent-only, by design.** Every deterministic heuristic was
  deleted (`_infer_period_type`, cadence state machine, EDGAR prior-year projection,
  exhibit regexes, filename dates, prescan period regex). If the agent fails, the run
  fails and the worker retries — there is deliberately no fallback path that could
  silently misclassify a period (the AMAT Q3-as-annual incident).
- **No separate company-memory store.** `normalized_concepts_*` docs are already
  per-CIK with company labels; the pipeline exploits them fully (prompt, pruning,
  Tier 0/1 mapping). The residual gap: 8-K press-release phrasing sometimes diverges
  from XBRL labels → unmapped keys land in `missing_concept_labels` observability.
- **Upstream data quality** (normalizer pipeline, not fixed here): cross-company
  contamination (`meta:*` members from other CIKs), trailing-whitespace label variants,
  geography members as income rows. Recent-period pruning masks most of it for
  established companies; bootstrap companies (no history) see the full polluted list.
- **`--allow-inconsistent` CLI flag is currently a no-op** — it sets
  `graph.STRICT_ACCURACY = False`, but `save.py` reads `STRICT_ACCURACY` from `config`,
  and no node populates `findings` in the current graph, so the gate never fires.
- `state.py` declares the `missing_*` label fields twice (harmless duplicate).
