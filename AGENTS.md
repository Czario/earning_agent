# AGENTS.md — SEC 8-K Earnings Extraction Pipeline

Agent-based pipeline that fetches SEC 8-K Exhibit 99.1 press releases and extracts
income-statement metrics into MongoDB `normalize_data`. Two tool-calling agents run per
filing — a **period agent** (decides the reporting period) and an **extraction agent**
(navigates the plain text, extracts metrics with per-value evidence). There is no
verifier agent and no retry loop — the pipeline is a single extraction pass.
Deterministic guardrails (scale pre-scan + per-value scale/currency evidence, Tier 0/1
concept mapping + `map_concept`, in-loop CALC derivation via `compute`, company-identity cross-check,
atomic period replace, strict save gate) keep it accurate.

- **Stack**: Python 3.12+, `uv`, LangChain + LangGraph, MongoDB, Redis
- **Deploy**: CLI (`earnings`) or Redis worker (`earnings-8k-worker`) via Docker

## Quick Commands

```bash
uv sync                                  # install deps
uv sync --extra dev                      # + pytest (test suite)
uv run pytest tests/ -q                  # run the test suite (69 tests)
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
| 1 | `fetch_filing` | `nodes/fetch.py` | Fetch **ALL** EX-99 text exhibits AND PDF documents (press release + presentation + supplemental — income statements often live in a supplemental exhibit, e.g. BofA's 99.3; manually-triggered shareholder letters arrive as PDFs, e.g. on Q4CDN), convert each to plain text (HTML via BeautifulSoup, PDF via pdfplumber with `PDF page n of m` separators), concatenate with `DOCUMENT n OF m` headers; records `document_map` (doc line ranges, truncation, skips) in state. Non-fetchable exhibits (images) skipped; per-exhibit/total size caps; `file_type` = `pdf` when any PDF was fetched, else `html`. |
| 2 | `detect_period` | `agent/period.py` | **The period agent** (below). Reads the document header and writes the canonical `detected_period` record `{period_type, period_end, quarter, period_label, fiscal_year}`. Failure = run failure — **no deterministic period inference anywhere**. |
| 3 | `check_period` | `nodes/check.py` | Consumes the canonical agent-detected period (including its already-resolved `fiscal_year` and `quarter`) and checks **exact-period existence only** (no accession checks): same fiscal period stored → schedule `_pending_replace` and **continue** (replacement stays deferred to save — atomic write-first inside `upsert_concept_values`); else proceed. An annual period is checked/replaced in `concept_values_annual` only — a quarterly Q4 record for the same fiscal year is never checked or touched. |
| 4 | `load_company_concepts` | `nodes/concepts.py` | Consumes the canonical `detected_period` (no period decision here). Loads the **full eligible concept universe**, then builds the target as **recent concepts ∪ all dimensional (segment/breakdown) rows ∪ `system:`/`calculated` concepts**. The recent-value window (`get_recently_valued_concept_ids`, last `PROMPT_HISTORY_PERIODS` periods) is a *prioritization* signal, not an eligibility filter — new segments/newly disclosed rows stay extractable without history. `system:`/`calculated` concepts are CALC derivation targets — rendered to the agent as compute-only (not extracted verbatim from the filing). No history → full-universe bootstrap; no concepts at all → skip. Malformed upstream rows (labels with no alphabetic word, e.g. `custom:404` page-number pollution) and rows whose hierarchy `path` carries a bare page-number segment (`404`/`555`) are dropped before the target is built — no filing prints them. |
| 5 | `agent_document_pipeline` | `agent/pipeline.py` | Prescan → prior values → prompt → extraction agent loop (single pass) → map → derive → findings (below). No verifier agent, no retry loop. |
| 6 | `mongodb_save` | `nodes/save.py` | STRICT_ACCURACY gate → currency gate → **atomic period replace** inside `upsert_concept_values` (write-first, then a stale sweep deletes only docs not carrying the current save token — no delete-before-write window) → upsert into `concept_values_{quarterly\|annual}`. |

> **Job-level retry only.** The graph runs **once per filing**; there is no
> in-graph re-extract loop. Retries exist only at the **job level** in the Redis
> worker (re-queue with an `attempts` counter, `--max-attempts`, then dead-letter
> queue). Deterministic provider failures (billing/auth — e.g. DeepSeek 402
> Insufficient Balance) are classified non-retryable and go straight to the DLQ.
> **Job-level retry is also the only safety net for period-agent failures.**
> `state.findings` is the LIVE save-gate input populated by the
> pipeline (currency, incomplete exhibits, missing concepts, hierarchy
> ambiguity).

### Period agent (`agent/period.py`) — the single source of truth

- Runs through the **shared agent loop + tools** (`get_document_info`, `search`,
  `read_lines`, `get_company_info` + terminal `finalize_period`), **open-ended** —
  no step cap.
- Given `fiscal_year_end` (MMDD from `normalize_data.companies`) as fiscal-calendar context; returns strict JSON
  `{period_type, period_end, quarter, fiscal_year, period_label}`. The filing's
  fiscal-year/quarter evidence is authoritative; the graph stores the complete
  agent result as the canonical `detected_period` record. Multi-exhibit bundles are
  surfaced via the `get_document_info` exhibit map — the FIRST document is the
  press release carrying the period header.
- Robust parsing: `period_end` accepts ISO, `M/D/YYYY`, and month-name forms;
  a `period_label` without a parseable date is normalized to a standard header
  (save needs to parse it).
- **Business rules** (`apply_period_business_rules`, a gate — not a fallback):
  - **Q4 == annual**: `quarter=4` or "Fourth Quarter" label → `annual`, `quarter=null`.
    When a release shows both Q4 and fiscal-year columns, the fiscal-year column wins.
  - Quarterly periods must name a quarter (1–3).
- Any failure (LLM error, unparseable output, rule rejection) → `status="failed"`, END.
  Deleted: `_infer_period_type`, cadence `get_next_period_type`, EDGAR
  `_infer_period_end`/`_infer_8k_fiscal_period`/`_extract_period_from_exhibit`,
  filename-date logic, prescan period regex.

### Extraction pipeline internals (`agent/pipeline.py`)

1. `prescan_document` (`agent/derive.py`) — deterministic **scale** detection only
   (thousands/millions/billions); period is the period agent's job
2. `load_prior_values` — prior-period DB values for the agent's `get_prior_value` tool
3. Prompt = `PIPELINE_SYSTEM_PROMPT` + `build_concept_list` (the verbatim
   extract list) + `build_calc_derivation_block` (CALC `system:`/`calculated`
   concepts rendered as COMPUTE-ONLY with their rollup formulas — the agent
   computes them with `compute()`, never extracts them; the target list itself
   is recent ∪ dimensional ∪ system/calculated from the concepts node, not a
   hard recent-only filter) + period hints from the detected period
   (incl. the Q4→annual column rule) + company-identity rule + advisory industry
   context (`agent/industry.py::build_industry_context` — SIC code/description from
   `companies.industry`, injected on the extraction pass); the system prompt also
   teaches the agent to read segment/brand revenue from the narrative
   "Business Segment Results" section (prose, mixed units) rather than only
   the income-statement table
4. `run_agent_loop` (ReAct; `agent/loop.py`) until `finalize_extraction` —
   **open-ended, no step cap**; fallback recovers a final JSON blob from the last
   AI message. In-loop the agent uses `detect_scale()`/`detect_currency()` per
   table, `map_concept()` for label mismatches, and `calculate()`/`compute()` for
   derived arithmetic; the finalize JSON carries `__scale__`, `__currency__` +
   `__currency_evidence__`, `__company_name__` (identity cross-check),
   `__evidence__` (per-value `{lines, scale, currency}`), `__missing__`,
   `__derived__` (computed keys), and an optional `__company_mismatch__` flag
5. `_parse_llm_response` applies the `__scale__` multiplier — **never** scales keys
   matching percentage/per-share/share-count regexes and **never** touches any
   `__*` metadata key (currency/evidence/company fields pass through verbatim)
6. `map_concepts` — Tier 0: `[taxonomy_key]` bracket key or raw `taxonomy_key`;
   Tier 1: exact or whitespace-normalized label. Then
   `semantically_map_unmapped_metrics` — one best-effort LLM repair pass resolves
   leftover numeric keys to concept IDs by meaning (values never change; a
   high-confidence-only resolver, absent/uncertain → stays in `missing_*`)
7. Per-value evidence → `value_metadata_by_id` (`source_lines`, per-value
   `scale`/`currency` from `__evidence__`, dimension identity, calculated status)
8. `build_calc_derivation_block` (`agent/derive.py`) — renders CALC
   (`system:`/`calculated`) concepts to the agent as COMPUTE-ONLY (marked "not
   printed in the filing"), each with its rollup formula: Gross Profit =
   Revenue − |Cost of Revenue| (sign convention taught: CoR is stored negative,
   so 15,400 − (−6,798) = 22,198 is WRONG, 8,602 is right; margins/ratios are
   excluded from the GP shortcut), every other parent = sum of its children.
   The agent computes them in-loop with `compute()` and reports them under
   their bracketed keys, listing them in `__derived__`; the pipeline marks
   those concept_ids as derived (`derived_concept_ids`).  Paths with
   **multiple same-path parent rows** are ambiguous: the block gives no sum
   formula there and the path is surfaced as a `hierarchy_ambiguity` finding +
   `ambiguous_paths` state (observability)
9. Company-identity gates: the agent may flag `__company_mismatch__` (hard fail);
   the pipeline additionally runs a deterministic `check_company_identity()` on
   the agent-reported `__company_name__` (normalized token overlap — a disjoint
   set means a wrong-document upload, e.g. a Netflix letter fed with ticker ORCL)
10. Observability: `missing_concept_labels` / `missing_toplevel_labels` /
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
`get_company_info` (cached SIC profile from state, DB fallback — advisory only),
`detect_currency` (currency detection over a line range, backing the agent's
`__currency__` report), `detect_scale` (scale declaration over a line range, backing
per-value `__evidence__`), `map_concept` (map a filing row label to a concept key
in-loop), `compute` (same exact arithmetic as `calculate`, for derived concept
values). `calculate`/`compute` share one plain-function evaluator (both are
StructuredTools). Tool results truncated to 8000 chars.

### Concept lookup & fiscal math (`integrations/normalize.py`)

- `get_statement_concepts(cik, statement_types, period=DetectedPeriod)` — returns `{_id, concept,
  label, path, statement_type, taxonomy_key, dimension, dimension_concept, calculated,
  dimension_member, dimension_member_label}`, sorted by `(path, order_key)`; labels
  disambiguated on collision; taxonomy keys path-qualified.
- Collection routing is DRY: `_concepts_collection(period)` →
  `normalized_concepts_{quarterly|annual}` and `_values_collection(period)` →
  `concept_values_{quarterly|annual}` are the single routing helpers used by every
  concept/value read & write (concept queries, recent-window, existence/delete,
  prior-value loads, and the upsert). Every production read/write API receives
  the explicit period-agent type or the canonical `DetectedPeriod` object; there
  is no quarter-only or default-quarter routing. Fiscal-year math lives in
  the period agent's returned `fiscal_year` and `quarter`; the display label
  `FY2026 Q3` / `FY2026 (annual)` is formatted by one shared
  `agent/period.py::format_period_label(period)` — no node recomputes
  period identity. All downstream period consumers call `require_detected_period(state)` and
  use the canonical `detected_period` record; no node reads duplicate scalar
  period fields or EDGAR/report dates for period identity.
- `upsert_concept_values` — accepts the canonical `DetectedPeriod`, routes
  quarterly/annual from its agent decision, and replaces the period's docs
  **atomically** (all values upserted first with a unique per-save
  `save_token`, then a stale sweep `delete_many`s only docs of that exact
  fiscal period not carrying the token — a failed write never leaves the
  period empty); persists `calculated`, `currency`, per-value `scale` +
  `source_lines` evidence, `dimension_value` + `dimension_member`/
  `dimension_member_label`/`dimension_axis` (from `value_metadata_by_id`),
  and `accession_number` (traceability only — never used for checks/dedup).
  The old `delete_fiscal_period` delete-before-write helper was removed.

## Code map

```
src/earnings_agents/
  graph.py, hooks.py, state.py, config.py, llm.py, registry.py, progress.py, filelog.py
  agent/        period.py · pipeline.py · loop.py · tools.py · prompts.py · derive.py · industry.py · currency.py · scale.py
  nodes/        fetch.py · check.py · concepts.py · detect.py · save.py
  integrations/ edgar.py · normalize.py · mongo.py · redis.py · http.py · html.py · playwright.py
  cli/          earnings.py · worker.py · failures.py
```

- `llm.py` — provider factory (`build_llm` → `invoke(str)->str` for the
  semantic-mapping pass; `build_chat_llm` → `bind_tools()` for the agent loops)
- `hooks.py` — `with_hooks` + per-thread callbacks (`report_call` drives CLI/worker progress)
- `progress.py` — `WorkerProgressPublisher` (Redis pub/sub `sec:worker:events`), heartbeat
- `filelog.py` — `RunLogFile`: every admin-panel event line is ALSO appended to
  `Logs/<TICKER>_<YYYY-MM-DD>_<HH-MM-SS>.log` (one per run, machine-local time;
  `RUN_LOGS_ENABLED`/`RUN_LOGS_DIR` to disable/relocate; auto-`mkdir`s the dir on
  every run; Docker mounts the repo root at `/project` with `RUN_LOGS_DIR=/project/Logs`
  so a deleted `Logs/` is re-created automatically, `TZ` env passthrough)
- `registry.py` — CIK/ticker lookup from `data/reference/sec_company_tickers.json` (24 h disk cache)
- `integrations/edgar.py` — submissions API → 8-K Item 2.02 → filing index → EX-99.1 URLs;
  `get_latest_earnings_url` returns `(url, supplemental, accession, exhibits)`;
  the period agent alone reads the document for reporting-period identity; token bucket (`EDGAR_RATE_LIMIT`, default 8 req/s); retry on 429/5xx
- `integrations/mongo.py` — raw earnings collection (`earnings_db.earnings`)

## Guardrails & invariants — do not break

- **Period comes from the period agent or the run fails.** No regex/filename/
  EDGAR metadata/cadence inference exists anywhere in the codebase.
- **Extraction target = recent concepts prioritized, never excluded.** The full
  eligible concept universe is loaded; the recent-value window (last
  `PROMPT_HISTORY_PERIODS` periods) is a *prioritization* signal, not an
  eligibility filter. The target is recent concepts ∪ all dimensional (segment/
  breakdown) rows ∪ `system:`/`calculated` concepts, so new segments and newly
  disclosed rows stay extractable even without history. `system:`/`calculated`
  concepts are always loaded as CALC derivation targets — rendered compute-only
  to the agent, never extracted verbatim. No history → full-universe bootstrap.
- **Currency is agent-decided, USD-only persisted.** There is no deterministic
  whole-document currency decision. The extraction agent inspects each
  table/section with the `detect_currency()` tool and reports `__currency__`
  (plus `__currency_evidence__`); a deterministic scan is only a fallback for a
  *confirmed* foreign/mixed code when the agent reports nothing. Non-USD
  monetary values are never saved as USD (no invented FX); confirmed foreign/
  mixed currency blocks the save, and `unknown` is not treated as a blocker.
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
  agent via `compute()` (tracked in `derived_concept_ids`)
- **Save gate** — `STRICT_ACCURACY` (default on) refuses the upsert on unresolved
  high-severity findings; a confirmed non-USD currency additionally fails the
  save unconditionally (hard invariant, not bypassable by relaxed accuracy).
  The CLI's `--allow-inconsistent` flag flips `config.STRICT_ACCURACY = False`
  at runtime and `mongodb_save_node` reads the knob lazily, so the override is
  honored — the currency gate still blocks.
- **Missing metrics never block the save.** Whatever values WERE found are
  always persisted — absence-only findings (agent `missing_concept`) are
  excluded from the save gate, so a few
  not-found metrics can never drop the whole period. Only findings that
  corrupt the values being stored block: wrong value/scale/currency, wrong
  segment parent, wrong-company document, truncated exhibit, non-USD currency.
- **Atomic replace (no delete-before-write window)** — `check_period` only
  *schedules* `_pending_replace` (informational) when the exact fiscal period
  (annual → annual collection, quarterly → quarterly collection) already
  exists. The replace itself happens inside `upsert_concept_values`: every
  value is upserted first (unique per-save `save_token`), then a stale sweep
  deletes only docs of that exact period not carrying the token. A failed/
  interrupted write never leaves the period empty — the previous run's docs
  stay until the new ones are fully written. No accession checks anywhere —
  re-runs always replace the same exact period; a quarterly Q4 record is never
  checked or deleted by annual processing.
- **Single-pass extraction (no verifier, no retry)** — the pipeline runs exactly
  one extraction-agent pass per filing. There is no independent second-read and
  no re-extraction loop: metrics that are not present in the filing are simply
  recorded as `missing_concept` observability and never retried. The save gate
  still enforces the deterministic guards produced by the single pass — non-USD
  currency, wrong-company document (identity cross-check), and truncated/
  incomplete exhibits — but it cannot catch wrong-value/scale/currency or wrong
  segment-parent mistakes that the extraction agent itself does not catch.
- **Company identity is cross-checked deterministically.** The extraction agent
  reports `__company_name__` (and may flag `__company_mismatch__`); the
  pipeline runs `check_company_identity()` (normalized token overlap) on the
  agent-reported name. A mismatch hard-fails the run — a manual filing URL must
  never save another company's numbers under the target ticker.
- **Per-value evidence travels with every value.** The finalize contract carries
  `__evidence__` (`{lines, scale, currency}` per metric key); the pipeline
  stores it as `value_metadata_by_id.source_lines`/`scale`/`currency`, persists
  it for traceability.
- **Manual filing URLs (admin panel)** — the worker honors a `filing_url` in the
  queue payload (press-release HTML or **PDF shareholder letter**; passed through
  from `POST /api/sec-rss/trigger-filing` → `sec:filings:8k`). When present, the
  EDGAR lookup is skipped inside `_build_8k_state`; no filing date is passed to
  the worker or used for period identity. No accession is stamped for manual
  URLs (informational only). PDFs use
  a minimal static-asset request first (important for CDNs such as Adobe that
  stall on a spoofed Chrome UA), then curl, then native-fingerprint Playwright
  as fallbacks; they are converted via pdfplumber with the same caps, headers,
  and `document_map` contract as HTML exhibits.
- **Scale handling** — deterministic document pre-scan + `__scale__` field; parser
  refuses to scale percentages, per-share values, and share counts — incl.
  CamelCase taxonomy keys (`margin`/`yield`/`growth`/`ratio`/`eps`/`per share`…,
  observed live: `custom:NetInterestMarginCompanyProvided` stored 2,080,000 for
  a 2.08% yield). The pipeline additionally passes `build_no_scale_keys()`
  (label-derived from the concept list) so member-tagged per-share keys whose
  as-is signal lives in the LABEL — e.g. `[custom:Basic|014.001]`
  "Basic (Earnings per ordinary share)" — are never scaled either (observed
  live: EPS 4.85 stored as 4,850,000 on PDD). Ratio matching is word-bounded so
  "Operating expenses" (contains "ration") still scales.
- **Sign handling (SEC/IR convention)** — parenthesized amounts are negative:
  `_parse_llm_response` first runs an unconditional `_coerce_number` pass that
  turns `"(175,685)"`, `"-175,685"`, `"$1,234"`, `"($1,234)"`, `"(-667,172)"`
  into correctly signed floats (parens → `-abs`), always before scaling so
  signs are never lost or doubled; strings that are not numbers stay untouched
  downstream. The agent's `calculate` tool converts a bare parenthesized number
  to `(0-n)` so `calculate("(175,685)")` → -175685 (operator groups like
  `(a - b)` are untouched). The extraction prompt instructs the agent that
  parenthesized/leading-minus rows are negative, that expense/loss rows
  (interest expense, income tax provision) are usually shown parenthesized, and
  to never copy a sign from another column.
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
(no LangChain chat model for google-genai); with Gemini configured, period detection,
extraction both fail → the run fails (worker retries are the only
net). The derive/semantic-mapping passes (`build_llm`) still work.

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
| `PROMPT_HISTORY_PERIODS` | `3` | Window (stored periods) used as an extraction *prioritization* signal, not an eligibility filter |
| `RUN_LOGS_ENABLED` | `1` | Write per-run admin-panel-mirror log files (`Logs/<date-time>.log`) |
| `RUN_LOGS_DIR` | `Logs` | Directory for the per-run log files (Docker sets `/app/Logs`) |
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
  geography members as income rows, and page-number rows leaking into the hierarchy
  path (bare `404`/`555` segments). Recent-period pruning masks most of it for
  established companies; bootstrap companies (no history) see the full polluted list.
  Page-number path rows are excluded at fetch time (`get_statement_concepts`).
- **`--allow-inconsistent` is wired to the real config knob.** The CLI flag
  sets `config.STRICT_ACCURACY = False` at runtime; `mongodb_save_node` reads
  the knob lazily from the config module, so the override is honored (the
  USD-only currency gate is intentionally NOT bypassable).
