# SEC 8-K Earnings Agent Pipeline

Agent-based pipeline that fetches SEC 8-K filings and press releases (HTML and PDF exhibits), extracts financial statements into MongoDB `normalize_data`, and derives period and Q4 metrics.

---

## Quick Start

```bash
# Install dependencies
uv sync
uv sync --extra dev

# Run extraction for a ticker (extracts ALL 3 statements by default)
uv run earnings --ticker HIMS

# Run dry run (connectivity & filing check, no LLM execution)
uv run earnings --ticker HIMS --dry-run

# Run with verbose debug logging
uv run earnings --ticker HIMS -v
```

---

## Financial Statement Selection (`--statements` / `-s`)

The pipeline supports extracting three financial statements:
- **`income`** — Income Statement / Condensed Consolidated Statements of Operations (Duration/Flow)
- **`balancesheet`** — Balance Sheet / Condensed Consolidated Balance Sheets (Point-in-Time)
- **`cashflow`** — Statement of Cash Flows (Duration/Flow & Point-in-Time ending balances)

### Default Behavior (When Flag is Omitted)

> **When you do NOT pass the `--statements` / `-s` flag, the pipeline automatically extracts and saves ALL THREE statements:**
>
> `["income", "balancesheet", "cashflow"]`

```bash
# Extracts Income Statement, Balance Sheet, and Cash Flow Statement by default:
uv run earnings --ticker MSFT
uv run earnings --cik 0000789019
```

---

### How to Pass Specific Statements

You can pass a comma-separated list of statement identifiers using either `--statements` or `-s`.

#### 1. Single Statement
To restrict extraction to only one statement:

```bash
# Extract only the Income Statement:
uv run earnings --ticker MSFT --statements income
# or using short flag:
uv run earnings --ticker MSFT -s income

# Extract only the Balance Sheet:
uv run earnings --ticker MSFT -s balancesheet

# Extract only the Cash Flow Statement:
uv run earnings --ticker MSFT -s cashflow
```

#### 2. Subset of Statements
To extract any two statements, separate them with a comma:

```bash
# Extract Income Statement and Balance Sheet:
uv run earnings --ticker MSFT --statements "income,balancesheet"

# Extract Income Statement and Cash Flow:
uv run earnings --ticker MSFT -s "income,cashflow"
```

#### 3. Explicitly Specify All Statements
```bash
uv run earnings --ticker MSFT --statements "income,balancesheet,cashflow"
```

*Note: Statement names are case-insensitive and whitespace-tolerant (e.g., `-s "Income , BalanceSheet"` is valid).*

---

### Safe Stale Sweep & Data Isolation

When running with a subset of statements (e.g. `--statements income`):
- The agent only targets concepts belonging to the specified statement(s).
- **Existing database records for omitted statements are protected:** The atomic stale sweep only deletes stale documents for the targeted statement types. If a balance sheet was previously saved for that period, running `--statements income` will **never** delete or overwrite the existing balance sheet data.

---

### Worker & Background Processing

#### Redis Worker CLI
The Redis worker (`earnings-8k-worker`) also supports the `--statements` / `-s` flag to set the default statements for incoming jobs:

```bash
uv run earnings-8k-worker --statements "income,balancesheet,cashflow"
```

#### Job Queue Payload
When jobs are enqueued via Redis (`sec:filings:8k`), the job payload can specify the statements:

```json
{
  "ticker": "MSFT",
  "accession_number": "0000789019-26-000012",
  "statements": ["income", "balancesheet"]
}
```
If the payload omits `statements`, it automatically falls back to the worker's `--statements` flag or `TARGET_STATEMENTS` in configuration (all three statements).

---

### Environment Variable Configuration

You can also set the default statements globally in your `.env` file:

```dotenv
# Comma-separated list of statements (default: income,balancesheet,cashflow)
TARGET_STATEMENTS=income,balancesheet,cashflow
```

---

## Testing

Run the full test suite (250 tests):

```bash
uv run pytest tests/ -q
```