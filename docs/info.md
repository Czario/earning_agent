  This service is an automated, agent-driven SEC earnings extraction and normalization pipeline
  built with Python, LangGraph, MongoDB, and Redis.

  ### Core Features & Capabilities

  • SEC Filing Ingestion (8-K & 6-K): Fetches US (8-K) and foreign private issuer (6-K) earnings
  filings, handling multi-exhibit bundles across HTML and PDF formats (press releases,
  presentations, shareholder letters).
  • Dual-Agent Architecture:
      • Period Agent: Parses document headers to determine reporting period metadata (fiscal year,
      quarter, period end date).
      • Extraction Agent: Navigates filing text to extract financial line items with per-value
      citations, scale (thousands/millions/billions), and currency evidence.
  • Financial Statement Coverage & Concept Mapping:
      • Extracts Income Statement, Balance Sheet, and Cash Flow metrics.
      • Maps reported labels to standard US-GAAP/custom taxonomy concepts and handles
      dimensional/segment data.
  • Accounting Logic & Derivations:
      • Calculated Rollups: Derives computed concepts (e.g., Gross Profit) via in-loop arithmetic.
      • Cash Flow De-accumulation: Converts YTD figures (6M / 9M) into quarterly flow metrics.
      • Automatic Q4 Derivation: Computes Q4 quarterly income statement values from annual and
      Q1–Q3 filings.
  • Forward-Looking Guidance: Extracts management guidance/outlook metrics, stores normalized
  future ranges, and automatically scores guidance against actuals once the period reports.
  • Data Integrity & Guardrails:
      • Company identity cross-checks to prevent mismatching documents.
      • Accounting identity verifications (Assets = Liabilities + Equity, Gross Profit = Rev − CoR,
      Cash Flow reconciliation).
      • Atomic period replacement and strict validation gates in MongoDB.
  • Flexible Execution: Runs either via command-line interface (CLI) for ad-hoc/backfill processing
  or as an asynchronous Redis worker in Docker.