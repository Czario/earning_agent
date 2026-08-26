from __future__ import annotations

from typing import Optional

from typing_extensions import NotRequired, TypedDict


class EarningsAgentState(TypedDict):
    ticker: str
    company_name: str
    discovered_file_url: Optional[str]
    # Local PDF path used for admin uploads. Local files are read directly;
    # they must never go through HTTP/curl/Playwright URL fetching.
    local_filing_path: NotRequired[Optional[str]]
    supplemental_file_urls: NotRequired[Optional[list[str]]]
    file_type: Optional[str]   # "pdf" | "html"
    raw_text: Optional[str]
    metrics: Optional[dict]    # serialised EarningsMetrics
    error: Optional[str]
    # pending → discovered → fetched → text_extracted → extracted → saved | failed
    status: str
    # Structured completeness findings: [{type, severity, message, evidence}].
    # Populated by the agent pipeline (currency, incomplete exhibits, missing
    # concepts, hierarchy ambiguity) and consumed by mongodb_save — unresolved
    # high-severity findings refuse the upsert under STRICT_ACCURACY.
    findings: Optional[list]

    # ── normalize_data targeted extraction ──────────────────────────────────
    # Populated by load_company_concepts_node when EARNINGS_SAVE_TARGET=normalize_data.
    # Empty list (not None) means the node ran but the company was not found,
    # triggering the generic extraction path.
    cik: NotRequired[Optional[str]]
    company_industry: NotRequired[Optional[dict]]  # {sic_code, sic_description} from normalize_data.companies
    target_concepts: NotRequired[Optional[list]]    # concept dicts from normalized_concepts_quarterly
    # concept_id strings (subset of target_concepts) that had a value in the
    # last N stored periods. Used to prune the extraction prompt to concepts the
    # company actually reports. Empty/None means no pruning (bootstrap / disabled).
    recent_concept_ids: NotRequired[Optional[list[str]]]
    calculated_concepts: NotRequired[Optional[list]]  # system:/calculated concept dicts for derivation
    concept_metrics: NotRequired[Optional[dict]]    # concept_id → float for normalize_data upsert
    derived_concept_ids: NotRequired[Optional[list[str]]]  # concept_ids filled by Tier-3 derivation
    fiscal_year_end_month: NotRequired[Optional[int]]
    fiscal_year_end_code: NotRequired[Optional[str]]  # raw MMDD string, e.g. "0130" or "1231"
    # ── Agent-detected reporting period (agent/period.py) ──────────────────
    # The reporting period is decided ONLY by the period agent reading the
    # filing document (Q4 == annual).  This is the sole period state record;
    # downstream code consumes it through require_detected_period().
    # {period_type, period_end, quarter, period_label, fiscal_year}
    detected_period: NotRequired[Optional[dict]]
    # Keys in metrics{} that were successfully matched to a concept_id during
    # extraction (Tier 0/1). Populated by agent pipeline.
    mapped_metric_keys: NotRequired[Optional[list[str]]]
    # Labels of target_concepts that had no value mapped after all tiers.
    # Stored by agent pipeline; consumed by the save/completeness gate.
    missing_concept_labels: NotRequired[Optional[list[str]]]   # all unmapped
    missing_segment_labels: NotRequired[Optional[list[str]]]   # dimensional only
    missing_toplevel_labels: NotRequired[Optional[list[str]]]  # non-dimensional only
    # Canonical currency metadata.  The extraction agent is the authority
    # (reports __currency__ per table via detect_currency); the deterministic
    # whole-document scan is only a fallback for confirmed foreign/mixed codes.
    # Drives the USD-only save gate.
    currency_metadata: NotRequired[Optional[dict]]
    # Per-concept value metadata (concept_id → dict): dimension flags/member
    # identity, currency, scale, source line evidence, calculated status, and
    # extraction status.  Built by the agent pipeline from the agent's
    # __evidence__ block and consumed by mongodb_save / upsert_concept_values.
    value_metadata_by_id: NotRequired[Optional[dict]]
    # Hierarchy paths that have multiple same-path parent rows.  Auto-derivation
    # refuses to attach children there; surfaced as observability.
    ambiguous_paths: NotRequired[Optional[list[str]]]
    # ── Deferred replace (informational) ──────────────────────────────────
    # Set by check_period_node when the exact fiscal period already exists.
    # mongodb_save_node performs the replace ATOMICALLY inside
    # upsert_concept_values (write-first + stale sweep) — this flag only
    # drives messaging ("replacing X").
    _pending_replace: NotRequired[Optional[dict]]  # {"cik"}; period is canonical detected_period
    _replace_period_label: NotRequired[Optional[str]]  # human-readable period label
    # ── Multi-exhibit documents ──────────────────────────────────────────
    # Filing exhibits as resolved by EDGAR: [{exhibit: "EX-99.1",
    # description: "The Press Release", url}] in filing-index order.
    exhibit_meta: NotRequired[Optional[list]]
    # Boundaries of each exhibit inside raw_text:
    # [{exhibit, url, line_start, line_end, truncated, skipped, error}]
    document_map: NotRequired[Optional[list]]
    # LLM-built section map from the period pass's find_sections call
    # (agent/indexer.py): {coverage, summary, sections: [{name, label,
    # lines, scale, currency, note}]}.  Consumed by the extraction pass so its
    # first read_lines goes straight to the income-statement range.
    document_sections: NotRequired[Optional[dict]]
    # ── SEC accession number ────────────────────────────────────────────────
    # Set from the EDGAR submissions API (CLI path) or Redis payload (worker).
    # Stored with every concept value for exact-filing dedup in the skip guard.
    accession_number: NotRequired[Optional[str]]
