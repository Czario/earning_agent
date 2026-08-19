from __future__ import annotations

from typing import Optional

from typing_extensions import NotRequired, TypedDict


class EarningsAgentState(TypedDict):
    ticker: str
    company_name: str
    discovered_file_url: Optional[str]
    supplemental_file_urls: NotRequired[Optional[list[str]]]
    file_type: Optional[str]   # "pdf" | "html"
    raw_text: Optional[str]
    metrics: Optional[dict]    # serialised EarningsMetrics
    error: Optional[str]
    # pending → discovered → fetched → text_extracted → extracted → saved | failed
    status: str
    # Agentic loop fields
    extraction_attempts: int          # incremented before each extraction pass; caps retries
    extraction_notes: Optional[str]   # reflection output: hints for the next extraction pass
    # Routing signal emitted by analyze_metrics_node. True = loop back to
    # agent_document_pipeline. Using a dedicated field avoids overloading the
    # status field as a routing signal.
    needs_reextract: bool
    # Snapshot of high-severity finding messages from the previous analysis pass.
    # Used by analyze_metrics_node to detect no-progress loops (same findings
    # across consecutive passes → break early rather than burn remaining attempts).
    previous_high_finding_keys: Optional[list]

    # Keys dropped by the agent's cleanup. Informational.
    cleanup_removed: Optional[list]
    # Structured Finding.to_dict() entries produced by analyze_metrics_node.
    # Drives the re-extract loop.
    findings: Optional[list]
    # Per-pass skill-effectiveness records appended by analyze_metrics_node on
    # each re-extract loop: {"to_attempt": int, "deltas": [...]}. Pure
    # observability (ADR-0006) — shows which skills' findings were resolved
    # between passes; never influences routing.
    skill_effectiveness: NotRequired[Optional[list]]
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
    # 8-K filing date ("YYYY-MM-DD") from EDGAR submissions — sanity-window
    # anchor for the period agent's output.
    filing_date: NotRequired[Optional[str]]
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
    # Stored by agent pipeline; consumed by analyze_metrics_node
    # to generate targeted retry hints.
    missing_concept_labels: NotRequired[Optional[list[str]]]   # all unmapped
    missing_segment_labels: NotRequired[Optional[list[str]]]   # dimensional (dimension_concept) only
    missing_toplevel_labels: NotRequired[Optional[list[str]]]  # non-dimensional only
    # Labels of target_concepts that had no value mapped after all tiers.
    # Stored by agent pipeline; consumed by analyze_metrics_node
    # to generate targeted retry hints.
    missing_concept_labels: NotRequired[Optional[list[str]]]   # all unmapped
    missing_segment_labels: NotRequired[Optional[list[str]]]   # dimensional only
    missing_toplevel_labels: NotRequired[Optional[list[str]]]  # non-dimensional only
    # Populated by the agent pipeline when HTML tables are extracted.
    raw_sections: NotRequired[Optional[dict]]
    # Per-metric chunk provenance (informational).
    chunk_metric_sources: NotRequired[Optional[dict]]  # str → list[int]
    # Per-metric verbatim source snippets.
    # Consumed by ``check_source_grounding`` in analyze_metrics_node to flag
    # values that cannot be grounded in the source document.
    metric_source_snippets: NotRequired[Optional[dict]]  # str → str
    # ── Deferred replace (internal) ─────────────────────────────────────────
    # Set by check_period_node when the period already exists in
    # normalize_data.  mongodb_save_node deletes old data before upserting
    # fresh data, ensuring no data loss if the pipeline fails mid-run.
    _pending_replace: NotRequired[Optional[dict]]  # {"cik"}; period is canonical detected_period
    _replace_period_label: NotRequired[Optional[str]]  # human-readable period label
    # ── Multi-exhibit documents ──────────────────────────────────────────
    # Filing exhibits as resolved by EDGAR: [{exhibit: "EX-99.1",
    # description: "The Press Release", url}] in filing-index order.
    exhibit_meta: NotRequired[Optional[list]]
    # Boundaries of each exhibit inside raw_text:
    # [{exhibit, url, line_start, line_end, truncated, skipped, error}]
    document_map: NotRequired[Optional[list]]
    # ── SEC accession number ────────────────────────────────────────────────
    # Set from the EDGAR submissions API (CLI path) or Redis payload (worker).
    # Stored with every concept value for exact-filing dedup in the skip guard.
    accession_number: NotRequired[Optional[str]]
