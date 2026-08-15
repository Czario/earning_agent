"""Save-gate tests: mongodb_save_node refuses to save when accuracy problems
remain unresolved — bad run = nothing saved, old data untouched.

The gate fires BEFORE any DB access (and before the deferred-replace delete),
so these tests need no MongoDB.
"""
from __future__ import annotations

from earnings_agents.graph import mongodb_save_node


def _save_state(findings=None, concept_metrics=None) -> dict:
    return {
        "ticker": "TEST",
        "company_name": "Test Co",
        "discovered_file_url": None,
        "file_type": "html",
        "raw_text": "x",
        "metrics": {"__period__": "Three Months Ended June 30, 2026",
                    "Total revenue": 100.0},
        "error": None,
        "status": "extracted",
        "extraction_attempts": 1,
        "extraction_notes": None,
        "needs_reextract": False,
        "cik": "000123",
        "fiscal_year_end_month": 12,
        "fiscal_year_end_code": "1231",
        "sec_report_date": "2026-06-30",
        "concept_metrics": concept_metrics if concept_metrics is not None
                           else {"507f1f77bcf86cd799439011": 100.0},
        "findings": findings,
    }


def test_save_refused_on_high_identity_finding():
    state = _save_state(findings=[{
        "type": "identity_violation",
        "severity": "high",
        "message": "Net Income (15,000) reconciles with neither Pre-tax − Tax nor Pre-tax + Tax.",
    }])
    out = mongodb_save_node(state)
    assert out["status"] == "failed"
    assert "Refusing to save" in out["error"]
    assert "high-severity" in out["error"]


def test_save_not_blocked_by_medium_or_low_findings():
    # medium/low findings (missing_expected, case duplicates, etc.) must not
    # block the save — only high severity gates.
    state = _save_state(
        findings=[
            {"type": "missing_expected", "severity": "medium", "message": "x"},
            {"type": "case_duplicate", "severity": "low", "message": "y"},
        ],
        concept_metrics={},   # empty → save node takes the skip path (no DB)
    )
    out = mongodb_save_node(state)
    assert out["status"] == "saved"


def test_save_proceeds_when_clean():
    state = _save_state(concept_metrics={})   # empty → skip path, no DB touched
    out = mongodb_save_node(state)
    assert out["status"] == "saved"
