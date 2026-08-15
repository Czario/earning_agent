"""Multi-exhibit fetching — filings with several EX-99 exhibits.

fetch_filing must concatenate ALL text exhibits (press release, presentation,
supplemental information) with DOCUMENT headers, record boundaries in
``document_map``, skip non-text exhibits, and honour the size caps.  The
income statement often lives in a supplemental exhibit (e.g. BofA's 99.3).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.nodes import fetch as fetch_mod
from earnings_agents.nodes.fetch import fetch_filing_node

_PRIMARY = "https://www.sec.gov/Archives/edgar/data/70858/000007085826000353/bac06302026ex991.htm"
_EX992 = "https://www.sec.gov/Archives/edgar/data/70858/000007085826000353/bac06302026ex992.htm"
_EX993 = "https://www.sec.gov/Archives/edgar/data/70858/000007085826000353/bac-06302026ex993.htm"

_HTML = {
    _PRIMARY: "<html><body><h1>BofA reports Q2 2026 results</h1><p>Revenue</p></body></html>",
    _EX992: "<html><body><h1>Presentation</h1><p>Slide revenue</p></body></html>",
    _EX993: "<html><body><h1>Supplemental Information</h1><p>Income statement detail</p></body></html>",
}


def _state(**over):
    base = {
        "ticker": "BAC",
        "discovered_file_url": _PRIMARY,
        "supplemental_file_urls": [_EX992, _EX993],
        "exhibit_meta": [
            {"exhibit": "EX-99.1", "description": "The Press Release", "url": _PRIMARY},
            {"exhibit": "EX-99.2", "description": "The Presentation Materials", "url": _EX992},
            {"exhibit": "EX-99.3", "description": "The Supplemental Information", "url": _EX993},
        ],
    }
    base.update(over)
    return base


@pytest.fixture
def fake_http():
    def _get(url, **kwargs):
        resp = MagicMock()
        resp.text = _HTML[url]
        return resp

    with patch.object(fetch_mod, "_http_get", side_effect=_get):
        yield


def test_fetch_concatenates_all_text_exhibits(fake_http):
    out = fetch_filing_node(_state())
    assert out["status"] == "fetched"
    text = out["raw_text"]

    assert "DOCUMENT 1 OF 3" in text and "The Press Release" in text
    assert "DOCUMENT 2 OF 3" in text and "The Presentation Materials" in text
    assert "DOCUMENT 3 OF 3" in text and "The Supplemental Information" in text
    assert text.index("Press Release") < text.index("Presentation") < text.index("Supplemental")

    doc_map = out["document_map"]
    assert len(doc_map) == 3
    for d in doc_map:
        assert not d["skipped"] and d["error"] is None
        assert d["line_start"] is not None and d["line_end"] >= d["line_start"]

    # Boundaries must point at the right content in the combined text.
    lines = text.split("\n")
    third = doc_map[2]
    assert "Supplemental Information" in "\n".join(
        lines[third["line_start"] - 1: third["line_end"]]
    )
    # Searchable content of exhibit 3 is inside its own range.
    joined = "\n".join(lines[third["line_start"] - 1: third["line_end"]])
    assert "Income statement detail" in joined


def test_fetch_skips_non_text_exhibits(fake_http):
    state = _state()
    state["exhibit_meta"].append(
        {"exhibit": "EX-99.4", "description": "Images", "url": "https://sec.gov/x.jpg"}
    )
    out = fetch_filing_node(state)
    doc_map = out["document_map"]
    assert doc_map[3]["skipped"] is True
    assert "non-text exhibit" in doc_map[3]["reason"]
    assert "DOCUMENT 4 OF" not in out["raw_text"]


def test_fetch_truncates_and_exhausts_budget(monkeypatch):
    """Per-exhibit cap truncates long exhibits; the total budget then cuts the
    next exhibit and skips the rest.  Budget counts exhibit TEXT only."""
    monkeypatch.setattr(fetch_mod, "FETCH_EXHIBIT_MAX_CHARS", 50)
    monkeypatch.setattr(fetch_mod, "FETCH_TOTAL_MAX_CHARS", 70)

    def _long(url, **kwargs):
        resp = MagicMock()
        resp.text = "<p>" + ("x" * 200) + "</p>"
        return resp

    with patch.object(fetch_mod, "_http_get", side_effect=_long):
        out = fetch_filing_node(_state())
    doc_map = out["document_map"]
    assert doc_map[0]["truncated"] is True   # 200 → 50 (per-exhibit cap)
    assert doc_map[1]["truncated"] is True   # 200 → 20 (remaining budget)
    assert doc_map[2]["skipped"] is True
    assert "total size budget exhausted" in doc_map[2]["reason"]


def test_fetch_fails_run_when_primary_fetch_fails():
    def _boom(url, **kwargs):
        raise RuntimeError("network down")

    with patch.object(fetch_mod, "_http_get", side_effect=_boom):
        out = fetch_filing_node(_state())
    assert out["status"] == "failed"
    assert "HTML fetch failed" in out["error"]


def test_fetch_records_supplemental_failure_but_continues(fake_http):
    original = fetch_mod._http_get

    def _flaky(url, **kwargs):
        if url == _EX992:
            raise RuntimeError("exhibit 99.2 unavailable")
        return original(url, **kwargs)

    with patch.object(fetch_mod, "_http_get", side_effect=_flaky):
        out = fetch_filing_node(_state())
    assert out["status"] == "fetched"
    doc_map = out["document_map"]
    assert doc_map[1]["skipped"] is True
    assert "fetch failed" in doc_map[1]["reason"]
    assert "Supplemental Information" in out["raw_text"]  # 99.3 still included


def test_get_document_info_lists_exhibit_map(fake_http):
    from earnings_agents.agent.tools import build_pi_tools

    out = fetch_filing_node(_state())
    tools = build_pi_tools(out["raw_text"], prior_values={}, document_map=out["document_map"])
    info_tool = next(t for t in tools if t.name == "get_document_info")
    info = info_tool.invoke({})

    assert "Document bundle:" in info
    assert "Exhibit map:" in info
    assert "EX-99.1" in info and "The Press Release" in info
    assert "EX-99.3" in info and "The Supplemental Information" in info
    assert "lines " in info
