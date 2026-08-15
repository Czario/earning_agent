"""Unit tests for normalize_data_client helpers (no DB required)."""
import pytest
from unittest.mock import MagicMock, patch
from earnings_agents.integrations.normalize import (
    _clean_label,
    _extract_member_tag,
    compute_fiscal_period,
    detect_period_type,
    parse_period_end_date,
    parse_period_start_date,
)
from datetime import date


# ── _clean_label ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected_head, expected_member", [
    # Plain label — no member
    ("Net sales", "Net sales", ""),
    # Dimensional label with member concept appended after blank lines
    ("Net sales\n\n\nus-gaap:ProductMember", "Net sales", "Product"),
    # Label that IS only a concept reference — should produce empty head
    ("us-gaap:Revenues", "", ""),
    # Whitespace-only head
    ("  \n\nus-gaap:Revenues", "", ""),
])
def test_clean_label(raw, expected_head, expected_member):
    head, member = _clean_label(raw)
    assert head == expected_head
    assert member == expected_member


# ── _extract_member_tag ───────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected_tag", [
    # Dimensional label
    ("Net sales\n\n\nus-gaap:ProductMember", "us-gaap:ProductMember"),
    # No member
    ("Net sales", ""),
    # Plain concept with no Member suffix — not a member tag
    ("us-gaap:Revenues", ""),
    # Multiple references — first Member wins
    ("Fee income\n\nus-gaap:MembershipMember\nus-gaap:ProductMember", "us-gaap:MembershipMember"),
])
def test_extract_member_tag(raw, expected_tag):
    assert _extract_member_tag(raw) == expected_tag


# ── detect_period_type ────────────────────────────────────────────────────────

@pytest.mark.parametrize("period_str, expected", [
    # Quarterly
    ("Three Months Ended March 31, 2026",       "quarterly"),
    ("Thirteen Weeks Ended May 3, 2025",         "quarterly"),
    ("Quarter Ended June 30, 2025",              "quarterly"),
    ("Three months ended January 31, 2026",      "quarterly"),
    # Annual
    ("Year Ended December 31, 2025",             "annual"),
    ("Twelve Months Ended December 31, 2025",    "annual"),
    ("52 Weeks Ended February 1, 2025",          "annual"),
    ("53 Weeks Ended February 3, 2024",          "annual"),
    ("Full Year Ended January 28, 2023",         "annual"),
    ("Annual Period Ended June 30, 2025",        "annual"),
])
def test_detect_period_type(period_str, expected):
    assert detect_period_type(period_str) == expected


# ── parse_period_end_date ─────────────────────────────────────────────────────

@pytest.mark.parametrize("period_str, expected", [
    ("Three Months Ended March 31, 2026",  date(2026, 3, 31)),
    ("Year Ended December 31, 2025",       date(2025, 12, 31)),
    ("Thirteen Weeks Ended May 3, 2025",   date(2025, 5, 3)),
    ("no date here",                       None),
])
def test_parse_period_end_date(period_str, expected):
    assert parse_period_end_date(period_str) == expected


# ── compute_fiscal_period ─────────────────────────────────────────────────────

@pytest.mark.parametrize("end_date, fy_end_month, expected_fy, expected_q", [
    # MSFT (June FY end): March 31 2026 → FY2026 Q3
    (date(2026, 3, 31), 6, 2026, 3),
    # MSFT: June 30 2026 → FY2026 Q4
    (date(2026, 6, 30), 6, 2026, 4),
    # MSFT: September 30 2025 → FY2026 Q1
    (date(2025, 9, 30), 6, 2026, 1),
    # Calendar FY (December): December 31 2025 → FY2025 Q4
    (date(2025, 12, 31), 12, 2025, 4),
    # Calendar FY: March 31 2026 → FY2026 Q1
    (date(2026, 3, 31), 12, 2026, 1),
    # BJ (Jan FY end): January 31 2026 → FY2026 Q4
    (date(2026, 1, 31), 1, 2026, 4),
    # BJ: November 1 2025 → FY2026 Q4
    (date(2025, 11, 1), 1, 2026, 4),
])
def test_compute_fiscal_period(end_date, fy_end_month, expected_fy, expected_q):
    fy, q = compute_fiscal_period(end_date, fy_end_month)
    assert fy == expected_fy
    assert q == expected_q


# ── compute_fiscal_period WITH period_str (52-week + explicit durations) ─────

@pytest.mark.parametrize("period_str, end_date, fy_end_month, expected_fy, expected_q", [
    # BJ (Jan FY end, 52-week calendar) — "Thirteen Weeks" is NOT inferred as Q1
    # (ambiguous — non-cumulative reporters use it for every quarter).
    # Calendar math with fy_end_month=1, date May 2 → month offset 3 → Q2.
    ("Thirteen Weeks Ended May 2, 2026",        date(2026,  5,  2), 1, 2027, 2),
    ("Twenty-Six Weeks Ended August 1, 2026",   date(2026,  8,  1), 1, 2027, 2),
    ("Thirty-Nine Weeks Ended November 1, 2025",date(2025, 11,  1), 1, 2026, 3),
    # Digit forms
    ("13 Weeks Ended May 2, 2026",              date(2026,  5,  2), 1, 2027, 2),
    ("26 Weeks Ended August 1, 2026",           date(2026,  8,  1), 1, 2027, 2),
    ("39 Weeks Ended November 1, 2025",         date(2025, 11,  1), 1, 2026, 3),
    # Month-based (Six/Nine unambiguous; Three falls back to date math)
    ("Six Months Ended June 30, 2026",          date(2026,  6, 30), 12, 2026, 2),
    ("Nine Months Ended September 30, 2026",    date(2026,  9, 30), 12, 2026, 3),
    # Ordinal labels — always trusted (explicit, not inferred)
    ("First Quarter Ended March 31, 2026",      date(2026,  3, 31), 12, 2026, 1),
    ("Second Quarter Ended June 30, 2026",      date(2026,  6, 30), 12, 2026, 2),
    ("Q3 Results Ended September 30, 2026",     date(2026,  9, 30), 12, 2026, 3),
    # MSFT (June FY end) with explicit period
    ("Nine Months Ended March 31, 2026",        date(2026,  3, 31),  6, 2026, 3),
    # SMPL (Aug FY end) — "Thirteen Weeks" is ambiguous, falls back to calendar math
    # May 30 → month offset 8 (Sep start) → Q3 (NOT Q1)
    ("Thirteen Weeks Ended May 30, 2026",       date(2026,  5, 30),  8, 2026, 3),
    ("Three Months Ended May 30, 2026",         date(2026,  5, 30),  8, 2026, 3),
    # SMPL Q1 — calendar math: Nov → month offset 2 → Q1
    ("Thirteen Weeks Ended November 30, 2025",  date(2025, 11, 30),  8, 2026, 1),
    ("Three Months Ended November 30, 2025",    date(2025, 11, 30),  8, 2026, 1),
])
def test_compute_fiscal_period_with_period_str(
    period_str, end_date, fy_end_month, expected_fy, expected_q
):
    fy, q = compute_fiscal_period(end_date, fy_end_month, period_str)
    assert fy == expected_fy, f"FY mismatch for {period_str!r}: got {fy}, want {expected_fy}"
    assert q == expected_q, f"Q mismatch for {period_str!r}: got {q}, want {expected_q}"


# ── parse_period_start_date ───────────────────────────────────────────────────

@pytest.mark.parametrize("period_str, end_date, expected_start", [
    # Week-based (13w × 7 = 91 days back + 1)
    ("Thirteen Weeks Ended May 2, 2026",         date(2026, 5, 2),  date(2026, 2, 1)),
    ("Twenty-Six Weeks Ended August 1, 2026",    date(2026, 8, 1),  date(2026, 2, 1)),
    ("Thirty-Nine Weeks Ended November 1, 2026", date(2026, 11, 1), date(2026, 2, 2)),
    # Month-based
    ("Three Months Ended March 31, 2026",        date(2026, 3, 31), date(2026, 1, 1)),
    ("Six Months Ended June 30, 2026",           date(2026, 6, 30), date(2026, 1, 1)),
    ("Nine Months Ended September 30, 2026",     date(2026, 9, 30), date(2026, 1, 1)),
    # Cross-year boundary (Q1 for Nov FY-end company: Dec–Feb)
    ("Three Months Ended February 28, 2026",     date(2026, 2, 28), date(2025, 12, 1)),
    # Unrecognised string → None
    ("Year Ended December 31, 2025",             date(2025, 12, 31), None),
    ("No date here",                             date(2026, 1, 1),  None),
])
def test_parse_period_start_date(period_str, end_date, expected_start):
    result = parse_period_start_date(period_str, end_date)
    assert result == expected_start, (
        f"start_date mismatch for {period_str!r}: got {result}, want {expected_start}"
    )


# ── load_company_concepts_node: recent-window query-time filter ──────────────

def test_load_concepts_uses_detected_period_type_from_state():
    """The node does NOT decide the period — it consumes ``detected_period_type``
    set by the period agent (detect_period_node)."""
    from earnings_agents.nodes import concepts as node

    company = {"cik": "000123", "fiscal_year_end_month": 12,
               "fiscal_year_end_code": "1231", "name": "Example Co"}
    state = {"ticker": "EXMP", "sec_report_date": "2025-12-31",
             "detected_period_type": "annual"}

    with (
        patch.object(node, "get_company_by_ticker", return_value=company),
        patch.object(node, "get_recently_valued_concept_ids", return_value={"x"}),
        patch.object(node, "get_statement_concepts", return_value=[]) as concepts_mock,
    ):
        result = node.load_company_concepts_node(state)  # type: ignore[arg-type]

    assert concepts_mock.call_args.kwargs["period_type"] == "annual"
    assert result["detected_period_type"] == "annual"


def _concepts(*ids):
    return [{"_id": i, "concept": f"us-gaap:C{i}", "label": f"Label {i}",
             "path": "001", "statement_type": "income"} for i in ids]


def _company():
    return {"cik": "000123", "fiscal_year_end_month": 12,
            "fiscal_year_end_code": "1231", "name": "Example Co"}


def test_load_concepts_queries_only_recently_valued_concepts():
    """The recent window is computed FIRST and passed as a query-level
    ``concept_ids`` filter — non-recent concepts are never even loaded."""
    from earnings_agents.nodes import concepts as node

    state = {"ticker": "EXMP", "sec_report_date": "2026-07-26",
             "detected_period_type": "quarterly"}
    with (
        patch.object(node, "get_company_by_ticker", return_value=_company()),
        patch.object(node, "get_recently_valued_concept_ids",
                     return_value={"b"}) as recent_mock,
        patch.object(node, "get_statement_concepts",
                     return_value=_concepts("b")) as concepts_mock,
    ):
        result = node.load_company_concepts_node(state)  # type: ignore[arg-type]

    # The concept query itself is restricted to the recent window.
    assert concepts_mock.call_args.kwargs["concept_ids"] == ["b"]
    assert [c["_id"] for c in result["target_concepts"]] == ["b"]
    # The window must be scoped to the detected period type.
    assert recent_mock.call_args.kwargs["period_type"] == "quarterly"
    assert recent_mock.call_args.kwargs["n_periods"] == 3


def test_load_concepts_annual_queries_annual_collection():
    from earnings_agents.nodes import concepts as node

    state = {"ticker": "EXMP", "sec_report_date": "2026-07-25",
             "detected_period_type": "annual"}
    with (
        patch.object(node, "get_company_by_ticker", return_value=_company()),
        patch.object(node, "get_recently_valued_concept_ids",
                     return_value={"a"}) as recent_mock,
        patch.object(node, "get_statement_concepts", return_value=_concepts("a")),
    ):
        node.load_company_concepts_node(state)  # type: ignore[arg-type]

    assert recent_mock.call_args.kwargs["period_type"] == "annual"


def test_load_concepts_skips_when_no_concept_valued_recently():
    """Empty recent set → skip BEFORE any concept loading (no full-list fallback)."""
    from earnings_agents.nodes import concepts as node

    state = {"ticker": "EXMP", "sec_report_date": "2026-07-26",
             "detected_period_type": "quarterly"}
    with (
        patch.object(node, "get_company_by_ticker", return_value=_company()),
        patch.object(node, "get_recently_valued_concept_ids", return_value=set()),
        patch.object(node, "get_statement_concepts") as concepts_mock,
    ):
        result = node.load_company_concepts_node(state)  # type: ignore[arg-type]

    assert result["status"] == "skipped"
    assert "nothing to extract" in result["error"]
    assert result["target_concepts"] == []
    concepts_mock.assert_not_called()


def test_load_concepts_skips_when_recent_lookup_fails():
    from earnings_agents.nodes import concepts as node

    state = {"ticker": "EXMP", "sec_report_date": "2026-07-26",
             "detected_period_type": "quarterly"}
    with (
        patch.object(node, "get_company_by_ticker", return_value=_company()),
        patch.object(node, "get_recently_valued_concept_ids",
                     side_effect=Exception("db down")),
        patch.object(node, "get_statement_concepts") as concepts_mock,
    ):
        result = node.load_company_concepts_node(state)  # type: ignore[arg-type]

    assert result["status"] == "skipped"
    assert "Recent-value lookup failed" in result["error"]
    concepts_mock.assert_not_called()


# ── check_period_node (post-detection existence check) ───────────────────────

from earnings_agents.nodes.check import check_period_node  # noqa: E402


def _check_state(**over):
    base = {
        "ticker": "AMAT", "company_name": "Applied Materials Inc",
        "cik": "0000006951", "fiscal_year_end_month": 10,
        "detected_period_type": "quarterly",
        "sec_report_date": "2026-07-26",
        "period_label": "Three Months Ended July 26, 2026",
        "accession_number": None,
    }
    base.update(over)
    return base


def test_check_period_schedules_deferred_replace_when_period_exists():
    with patch("earnings_agents.nodes.check.fiscal_period_exists", return_value=True):
        out = check_period_node(_check_state())
    assert out["_pending_replace"] == {
        "cik": "0000006951", "fiscal_year": 2026, "quarter": 3,
    }
    assert out["_replace_period_label"] == "FY2026 Q3"


def test_check_period_annual_uses_quarter_none():
    state = _check_state(
        detected_period_type="annual",
        sec_report_date="2026-07-25",
        period_label="Fiscal Year Ended July 25, 2026",
        fiscal_year_end_month=7,
    )
    with patch("earnings_agents.nodes.check.fiscal_period_exists", return_value=True):
        out = check_period_node(state)
    assert out["_pending_replace"]["quarter"] is None
    assert out["_replace_period_label"] == "FY2026 (annual)"


def test_check_period_annual_never_checks_quarterly_q4():
    """Annual processing must not look at (or delete) a quarterly Q4 record for
    the same fiscal year — the exact-period check is annual-only (quarter None)."""
    state = _check_state(
        detected_period_type="annual",
        sec_report_date="2026-07-25",
        period_label="Fiscal Year Ended July 25, 2026",
        fiscal_year_end_month=7,
    )
    with patch("earnings_agents.nodes.check.fiscal_period_exists", return_value=False) as mock:
        out = check_period_node(state)
    # The existence check must be scoped to the annual collection (quarter=None)
    assert mock.call_args.args[2] is None
    assert out.get("_pending_replace") is None


def test_check_period_proceeds_for_new_period():
    with patch("earnings_agents.nodes.check.fiscal_period_exists", return_value=False):
        out = check_period_node(_check_state())
    assert out.get("_pending_replace") is None
    assert out.get("status") != "failed"


# ── get_statement_concepts uses correct collection ───────────────────────────

def test_get_statement_concepts_uses_quarterly_collection_by_default():
    """get_statement_concepts(...) without period_type queries normalized_concepts_quarterly."""
    from earnings_agents.integrations import normalize as ndc

    # find() must return something with a .sort() method that itself is iterable
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = iter([])
    mock_col = MagicMock()
    mock_col.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        ndc.get_statement_concepts("000123", statement_types=["income"])

    called_col = mock_db.__getitem__.call_args[0][0]
    assert called_col == "normalized_concepts_quarterly"


def test_get_statement_concepts_uses_annual_collection_when_specified():
    """get_statement_concepts(..., period_type='annual') queries normalized_concepts_annual."""
    from earnings_agents.integrations import normalize as ndc

    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = iter([])
    mock_col = MagicMock()
    mock_col.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        ndc.get_statement_concepts(
            "000123", statement_types=["income"], period_type="annual"
        )

    called_col = mock_db.__getitem__.call_args[0][0]
    assert called_col == "normalized_concepts_annual"


def test_get_statement_concepts_window_filter_exempts_system_and_calculated():
    """The recent-window `_id $in` must NOT filter out system:/calculated
    concepts — they are always loaded (derivation targets)."""
    from earnings_agents.integrations import normalize as ndc

    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = iter([])
    mock_col = MagicMock()
    mock_col.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        ndc.get_statement_concepts(
            "000123", statement_types=["income"],
            concept_ids=["507f1f77bcf86cd799439011"],
        )

    query = mock_col.find.call_args.args[0]
    assert "$and" in query
    window_or = query["$and"][0]["$or"]
    # All three arms must exist: window ids, system: regex, calculated flag.
    arms = {
        "_id" in arm or "concept" in arm or "calculated" in arm
        for arm in window_or
    }
    assert len(window_or) == 3
    assert any("_id" in arm for arm in window_or)
    assert any(arm.get("concept", {}).get("$regex") == "^system:" for arm in window_or)
    assert any("calculated" in arm for arm in window_or)


def test_get_statement_concepts_without_window_has_no_and_clause():
    """Without concept_ids there is no window restriction at all."""
    from earnings_agents.integrations import normalize as ndc

    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = iter([])
    mock_col = MagicMock()
    mock_col.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        ndc.get_statement_concepts("000123", statement_types=["income"])

    query = mock_col.find.call_args.args[0]
    assert "$and" not in query


def test_get_statement_concepts_keeps_same_label_children_under_distinct_parents():
    """Rows sharing a label (e.g. 'Hardware' under Revenue and Cost of Revenue)
    are NOT deduplicated — each is kept and disambiguated by its parent path,
    and the shared GAAP tag gets a path-qualified taxonomy key so mapping stays
    unambiguous."""
    from earnings_agents.integrations import normalize as ndc

    rows = [
        {"_id": "a", "concept": "us-gaap:Revenue", "label": "Total revenue", "path": "001", "statement_type": "income"},
        {"_id": "b", "concept": "us-gaap:ProductMember", "label": "Hardware", "path": "001.001", "statement_type": "income"},
        {"_id": "c", "concept": "us-gaap:CostOfRevenue", "label": "Cost of Revenue", "path": "002", "statement_type": "income"},
        {"_id": "d", "concept": "us-gaap:ProductMember", "label": "Hardware", "path": "002.001", "statement_type": "income"},
        {"_id": "e", "concept": "us-gaap:ServiceMember", "label": "Cloud and other services", "path": "001.002", "statement_type": "income"},
        {"_id": "f", "concept": "us-gaap:ServiceMember", "label": "Cloud and other services", "path": "002.002", "statement_type": "income"},
    ]

    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = iter(rows)
    mock_col = MagicMock()
    mock_col.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        result = ndc.get_statement_concepts("000123", statement_types=["income"])

    labels = [c["label"] for c in result]
    keys = [c["taxonomy_key"] for c in result]

    # Both Hardware rows survive, disambiguated by their parent section.
    assert "Hardware (Total revenue)" in labels
    assert "Hardware (Cost of Revenue)" in labels
    assert "Cloud and other services (Total revenue)" in labels
    assert "Cloud and other services (Cost of Revenue)" in labels
    # The shared GAAP tag is path-qualified so JSON keys never collide.
    assert len(keys) == len(set(keys))
    assert "us-gaap:ProductMember|001.001" in keys
    assert "us-gaap:ProductMember|002.001" in keys
    assert "us-gaap:ServiceMember|001.002" in keys
    assert "us-gaap:ServiceMember|002.002" in keys


# ── get_latest_period returns correct structure ───────────────────────────────

def test_upsert_concept_values_separates_quarterly_and_annual_filters():
    """Upserts include period-type information so annual vs quarterly rows stay distinct."""
    from datetime import date
    from earnings_agents.integrations import normalize as ndc

    mock_collection = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)

        result = ndc.upsert_concept_values(
            cik="000123",
            company_name="Example Co",
            concept_metrics={"507f1f77bcf86cd799439011": 1.23},
            period_str="Three Months Ended March 31, 2026",
            fiscal_year_end_month=12,
            report_date=date(2026, 3, 31),
        )

    assert result == 1
    ops = mock_collection.bulk_write.call_args[0][0]
    assert len(ops) == 1
    state = ops[0].__getstate__()[1]
    assert state["_filter"] == {
        "cik": "000123",
        "concept_id": ndc.ObjectId("507f1f77bcf86cd799439011"),
        "reporting_period.fiscal_year": 2026,
        "reporting_period.quarter": 1,
    }
    # earning_data flag must be present in the upserted document
    update_doc = state["_doc"]["$set"]
    assert update_doc["earning_data"] is True


def test_upsert_concept_values_uses_annual_collection_for_annual_periods():
    """Annual inserts target the annual collection and keep annual form_type in the filter."""
    from datetime import date
    from earnings_agents.integrations import normalize as ndc

    mock_collection = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)

        ndc.upsert_concept_values(
            cik="000123",
            company_name="Example Co",
            concept_metrics={"507f1f77bcf86cd799439011": 1.23},
            period_str="Year Ended December 31, 2025",
            fiscal_year_end_month=12,
            report_date=date(2025, 12, 31),
        )

    called_collection = mock_db.__getitem__.call_args_list[0][0][0]
    assert called_collection == "concept_values_annual"
    ops = mock_collection.bulk_write.call_args[0][0]
    state = ops[0].__getstate__()[1]
    assert state["_filter"]["reporting_period.fiscal_year"] == 2025
    assert "reporting_period.quarter" not in state["_filter"]
    # earning_data flag must be present; annual reporting_period must have no start_date
    update_doc = state["_doc"]["$set"]
    assert update_doc["earning_data"] is True
    assert "start_date" not in update_doc["reporting_period"]


def test_upsert_concept_values_period_type_override_is_authoritative():
    """An explicit period_type_override wins over month-equality and __period__.

    Guards the single-source-of-truth contract: ``detected_period_type`` from
    the upstream node routes the save collection so the prompt's column
    selection and the persisted period type can never diverge.
    """
    from datetime import date
    from earnings_agents.integrations import normalize as ndc

    mock_collection = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)

        # period_str/report_date both look quarterly, but the override says annual.
        ndc.upsert_concept_values(
            cik="000123",
            company_name="Example Co",
            concept_metrics={"507f1f77bcf86cd799439011": 1.23},
            period_str="Three Months Ended March 31, 2026",
            fiscal_year_end_month=12,
            report_date=date(2026, 3, 31),
            period_type_override="annual",
        )

    called_collection = mock_db.__getitem__.call_args_list[0][0][0]
    assert called_collection == "concept_values_annual"
    ops = mock_collection.bulk_write.call_args[0][0]
    state = ops[0].__getstate__()[1]
    assert state["_filter"]["reporting_period.fiscal_year"] == 2026
    assert "reporting_period.quarter" not in state["_filter"]



def test_get_latest_period_returns_most_recent_across_both_collections():
    """get_latest_period picks the most recent end_date from both collections."""
    from datetime import datetime, timezone
    from earnings_agents.integrations import normalize as ndc

    quarterly_doc = {
        "reporting_period": {
            "end_date": datetime(2025, 9, 30, tzinfo=timezone.utc),
            "fiscal_year": 2026,
            "quarter": 1,
        }
    }
    annual_doc = {
        "reporting_period": {
            "end_date": datetime(2025, 6, 30, tzinfo=timezone.utc),
            "fiscal_year": 2025,
        }
    }

    def make_collection(doc):
        col = MagicMock()
        col.find_one.return_value = doc
        return col

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(side_effect=lambda name: {
        "concept_values_quarterly": make_collection(quarterly_doc),
        "concept_values_annual": make_collection(annual_doc),
    }[name])

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        result = ndc.get_latest_period("000123")

    assert result is not None
    assert result["period_type"] == "quarterly"
    assert result["fiscal_year"] == 2026
    assert result["quarter"] == 1


def test_get_latest_period_returns_none_when_no_data():
    """get_latest_period returns None when both collections have no data for the CIK."""
    from earnings_agents.integrations import normalize as ndc

    mock_col = MagicMock()
    mock_col.find_one.return_value = None
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        result = ndc.get_latest_period("000123")

    assert result is None


def test_build_initial_state_skips_sec_path_without_history():
    """Returns a skipped state without querying EDGAR when no normalize_data history exists."""
    info = {"ticker": "AAPL", "company_name": "Apple Inc", "cik": "0000320193"}

    with (
        patch("earnings_agents.cli.earnings._has_existing_period_data", return_value=False),
        patch("earnings_agents.cli.earnings.get_latest_earnings_url") as mock_url,
    ):
        from importlib import import_module
        mod = import_module("earnings_agents.cli.earnings")
        state = mod._build_initial_state(info, printer=lambda *_: None)

    assert state["status"] == "skipped"
    assert "no existing normalize_data period data" in state["error"]
    mock_url.assert_not_called()


