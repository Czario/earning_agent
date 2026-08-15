"""Regression tests for the concept-flag audit fixes.

Covers three bugs found in the normalize_data concept usage audit:

1. Dimensional detection — ``dimension_concept``/``dimension`` DB flags are
   the authoritative signal; the old ``"|" in taxonomy_key`` proxy fired for
   only 30 of ~109k docs (and also matched path-qualified non-dimensional
   rows).
2. Path collisions — sibling rows can share a ``path`` (disambiguated by
   ``order_key``); head/hierarchy maps must be first-wins so ancestor
   disambiguation and derivation edges are deterministic.
3. Margin concepts — "Gross Profit Margin" must never get the dollar
   "GP = Revenue − CoR" derivation shortcut (observed live: 16,800,000
   stored for system:GrossProfitMargin).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _load_concepts(rows: list[dict]):
    """Run get_statement_concepts against a mocked cursor returning *rows*."""
    from earnings_agents.integrations import normalize as ndc

    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = iter(rows)
    mock_col = MagicMock()
    mock_col.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        result = ndc.get_statement_concepts("000123", statement_types=["income"])
    return result, mock_cursor


# ── Bug #1: dimension flags are the dimensional signal ──────────────────────

def test_dimension_flags_propagate_to_loaded_concepts():
    """dimension/dimension_concept must survive loading (projection + output)."""
    rows = [
        {"_id": "a", "concept": "us-gaap:Revenue", "label": "Total revenue",
         "path": "001", "statement_type": "income",
         "dimension": False, "dimension_concept": False},
        {"_id": "b", "concept": "tsla:AutomotiveSalesMember", "label": "Automotive sales",
         "path": "001.001", "statement_type": "income",
         "dimension": False, "dimension_concept": True},
    ]
    result, _ = _load_concepts(rows)
    flags = {c["_id"]: (c["dimension"], c["dimension_concept"]) for c in result}
    assert flags["a"] == (False, False)
    assert flags["b"] == (False, True)


def test_segment_tag_uses_db_flags_not_pipe_proxy():
    """SEGMENT tag: flag-driven.  A path-qualified '|' key alone is NOT
    dimensional (it also appears on duplicated non-member concepts)."""
    from earnings_agents.agent.prompts import build_concept_list

    concepts = [
        {"_id": "b", "concept": "tsla:AutomotiveSalesMember", "label": "Automotive sales",
         "path": "001.001", "taxonomy_key": "tsla:AutomotiveSalesMember",
         "dimension": False, "dimension_concept": True},
        {"_id": "c", "concept": "us-gaap:ProductMember", "label": "Hardware",
         "path": "002.001", "taxonomy_key": "us-gaap:ProductMember|002.001",
         "dimension": False, "dimension_concept": False},
    ]
    text = build_concept_list(concepts, recent_concept_ids=None, calculated_concepts=None)

    member_line = next(l for l in text.splitlines() if "AutomotiveSalesMember" in l)
    assert "[SEGMENT]" in member_line

    hardware_line = next(l for l in text.splitlines() if "Hardware" in l)
    assert "[SEGMENT]" not in hardware_line


def test_missing_classification_uses_db_flags():
    """missing_toplevel vs missing_segments split follows the DB flags,
    not the '|' in the taxonomy key."""
    from earnings_agents.agent.pipeline import classify_missing_labels

    target_concepts = [
        {"_id": "seg", "label": "Automotive sales",
         "taxonomy_key": "tsla:AutomotiveSalesMember",
         "dimension": False, "dimension_concept": True},
        {"_id": "seg2", "label": "Hardware",
         "taxonomy_key": "us-gaap:ProductMember|002.001",  # '|' but NOT dimensional
         "dimension": False, "dimension_concept": False},
        {"_id": "top", "label": "Revenue", "taxonomy_key": "us-gaap:Revenue",
         "dimension": False, "dimension_concept": False},
    ]
    toplevel, segments = classify_missing_labels(
        {"seg", "seg2", "top"}, target_concepts
    )
    assert toplevel == ["Hardware", "Revenue"]
    assert segments == ["Automotive sales"]


def test_sort_uses_path_then_order_key():
    """Cursor must sort by (path, order_key) so same-path siblings are
    deterministic."""
    rows = [
        {"_id": "a", "concept": "us-gaap:Revenue", "label": "Total revenue",
         "path": "001", "statement_type": "income"},
    ]
    _, mock_cursor = _load_concepts(rows)
    args, _kwargs = mock_cursor.sort.call_args
    assert args[0] == [("path", 1), ("order_key", 1)]


# ── Bug #2: path collisions are first-wins ───────────────────────────────────

def test_hierarchy_first_wins_on_path_collision():
    """Two sibling rows sharing path '014': the child under '014.001' attaches
    to the FIRST concept at '014', deterministically."""
    from earnings_agents.agent.derive import _build_hierarchy

    concepts = [
        {"_id": "id-net", "concept": "us-gaap:NetIncomeLoss", "label": "Net income",
         "path": "014"},
        {"_id": "id-buy", "concept": "custom:LessBuyout", "label": "Less buyout",
         "path": "014"},
        {"_id": "id-child", "concept": "custom:Child", "label": "Child",
         "path": "014.001"},
    ]
    h = _build_hierarchy(concepts)
    assert h["id-net"] == ["id-child"]
    assert "id-buy" not in h


def test_disambiguation_first_wins_on_colliding_path():
    """path_to_head is first-wins: a duplicate-label row under a colliding
    path is qualified by the FIRST row's head at that path."""
    rows = [
        {"_id": "1", "concept": "us-gaap:Revenue", "label": "Revenue",
         "path": "001", "statement_type": "income"},
        {"_id": "2", "concept": "custom:X", "label": "Segmentation",
         "path": "001.001", "statement_type": "income", "order_key": "a"},
        {"_id": "3", "concept": "custom:Y", "label": "Hardware",
         "path": "001.001", "statement_type": "income", "order_key": "b"},
        {"_id": "4", "concept": "custom:Z", "label": "Hardware",
         "path": "001.001.001", "statement_type": "income"},
    ]
    result, _ = _load_concepts(rows)
    labels = [c["label"] for c in result]
    # The nested Hardware is disambiguated by the first row at 001.001.
    assert "Hardware (Segmentation)" in labels


# ── Bug #3: margins never get the dollar GP shortcut ────────────────────────

def test_margin_concepts_excluded_from_gp_shortcut():
    """'Gross Profit Margin' must not be treated as the dollar Gross Profit."""
    from earnings_agents.agent.derive import _build_derivation_prompt

    concepts = [
        {"_id": "m", "concept": "system:GrossProfitMargin",
         "label": "Gross Profit Margin", "path": "020.001",
         "taxonomy_key": "system:GrossProfitMargin"},
        {"_id": "g", "concept": "system:GrossProfit",
         "label": "Gross Profit", "path": "019",
         "taxonomy_key": "system:GrossProfit"},
    ]
    prompt = _build_derivation_prompt({}, concepts)
    # Margin has no children and no GP shortcut → silently omitted (stays
    # underived rather than corrupting the percentage concept with dollars).
    assert "Gross Profit Margin" not in prompt
    # The formula appears once in the template RULES and once for the real
    # "Gross Profit" concept.  Without the fix the margin row adds a third.
    assert prompt.count("Gross Profit = Revenue \u2212 Cost of Revenue") == 2
