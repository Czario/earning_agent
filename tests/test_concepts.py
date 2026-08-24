"""Unit tests for concept-list rendering and calculated-concept exclusion."""
from __future__ import annotations

import unittest

from earnings_agents.agent.prompts import build_concept_list
from earnings_agents.integrations.normalize import _is_polluted_path
from earnings_agents.nodes.concepts import _is_garbage_concept


def _concept(_id, label, concept, taxonomy_key=None, calculated=False, dimension=False):
    return {
        "_id": _id,
        "label": label,
        "concept": concept,
        "taxonomy_key": taxonomy_key or concept,
        "calculated": calculated,
        "dimension": dimension,
        "dimension_concept": False,
    }


class TestBuildConceptList(unittest.TestCase):
    def test_excludes_system_and_calculated_concepts(self):
        concepts = [
            _concept("1", "Revenue", "us-gaap:Revenues"),
            _concept("2", "Gross Profit", "system:GrossProfit"),
            _concept("3", "EBIT", "custom:EBIT", calculated=True),
            _concept("4", "Cloud revenue", "us-gaap:Revenues", dimension=True),
        ]
        out = build_concept_list(concepts)
        self.assertIn("us-gaap:Revenues", out)
        self.assertIn("Cloud revenue", out)
        self.assertNotIn("system:GrossProfit", out)
        self.assertNotIn("custom:EBIT", out)

    def test_tags_dimensional_rows_as_segment(self):
        concepts = [
            _concept("1", "Cloud revenue", "us-gaap:Revenues", dimension=True),
        ]
        out = build_concept_list(concepts)
        self.assertIn("SEGMENT", out)

    def test_keeps_recent_and_nonrecent_ordering_agnostic(self):
        # build_concept_list renders whatever target it is given; the recent
        # filter is applied upstream, not here.
        concepts = [_concept("1", "Revenue", "us-gaap:Revenues")]
        self.assertIn("Revenue", build_concept_list(concepts))


if __name__ == "__main__":
    unittest.main()


class TestGarbageConceptFilter(unittest.TestCase):
    """Malformed rows from upstream normalizer pollution (label "404" — a
    page/footnote number, observed live on PDD) must never reach the target
    list: no filing prints them."""

    def test_numeric_only_labels_are_garbage(self):
        self.assertTrue(_is_garbage_concept({"label": "404"}))
        self.assertTrue(_is_garbage_concept({"label": "1,234.5"}))
        self.assertTrue(_is_garbage_concept({"label": ""}))
        self.assertTrue(_is_garbage_concept({"label": "-"}))

    def test_real_labels_are_kept(self):
        self.assertFalse(_is_garbage_concept({"label": "Revenue"}))
        self.assertFalse(_is_garbage_concept({"label": "Basic"}))
        self.assertFalse(_is_garbage_concept(
            {"label": "Earnings per ADS (4 ordinary shares equal 1 ADS)"}
        ))


class TestPollutedPathFilter(unittest.TestCase):
    """Concepts whose path carries a bare page-number segment (404/555) are
    skipped at fetch time — they are page/status numbers that leaked into the
    normalizer, never real income-statement rows."""

    def test_bare_page_numbers_are_polluted(self):
        self.assertTrue(_is_polluted_path("404"))
        self.assertTrue(_is_polluted_path("555"))

    def test_page_number_segments_are_polluted(self):
        self.assertTrue(_is_polluted_path("001.404.002"))
        self.assertTrue(_is_polluted_path("404.001"))
        self.assertTrue(_is_polluted_path("001.555.002"))

    def test_real_paths_are_kept(self):
        self.assertFalse(_is_polluted_path(""))
        self.assertFalse(_is_polluted_path("001.002.003"))
        self.assertFalse(_is_polluted_path("1404"))
        self.assertFalse(_is_polluted_path("4040"))
        self.assertFalse(_is_polluted_path("001.4040.002"))
