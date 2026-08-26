"""Unit tests for the LLM-backed section locator (agent/indexer.py + find_sections tool)."""
from __future__ import annotations

import unittest

from earnings_agents.agent.indexer import (
    _parse_index_json,
    build_section_index,
    format_section_index,
)
from earnings_agents.agent.tools import build_pi_tools


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompt = None

    def invoke(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


class TestParseIndexJson(unittest.TestCase):
    def test_strips_fences_flat_form(self):
        out = _parse_index_json('```json\n{"coverage": "full", "income_statement": [285, 474]}\n```')
        self.assertEqual(out["coverage"], "full")
        self.assertEqual(out["sections"][0]["name"], "income_statement")
        self.assertEqual(out["sections"][0]["lines"], [285, 474])

    def test_legacy_list_form_still_accepted(self):
        out = _parse_index_json(
            '{"coverage": "full", "sections": [{"name": "income_statement", "lines": [285, 474]}]}'
        )
        self.assertEqual(out["sections"][0]["name"], "income_statement")

    def test_coerces_line_numbers_and_sorts(self):
        out = _parse_index_json(
            '{"coverage": "full", "income_statement": [474.0, 285.0], "eps_data": [5, 2]}'
        )
        self.assertEqual(out["sections"][0]["lines"], [285, 474])
        self.assertEqual(out["sections"][1]["lines"], [2, 5])

    def test_drops_sections_without_valid_range(self):
        out = _parse_index_json(
            '{"coverage": "full", "income_statement": [1], "guidance": null, '
            '"cash_flow": "x", "notes": [100, 50], "eps_data": [2, 9]}'
        )
        self.assertEqual([s["name"] for s in out["sections"]], ["notes", "eps_data"])

    def test_unparseable_returns_none(self):
        self.assertIsNone(_parse_index_json("not json at all"))

    def test_unknown_coverage_normalized(self):
        out = _parse_index_json('{"coverage": "everything"}')
        self.assertEqual(out["coverage"], "unknown")
        self.assertEqual(out["sections"], [])


class TestBuildSectionIndex(unittest.TestCase):
    def test_passes_numbered_text_and_parses(self):
        fake = _FakeLLM(
            '{"coverage": "full", "income_statement": [3, 8], '
            '"segment_results": [1, 2]}'
        )
        index, elapsed = build_section_index("a\nb\nc\nd\ne\nf\ng\nh", llm=fake)
        self.assertIn("  3: c", fake.prompt)  # numbered text with 1-based line numbers
        self.assertEqual(index["sections"][0]["name"], "income_statement")
        self.assertEqual(index["sections"][0]["lines"], [3, 8])
        self.assertEqual(index["sections"][1]["name"], "segment_results")
        self.assertGreaterEqual(elapsed, 0)

    def test_query_hint_included_in_prompt(self):
        fake = _FakeLLM('{"coverage": "full", "sections": []}')
        build_section_index("a\nb", llm=fake, query="where is EPS?")
        self.assertIn("where is EPS?", fake.prompt)

    def test_marks_partial_when_truncated(self):
        fake = _FakeLLM('{"coverage": "full"}')
        index, _ = build_section_index("x\n" * 5000, llm=fake, max_chars=200)
        self.assertEqual(index["coverage"], "partial")

    def test_unparseable_response_degrades_to_empty_map(self):
        fake = _FakeLLM("sorry, no json")
        index, _ = build_section_index("a\nb", llm=fake)
        self.assertEqual(index["sections"], [])
        self.assertEqual(index["coverage"], "unknown")


class TestFormatSectionIndex(unittest.TestCase):
    def test_formats_sections_with_ranges_and_metadata(self):
        out = format_section_index(
            {
                "coverage": "full",
                "sections": [
                    {
                        "name": "income_statement",
                        "label": "Consolidated Statements of Operations",
                        "lines": [153, 460],
                        "scale": "millions",
                        "currency": "USD",
                    }
                ],
            }
        )
        self.assertIn("income_statement", out)
        self.assertIn("lines 153-460", out)
        self.assertIn("millions", out)
        self.assertIn("USD", out)


class TestFindSectionsTool(unittest.TestCase):
    def test_prebuilt_sections_returned_instantly(self):
        prebuilt = {
            "coverage": "full",
            "sections": [{"name": "income_statement", "lines": [1, 5]}],
        }
        tools = build_pi_tools("line1\nline2", {}, prebuilt_sections=prebuilt)
        out = next(t for t in tools if t.name == "find_sections").invoke({})
        self.assertIn("income_statement", out)
        self.assertIn("lines 1-5", out)

    def test_builds_lazily_caches_and_writes_store(self):
        calls = []

        def _builder(text, query=None):
            calls.append(1)
            return {
                "coverage": "full",
                "sections": [{"name": "income_statement", "lines": [1, 2]}],
            }, 0.01

        store = {}
        tools = build_pi_tools("a\nb", {}, section_store=store, section_builder=_builder)
        tool = next(t for t in tools if t.name == "find_sections")
        out1 = tool.invoke({})
        out2 = tool.invoke({"query": "where is EPS?"})
        self.assertEqual(len(calls), 1)  # cached after first build
        self.assertIn("income_statement", out1)
        self.assertIn("income_statement", out2)
        self.assertIn("index", store)  # persisted for the extraction pass
        self.assertEqual(store["index"]["sections"][0]["name"], "income_statement")

    def test_builder_failure_returns_navigation_guidance(self):
        def _builder(text, query=None):
            raise RuntimeError("boom")

        tools = build_pi_tools("a\nb", {}, section_builder=_builder)
        out = next(t for t in tools if t.name == "find_sections").invoke({})
        self.assertIn("search()", out)
        self.assertIn("read_lines", out)


class TestIndexerProviderRouting(unittest.TestCase):
    def test_routed_provider_unavailable_falls_back(self):
        from unittest import mock

        calls = []

        def _fake_build_llm(**kwargs):
            calls.append(kwargs)
            if kwargs.get("provider"):
                raise ValueError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
            return _FakeLLM('{"coverage": "full"}')

        with mock.patch(
            "earnings_agents.llm.build_llm", side_effect=_fake_build_llm
        ), mock.patch(
            "earnings_agents.config.INDEX_LLM_PROVIDER", "gemini"
        ), mock.patch(
            "earnings_agents.config.INDEX_LLM_MODEL", "gemini-2.5-flash"
        ):
            index, _ = build_section_index("a\nb")

        # first tried the routed provider, then fell back to the default
        self.assertEqual(calls[0]["provider"], "gemini")
        self.assertEqual(calls[0]["model"], "gemini-2.5-flash")
        self.assertIsNone(calls[1].get("provider"))  # default provider
        self.assertEqual(index["sections"], [])


if __name__ == "__main__":
    unittest.main()
