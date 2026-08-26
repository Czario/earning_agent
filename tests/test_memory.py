"""Unit tests for learned memory (label_alias + layout) with format guards."""
from __future__ import annotations

import unittest
from unittest import mock

from earnings_agents.agent import memory as mem


class _MemoryCol:
    """Stateful stand-in for the ``agent_memory_company`` collection."""

    def __init__(self):
        self._docs: dict = {}

    def find_one(self, query, *args, **kwargs):
        return self._docs.get(query.get("cik"))

    def update_one(self, query, update, upsert=False, *args, **kwargs):
        self._docs[query["cik"]] = update["$set"]
        return mock.Mock()


class _FakeDB(dict):
    def __getitem__(self, key):
        return dict.get(self, key, _MemoryCol())


def _fake_db(col=None):
    return _FakeDB({mem._COMPANY_COL: col or _MemoryCol()})


class TestRememberAlias(unittest.TestCase):
    def test_records_valid_alias_minimal_and_dedupes(self):
        col = _MemoryCol()
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            out = mem._remember_alias(
                "1", "Fair value adjustment of liabilities",
                "Change in fair value of liabilities",
            )
            self.assertIn("Remembered", out)
            doc = col._docs["1"]
            self.assertEqual(len(doc["learnings"]), 1)
            self.assertEqual(set(doc["learnings"][0].keys()), {"id", "type", "text"})
            self.assertEqual(doc["learnings"][0]["type"], "label_alias")

            out2 = mem._remember_alias(
                "1", "Fair value adjustment of liabilities",
                "Change in fair value of liabilities",
            )
            self.assertIn("Already remembered", out2)
            self.assertEqual(len(col._docs["1"]["learnings"]), 1)

    def test_skips_identical_labels(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            out = mem._remember_alias("1", "United States Revenue", "United States Revenue")
            self.assertIn("Skipped", out)
            self.assertIn("identical", out)

    def test_skips_member_taxonomy_noise(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            out = mem._remember_alias("1", "United States Revenue (segment → country:US)",
                                      "United States Revenue")
            self.assertIn("Skipped", out)
            self.assertIn("member", out.lower())
            # taxonomy key inside the label
            out2 = mem._remember_alias("1", "Revenue [us-gaap:NonUsMember]",
                                       "Revenue")
            self.assertIn("Skipped", out2)

    def test_rejects_empty(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            self.assertIn("Error", mem._remember_alias("1", "", "Revenue"))
            self.assertIn("Error", mem._remember_alias("1", "Revenue", ""))


class TestRememberLayout(unittest.TestCase):
    def test_records_stable_layout_fact(self):
        col = _MemoryCol()
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            out = mem._remember_layout(
                "1", "Income statement is in the second half of the release."
            )
            self.assertIn("Remembered", out)
            l = col._docs["1"]["learnings"][0]
            self.assertEqual(l["type"], "layout")

    def test_rejects_near_duplicate_layout_note(self):
        col = _MemoryCol()
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            out1 = mem._remember_layout(
                "1",
                "Income statement is in the press release (Exhibit 99.1) after "
                "the balance sheet; revenue by geography (US/Rest of World) is "
                "in a separate earlier table.",
            )
            self.assertIn("Remembered", out1)
            out2 = mem._remember_layout(
                "1",
                "Income statement is in the press release (Exhibit 99.1) after "
                "the balance sheet; revenue by geography (United States / Rest "
                "of the World) is in a separate earlier table near the top.",
            )
            self.assertIn("Already remembered (similar)", out2)
            self.assertEqual(len(col._docs["1"]["learnings"]), 1)

    def test_different_layout_notes_both_kept(self):
        col = _MemoryCol()
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            mem._remember_layout("1", "Income statement is in Exhibit 99.2 (supplemental).")
            mem._remember_layout("1", "Financial tables are near the top of the release.")
            self.assertEqual(len(col._docs["1"]["learnings"]), 2)

    def test_rejects_long_rephrased_near_duplicate(self):
        # The observed failing case: two long layout notes, same fact rephrased
        # + extended (Jaccard ~0.56 but containment high).
        col = _MemoryCol()
        note1 = (
            "Income statement (Condensed Consolidated Statements of Operations "
            "and Comprehensive (Loss) Income) is in the EX-99.1 earnings "
            "release, ~30% in, after the Key Business Metrics/revenue table; "
            "revenue breakdown (US vs Rest of the World) is in a separate "
            "table near the top. Footnote below income statement shows "
            "stock-based compensation by expense line."
        )
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            # note1 is too long — must be rejected as a paragraph
            out1 = mem._remember_layout("1", note1)
            self.assertIn("Skipped", out1)
            self.assertIn("too long", out1)
            self.assertEqual(col._docs.get("1", {}).get("learnings", []), [])

    def test_short_near_duplicate_rejected_by_similarity(self):
        col = _MemoryCol()
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            out1 = mem._remember_layout(
                "1", "Income statement is in EX-99.1, about 30% in after the revenue table."
            )
            self.assertIn("Remembered", out1)
            out2 = mem._remember_layout(
                "1", "Income statement is in the EX-99.1 press release, roughly 30% in after the revenue breakdown table."
            )
            self.assertIn("Already remembered (similar)", out2)
            self.assertEqual(len(col._docs["1"]["learnings"]), 1)

    def test_rejects_per_filing_line_numbers(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            out = mem._remember_layout(
                "1", "Income statement at lines ~273-420 in this filing."
            )
            self.assertIn("Skipped", out)
            self.assertIn("filing-specific", out)

    def test_rejects_empty(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            self.assertIn("Error", mem._remember_layout("1", "  "))


class TestTools(unittest.TestCase):
    def test_tools_are_callable(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            alias_tool = mem.build_remember_alias_tool("1")
            self.assertEqual(alias_tool.name, "remember_alias")
            self.assertIn("Remembered", alias_tool.invoke({
                "filing_label": "Net sales", "canonical": "Revenue",
            }))

            layout_tool = mem.build_remember_layout_tool("1")
            self.assertEqual(layout_tool.name, "remember_layout")
            self.assertIn("Remembered", layout_tool.invoke({
                "note": "Income statement is in Exhibit 99.2 (supplemental).",
            }))


class TestLearningKey(unittest.TestCase):
    def test_short_stable_hash_id(self):
        l = {"type": "layout", "text": "Income statement is in Exhibit 99.2."}
        key = mem._learning_key(l)
        self.assertTrue(key.startswith("layout:"))
        self.assertEqual(key, mem._learning_key(dict(l)))


class TestDetectorFacts(unittest.TestCase):
    def test_remember_currency_single(self):
        col = _MemoryCol()
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            out = mem._remember_currency("1", "USD")
            self.assertIn("Remembered", out)
            l = col._docs["1"]["learnings"][0]
            self.assertEqual(l["type"], "currency")
            self.assertEqual(l["text"], "Reports in USD.")

    def test_remember_currency_with_other_codes(self):
        col = _MemoryCol()
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            out = mem._remember_currency("1", "USD", "EUR,GBP")
            self.assertIn("Remembered", out)
            text = col._docs["1"]["learnings"][0]["text"]
            self.assertIn("Reports in USD", text)
            self.assertIn("EUR", text)
            self.assertIn("GBP", text)

    def test_remember_currency_dedupes_primary_and_unknown(self):
        col = _MemoryCol()
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            # passing USD in other_codes is ignored; junk codes dropped
            out = mem._remember_currency("1", "USD", "USD,eur,xx")
            self.assertIn("Remembered", out)
            text = col._docs["1"]["learnings"][0]["text"]
            self.assertIn("EUR", text)
            self.assertNotIn("xx", text)
            self.assertNotIn("GBP", text)

    def test_remember_currency_rejects_bad_code(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            self.assertIn("Error", mem._remember_currency("1", "US"))
            self.assertIn("Error", mem._remember_currency("1", ""))

    def test_remember_period_derives_stable_prefix(self):
        col = _MemoryCol()
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            out = mem._remember_period(
                "1", "quarterly", "Three Months Ended June 30, 2026"
            )
            self.assertIn("Remembered", out)
            l = col._docs["1"]["learnings"][0]
            self.assertEqual(l["type"], "period")
            self.assertIn("quarterly", l["text"])
            self.assertIn("Three Months Ended", l["text"])
            # the per-filing date is NOT stored
            self.assertNotIn("June 30, 2026", l["text"])

    def test_remember_period_skips_invalid(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            self.assertIn("Skipped", mem._remember_period("1", "bogus", "label"))

    def test_tools_are_callable(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            cur_tool = mem.build_remember_currency_tool("1")
            self.assertEqual(cur_tool.name, "remember_currency")
            self.assertIn("Remembered", cur_tool.invoke({"currency": "USD"}))

            per_tool = mem.build_remember_period_tool("1")
            self.assertEqual(per_tool.name, "remember_period")
            self.assertIn("Remembered", per_tool.invoke({
                "period_type": "quarterly",
                "period_label": "Three Months Ended June 30, 2026",
            }))


class TestRecall(unittest.TestCase):
    def test_block_includes_aliases_and_layout_with_never_skip_warning(self):
        col = _MemoryCol()
        col._docs["1"] = {
            "cik": "1",
            "learnings": [
                {"id": "a", "type": "label_alias",
                 "text": '"Revenue" is labeled "Net sales"'},
                {"id": "b", "type": "layout",
                 "text": "Income statement is in Exhibit 99.2."},
                {"id": "c", "type": "currency", "text": "Reports in USD."},
            ],
        }
        with mock.patch.object(mem, "_get_db", return_value=_fake_db(col)):
            block = mem.recall_memory("1")["local_block"]
            self.assertIn("Net sales", block)
            self.assertIn("Exhibit 99.2", block)
            self.assertIn("Reports in USD", block)
            self.assertIn("NEVER skip", block)
            self.assertIn("FASTER and MORE ACCURATELY", block)

            # type filter: only layout returned
            only_layout = mem.recall_memory("1", types={"layout"})["local_block"]
            self.assertIn("Exhibit 99.2", only_layout)
            self.assertNotIn("Net sales", only_layout)
            self.assertNotIn("Reports in USD", only_layout)

    def test_empty_block_without_memory(self):
        with mock.patch.object(mem, "_get_db", return_value=_fake_db()):
            self.assertEqual(mem.recall_memory("1")["local_block"], "")


if __name__ == "__main__":
    unittest.main()
