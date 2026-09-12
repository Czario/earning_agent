"""Tests for forward-looking guidance extraction (guidance_values schema).

Covers: form → band normalization, covered-period derivation + future-only
gates, __guidance__ passthrough through the scale-aware finalize parser,
guidance_values upsert semantics (keyed dedup, is_current demotion,
supersedes, source=manual protection), score() outcomes with metric
direction, and the save_guidance node's non-fatal behavior.
"""
from __future__ import annotations

import json
import unittest
from datetime import date
from unittest import mock

from earnings_agents import config as _config
from earnings_agents.agent.guidance import (
    normalize_guidance_records,
    period_arrived,
    score,
)
from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.agent.loop import _parse_llm_response
from earnings_agents.nodes.save_guidance import save_guidance_node


# ── Fake MongoDB (mirrors tests/test_memory.py pattern) ──────────────────────

def _get_path(doc, path):
    cur = doc
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _match(doc, query):
    if not query:
        return True
    for k, v in query.items():
        dv = _get_path(doc, k)
        if isinstance(v, dict):
            if "$ne" in v and dv == v["$ne"]:
                return False
            if "$in" in v and (dv is None or dv not in v["$in"]):
                return False
            continue
        if dv != v:
            return False
    return True


def _set_path(doc, path, value):
    parts = path.split(".")
    cur = doc
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _noop_ok():
    return mock.Mock(modified_count=0, matched_count=0)


class _Result:
    def __init__(self, modified=0, upserted=0):
        self.modified_count = modified
        self.matched_count = modified
        self.upserted_count = upserted


class _FakeCol:
    def __init__(self):
        self._docs = []

    def find(self, query=None, *a, **k):
        return [d for d in self._docs if _match(d, query or {})]

    def find_one(self, query=None, *a, **k):
        for d in self._docs:
            if _match(d, query or {}):
                return d
        return None

    def update_one(self, query, update, upsert=False, *a, **k):
        for d in self._docs:
            if _match(d, query):
                self._apply(d, update, fresh=False)
                return _Result(modified=1)
        if upsert:
            doc = {}
            for k, v in (query or {}).items():
                if isinstance(v, dict) and ("$ne" in v or "$in" in v):
                    continue
                _set_path(doc, k, v)
            self._apply(doc, update, fresh=True)
            doc["_id"] = f"id{len(self._docs) + 1}"
            self._docs.append(doc)
            return _Result(upserted=1)
        return _Result(modified=0)

    def update_many(self, query, update, *a, **k):
        n = 0
        for d in self._docs:
            if _match(d, query):
                self._apply(d, update, fresh=False)
                n += 1
        return _Result(modified=n)

    def delete_many(self, query, *a, **k):
        n = len([d for d in self._docs if _match(d, query)])
        self._docs = [d for d in self._docs if not _match(d, query)]
        return _Result(modified=n)

    @staticmethod
    def _apply(doc, update, fresh):
        if "$set" in update:
            for k, v in update["$set"].items():
                _set_path(doc, k, v)
        if "$setOnInsert" in update and fresh:
            for k, v in update["$setOnInsert"].items():
                if k not in doc:
                    _set_path(doc, k, v)


class _FakeDB(dict):
    def __getitem__(self, key):  # noqa: A003
        if key not in self:
            self[key] = _FakeCol()
        return dict.__getitem__(self, key)


def _fake_db(**cols):
    db = _FakeDB()
    for k, v in cols.items():
        db[k] = v
    return db


def _q_period(fy=2026, quarter=2, period_type="quarterly"):
    return DetectedPeriod(
        period_type=period_type,
        period_end=date(2026, 6, 30) if period_type == "quarterly" else date(2026, 12, 31),
        quarter=quarter,
        fiscal_year=fy,
        period_label="Test period",
    )


# ── Normalization: forms, periods, gates ─────────────────────────────────────

class TestNormalizeForms(unittest.TestCase):
    def test_plus_minus_pct_band(self):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "form": "plus_minus_pct", "value": 108.0,
            "plus_minus": 2, "plus_minus_unit": "percent",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period())
        self.assertEqual(issues, [])
        self.assertEqual(len(recs), 1)
        d = recs[0]
        self.assertAlmostEqual(d["value"], 108.0)
        self.assertAlmostEqual(d["value_low"], 105.84)
        self.assertAlmostEqual(d["value_high"], 110.16)
        self.assertEqual(d["plus_minus_unit"], "percent")
        self.assertEqual(d["currency"], "USD")
        self.assertEqual(d["source"], "llm")
        self.assertEqual(d["standard_label"], "Total Revenues")
        self.assertEqual(d["statement_type"], "income")
        self.assertEqual(d["basis"], "gaap")
        self.assertEqual(d["period"]["label"], "Q3 FY2026")

    def test_range_derives_midpoint(self):
        recs, _ = normalize_guidance_records([{
            "metric": "eps_diluted", "form": "range",
            "value_low": 2.0, "value_high": 2.4,
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period())
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0]["value"], 2.2)
        self.assertEqual(recs[0]["standard_label"], "Earnings Per Share, Diluted")

    def test_min_max_forms(self):
        recs, _ = normalize_guidance_records([
            {"metric": "revenue", "form": "min", "value": 10.0, "as_printed": "at least $10B",
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
            {"metric": "capex", "form": "max", "value": 5.0,
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
        ], _q_period())
        self.assertEqual(len(recs), 2)
        self.assertEqual((recs[0]["value_low"], recs[0]["value_high"]), (10.0, None))
        self.assertEqual((recs[1]["value_low"], recs[1]["value_high"]), (None, 5.0))

    def test_qualitative_kept_without_number(self):
        recs, _ = normalize_guidance_records([{
            "metric": "other", "form": "qualitative",
            "as_printed": "revenue expected to be flat sequentially",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period())
        self.assertEqual(len(recs), 1)
        self.assertIsNone(recs[0]["value"])
        self.assertIsNone(recs[0]["value_low"])
        self.assertEqual(recs[0]["form"], "qualitative")

    def test_percentage_growth_unit(self):
        recs, _ = normalize_guidance_records([{
            "metric": "revenue", "form": "percentage_growth", "value": 12.0,
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period())
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["unit"], "percent")
        self.assertEqual(recs[0]["scale"], "as-is")

    def test_inverted_band_dropped(self):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "form": "range", "value_low": 20.0, "value_high": 10.0,
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period())
        self.assertEqual(recs, [])
        self.assertTrue(any("value_low > value_high" in i for i in issues))

    def test_non_usd_monetary_dropped(self):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "form": "point", "value": 100.0, "currency": "EUR",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period())
        self.assertEqual(recs, [])
        self.assertTrue(any("not USD" in i for i in issues))


    def test_display_string_value_parsed_to_band(self):
        # Model emitted the number as a display string instead of numbers
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "form": "range",
            "value": "$61-64 billion",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(issues, [])
        d = recs[0]
        self.assertAlmostEqual(d["value"], 62.5)
        self.assertAlmostEqual(d["value_low"], 61.0)
        self.assertAlmostEqual(d["value_high"], 64.0)
        self.assertEqual(d["form"], "range")

    def test_percent_display_string_gets_percent_unit(self):
        recs, _ = normalize_guidance_records([{
            "metric": "tax_rate", "value": "between 15-17%",
            "period": {"fiscal_year": 2026, "period_type": "annual"},
        }], _q_period(fy=2026, quarter=2))
        d = recs[0]
        self.assertAlmostEqual(d["value"], 16.0)
        self.assertEqual(d["form"], "range")
        self.assertEqual(d["unit"], "percent")

    def test_dict_form_guidance_merged(self):
        recs, issues = normalize_guidance_records({
            "revenue": {"form": "point", "value": 62.0,
                        "period": {"fiscal_year": 2026, "quarter": 3,
                                   "period_type": "quarterly"}},
            "capex": {"form": "range", "value": 137.5,
                      "value_low": 130.0, "value_high": 145.0,
                      "period": {"fiscal_year": 2026, "period_type": "annual"}},
        }, _q_period(fy=2026, quarter=2))
        self.assertEqual(issues, [])
        self.assertEqual(len(recs), 2)
        by_metric = {r["metric"]: r for r in recs}
        self.assertEqual(by_metric["revenue"]["value"], 62.0)
        self.assertAlmostEqual(by_metric["capex"]["value"], 137.5)

    def test_period_string_label_parsed(self):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "value": 62.0, "period": "Q3 2026",
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(issues, [])
        self.assertEqual(recs[0]["period"]["fiscal_year"], 2026)
        self.assertEqual(recs[0]["period"]["quarter"], 3)

    def test_custom_metric_stored_as_is(self):
        # Unlisted metrics are NOT skipped — stored with a free-text metric
        # name and scored with the default (up) direction.
        recs, issues = normalize_guidance_records([{
            "metric": "net_shipments", "standard_label": "Vehicle Deliveries",
            "value": 480000, "unit": "shares", "scale": "as-is",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(issues, [])
        self.assertEqual(recs[0]["metric"], "net_shipments")
        self.assertEqual(recs[0]["standard_label"], "Vehicle Deliveries")
        # default direction is up for unlisted metrics
        self.assertEqual(score(recs[0], 500000)["outcome"], "beat")

    def test_eps_guidance_as_is(self):
        recs, _ = normalize_guidance_records([{
            "metric": "eps_diluted", "form": "range", "value": 4.80,
            "value_low": 4.70, "value_high": 4.90, "scale": "as-is",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs[0]["scale"], "as-is")
        self.assertEqual(recs[0]["standard_label"], "Earnings Per Share, Diluted")

    def test_new_curated_metrics_have_directions(self):
        from earnings_agents.agent.guidance import metric_profile
        self.assertEqual(metric_profile("ebitda")[2], "up")
        self.assertEqual(metric_profile("adjusted_ebitda")[2], "up")
        self.assertEqual(metric_profile("cash_flow_from_operations")[2], "up")
        self.assertEqual(metric_profile("research_development")[2], "down")
        self.assertEqual(metric_profile("sales_marketing")[2], "down")
        self.assertIsNone(metric_profile("some_random_metric"))


# ── Scale application: guidance values are stored in RAW units ──────────────

class TestGuidanceScaleToRaw(unittest.TestCase):
    """Monetary guidance is stored with zeros (raw units), matching the
    concept_values_* actuals it is scored against; %/per-share/share-count
    values are never scaled."""

    def test_monetary_range_scaled_to_raw(self):
        # The live failure: "We expect third quarter 2026 total revenue to be
        # in the range of $61-64 billion." arrived as value 62.5 / scale
        # billions and was stored unscaled.
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "form": "range", "value": 62.5,
            "value_low": 61.0, "value_high": 64.0, "scale": "billions",
            "unit": "USD", "currency": "USD",
            "as_printed": "We expect third quarter 2026 total revenue to be "
                          "in the range of $61-64 billion.",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(issues, [])
        d = recs[0]
        self.assertEqual(d["value"], 62_500_000_000.0)
        self.assertEqual(d["value_low"], 61_000_000_000.0)
        self.assertEqual(d["value_high"], 64_000_000_000.0)
        # printed unit kept as provenance
        self.assertEqual(d["scale"], "billions")

    def test_point_millions_and_thousands_scaled(self):
        recs, _ = normalize_guidance_records([
            {"metric": "operating_income", "value": 108.0, "scale": "millions",
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
            {"metric": "net_income", "value": 480.0, "scale": "thousands",
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
        ], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs[0]["value"], 108_000_000.0)
        self.assertEqual(recs[1]["value"], 480_000.0)

    def test_plus_minus_abs_scaled_but_pct_not(self):
        recs, _ = normalize_guidance_records([
            {"metric": "revenue", "form": "plus_minus_abs", "value": 100.0,
             "plus_minus": 2, "plus_minus_unit": "absolute", "scale": "billions",
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
            {"metric": "revenue", "form": "plus_minus_pct", "value": 100.0,
             "plus_minus": 2, "plus_minus_unit": "percent", "scale": "billions",
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
        ], _q_period(fy=2026, quarter=2))
        abs_rec, pct_rec = recs
        self.assertEqual(abs_rec["value"], 100_000_000_000.0)
        self.assertEqual(abs_rec["value_low"], 98_000_000_000.0)
        self.assertEqual(abs_rec["value_high"], 102_000_000_000.0)
        self.assertEqual(abs_rec["plus_minus"], 2_000_000_000.0)
        # ± % : band scales, the percentage stays a percentage
        self.assertEqual(pct_rec["value"], 100_000_000_000.0)
        self.assertEqual(pct_rec["plus_minus"], 2)
        self.assertAlmostEqual(pct_rec["value_low"], 98_000_000_000.0)
        self.assertAlmostEqual(pct_rec["value_high"], 102_000_000_000.0)

    def test_eps_and_percent_never_scaled(self):
        recs, _ = normalize_guidance_records([
            {"metric": "eps_diluted", "form": "range", "value": 4.80,
             "value_low": 4.70, "value_high": 4.90, "scale": "as-is",
             "standard_label": "Earnings Per Share, Diluted",
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
            {"metric": "gross_margin", "value": 74.0, "unit": "percent",
             "scale": "as-is",
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
            {"metric": "tax_rate", "value": 16.0, "unit": "percent",
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
            {"metric": "share_count", "value": 480000.0, "unit": "shares",
             "scale": "as-is",
             "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"}},
        ], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs[0]["value"], 4.80)
        self.assertEqual(recs[0]["value_low"], 4.70)
        self.assertEqual(recs[1]["value"], 74.0)
        self.assertEqual(recs[2]["value"], 16.0)
        self.assertEqual(recs[3]["value"], 480000.0)

    def test_model_mistagged_eps_with_monetary_scale_not_scaled(self):
        # Belt-and-braces: EPS guidance tagged scale "billions" by a model must
        # never become 4,850,000,000.
        recs, _ = normalize_guidance_records([{
            "metric": "eps_diluted", "value": 4.85, "scale": "billions",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs[0]["value"], 4.85)

    def test_share_count_scales_with_printed_magnitude(self):
        # "approximately 2.5 billion diluted shares" — share counts DO scale
        # when the filing prints a magnitude; they stay as-is otherwise.
        recs, _ = normalize_guidance_records([{
            "metric": "share_count", "value": 2.5, "unit": "shares",
            "scale": "billions",
            "as_printed": "approximately 2.5 billion diluted shares",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs[0]["value"], 2_500_000_000.0)

    def test_invalid_scale_ignored(self):
        recs, _ = normalize_guidance_records([{
            "metric": "revenue", "value": 62.5, "scale": "zillions",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertIsNone(recs[0]["scale"])
        self.assertEqual(recs[0]["value"], 62.5)

    def test_absurd_magnitude_after_scaling_dropped(self):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "value": 1e12, "scale": "billions",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs, [])
        self.assertTrue(any("implausible" in i for i in issues))


class TestNormalizePeriods(unittest.TestCase):
    def test_explicit_year_wins(self):
        recs, _ = normalize_guidance_records([{
            "metric": "revenue", "value": 1.0,
            "period": {"fiscal_year": 2027, "quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs[0]["period"]["fiscal_year"], 2027)

    def test_next_quarter_default(self):
        recs, _ = normalize_guidance_records([{
            "metric": "revenue", "value": 1.0,
            "period": {"quarter": 3, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs[0]["period"]["fiscal_year"], 2026)
        self.assertEqual(recs[0]["period"]["quarter"], 3)

    def test_q3_to_q4_same_fy(self):
        recs, _ = normalize_guidance_records([{
            "metric": "revenue", "value": 1.0,
            "period": {"quarter": 4, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=3))
        self.assertEqual(recs[0]["period"]["fiscal_year"], 2026)
        self.assertEqual(recs[0]["period"]["quarter"], 4)

    def test_annual_filing_next_fy_q1(self):
        recs, _ = normalize_guidance_records([{
            "metric": "revenue", "value": 1.0,
            "period": {"period_type": "quarterly", "quarter": 1},
        }], _q_period(fy=2026, quarter=None, period_type="annual"))
        self.assertEqual(recs[0]["period"]["fiscal_year"], 2027)
        self.assertEqual(recs[0]["period"]["quarter"], 1)

    def test_quarter_not_after_filing_dropped(self):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "value": 1.0,
            "period": {"fiscal_year": 2026, "quarter": 2, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs, [])
        self.assertTrue(any("not after reported Q2" in i for i in issues))

    def test_same_fy_quarter_from_annual_filing_dropped(self):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "value": 1.0,
            "period": {"fiscal_year": 2026, "quarter": 2, "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=None, period_type="annual"))
        self.assertEqual(recs, [])
        self.assertTrue(any("not future" in i for i in issues))

    def test_same_fy_annual_guidance_from_annual_filing_dropped(self):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "value": 1.0,
            "period": {"fiscal_year": 2026, "period_type": "annual"},
        }], _q_period(fy=2026, quarter=None, period_type="annual"))
        self.assertEqual(recs, [])
        self.assertTrue(any("(annual) not after filing" in i for i in issues))

    def test_same_fy_annual_guidance_from_quarterly_filing_kept(self):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "value": 1.0,
            "period": {"fiscal_year": 2026, "period_type": "annual"},
        }], _q_period(fy=2026, quarter=3))
        self.assertEqual(issues, [])
        self.assertEqual(recs[0]["period"]["period_type"], "annual")
        self.assertIsNone(recs[0]["period"]["quarter"])


# ── Top-level period_type (binary quarterly|annual, derived) ─────────────────

class TestGuidancePeriodType(unittest.TestCase):
    """The doc's top-level ``period_type`` field classifies the covered
    (target) period as "quarterly" or "annual" — derived from the nested
    ``period.period_type`` extended enum, never accepted as raw input."""

    def _one(self, period):
        recs, issues = normalize_guidance_records([{
            "metric": "revenue", "value": 62.0, "period": period,
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(issues, [])
        return recs[0]

    def test_quarterly_target(self):
        d = self._one({"fiscal_year": 2026, "quarter": 3,
                       "period_type": "quarterly"})
        self.assertEqual(d["period_type"], "quarterly")
        self.assertEqual(d["period"]["period_type"], "quarterly")

    def test_annual_target(self):
        d = self._one({"fiscal_year": 2027, "period_type": "annual"})
        self.assertEqual(d["period_type"], "annual")
        self.assertIsNone(d["period"]["quarter"])

    def test_multi_year_maps_to_annual(self):
        d = self._one({"fiscal_year": 2029, "period_type": "multi_year"})
        self.assertEqual(d["period_type"], "annual")
        self.assertEqual(d["period"]["period_type"], "multi_year")

    def test_ytd_maps_to_quarterly(self):
        d = self._one({"fiscal_year": 2026, "period_type": "ytd"})
        self.assertEqual(d["period_type"], "quarterly")

    def test_raw_top_level_input_is_ignored(self):
        """A conflicting raw top-level value never wins — period is the single
        source of truth (derivation, not passthrough)."""
        recs, _ = normalize_guidance_records([{
            "metric": "revenue", "value": 62.0, "period_type": "annual",
            "period": {"fiscal_year": 2026, "quarter": 3,
                       "period_type": "quarterly"},
        }], _q_period(fy=2026, quarter=2))
        self.assertEqual(recs[0]["period_type"], "quarterly")

    def test_string_period_label_classified(self):
        self.assertEqual(self._one("Q3 2026")["period_type"], "quarterly")
        self.assertEqual(self._one("FY2027")["period_type"], "annual")

    def test_default_derivation_from_quarter_presence(self):
        self.assertEqual(self._one({"fiscal_year": 2026, "quarter": 3})["period_type"],
                         "quarterly")
        self.assertEqual(self._one({"fiscal_year": 2027})["period_type"], "annual")


# ── Parser passthrough (metadata keys are never scaled) ──────────────────────

class TestGuidanceParserPassthrough(unittest.TestCase):
    def test_guidance_values_never_scaled(self):
        payload = {
            "__scale__": "millions",
            "us-gaap:Revenues": 1000.0,
            "__guidance__": [
                {"metric": "revenue", "value": 108.0, "value_low": 105.84,
                 "value_high": 110.16,
                 "period": {"fiscal_year": 2027, "quarter": 3, "period_type": "quarterly"}},
            ],
        }
        out = _parse_llm_response(json.dumps(payload), no_scale_keys=set())
        self.assertEqual(out["us-gaap:Revenues"], 1000.0 * 1_000_000)
        g = out["__guidance__"][0]
        self.assertEqual(g["value"], 108.0)         # guidance values unchanged
        self.assertEqual(g["value_low"], 105.84)
        self.assertEqual(g["period"]["fiscal_year"], 2027)


# ── Persistence: upsert, demotion, supersedes, manual protection ────────────

class TestGuidanceUpsert(unittest.TestCase):
    def _guidance_values_col(self):
        db = _fake_db()
        guid = db["guidance_values"]
        guid._docs.append({
            "_id": "old1", "cik": "0001045810", "metric": "revenue", "basis": "gaap",
            "period": {"fiscal_year": 2026, "quarter": 3, "period_type": "quarterly",
                       "label": "Q3 FY2026"},
            "accession_number": "acc-1", "source": "manual", "is_current": True,
        })
        return db["guidance_values"]

    def _record(self, fy=2026, quarter=3, metric="revenue", ptype="quarterly"):
        return {
            "metric": metric, "standard_label": "Total Revenues",
            "statement_type": "income", "basis": "gaap", "form": "point",
            "value": 108.0, "value_low": 108.0, "value_high": 108.0,
            "unit": "USD", "scale": "billions", "currency": "USD",
            "period": {"fiscal_year": fy, "quarter": quarter,
                       "period_type": ptype, "label": f"Q{quarter} FY{fy}"},
            "event_type": "initial",
        }

    def test_upsert_shapes_doc_and_is_idempotent(self):
        from earnings_agents.integrations.guidance import upsert_guidance_records
        col = _FakeCol()
        db = _fake_db(guidance_values=col)
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            s1 = upsert_guidance_records(
                "0001045810", [self._record()], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
            s2 = upsert_guidance_records(
                "0001045810", [self._record()], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
        self.assertEqual(s1["upserted"], 1)
        self.assertEqual(s1["skipped_manual"], 0)
        self.assertEqual(s2["upserted"], 1)     # re-run replaces in place
        self.assertEqual(len(col._docs), 1)
        d = col._docs[0]
        self.assertEqual(d["source"], "llm")
        self.assertEqual(d["filing_period"]["fiscal_year"], 2026)
        self.assertEqual(d["filing_period"]["quarter"], 2)
        self.assertEqual(d["period"]["label"], "Q3 FY2026")
        self.assertEqual(d["period_type"], "quarterly")
        self.assertEqual(d["form_type"], "8-K")
        self.assertIsNone(d["edited_by"])
        self.assertIn("created_at", d)

    def test_upsert_derives_period_type_for_hand_built_record(self):
        """Belt-and-braces: a record WITHOUT the top-level field (hand-built/
        legacy-shaped) still persists the derived quarterly|annual class."""
        from earnings_agents.integrations.guidance import upsert_guidance_records
        col = _FakeCol()
        db = _fake_db(guidance_values=col)
        rec = self._record()
        rec.pop("period_type", None)  # normalized records always carry it
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            s = upsert_guidance_records(
                "0001045810", [rec], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
        self.assertEqual(s["upserted"], 1)
        self.assertEqual(col._docs[0]["period_type"], "quarterly")

    def test_upsert_derives_annual_period_type(self):
        from earnings_agents.integrations.guidance import upsert_guidance_records
        col = _FakeCol()
        db = _fake_db(guidance_values=col)
        rec = self._record(fy=2027, quarter=None, ptype="annual")
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            upsert_guidance_records(
                "0001045810", [rec], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
        self.assertEqual(col._docs[0]["period_type"], "annual")

    def test_demotes_prior_accession_and_sets_supersedes(self):
        from earnings_agents.integrations.guidance import upsert_guidance_records
        col = _FakeCol()
        db = _fake_db(guidance_values=col)
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            upsert_guidance_records(
                "0001045810", [self._record()], accession_number="acc-1",
                filing_period=_q_period(fy=2026, quarter=2),
            )
            upsert_guidance_records(
                "0001045810", [self._record()], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
        docs = {d["accession_number"]: d for d in col._docs}
        self.assertFalse(docs["acc-1"]["is_current"])
        self.assertTrue(docs["acc-2"]["is_current"])
        self.assertEqual(docs["acc-2"]["supersedes"], "acc-1")
        self.assertEqual(docs["acc-2"]["event_type"], "updated")

    def test_manual_protection_skips_llm_write(self):
        from earnings_agents.integrations.guidance import upsert_guidance_records
        col = self._guidance_values_col()  # contains a source=manual doc for the key
        db = _fake_db(guidance_values=col)
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            s = upsert_guidance_records(
                "0001045810", [self._record()], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
        self.assertEqual(s["upserted"], 0)
        self.assertEqual(s["skipped_manual"], 1)
        self.assertEqual(col._docs[0]["source"], "manual")  # untouched

    def test_concept_filled_when_missing(self):
        """concept must never stay null when the company has a matching row:
        resolved deterministically from standard_label via the mapping
        vocabulary (the same rows the backend enrichment joins against)."""
        from earnings_agents.integrations.guidance import upsert_guidance_records
        guid = _FakeCol()
        mapping = _FakeCol()
        mapping._docs.append({
            "standard_label": "Total Revenues", "statement_type": "income",
            "concepts": ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"],
            "isActive": True,
        })
        concepts = _FakeCol()
        concepts._docs.append({
            "_id": "c1", "cik": "0001045810",
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "statement_type": "income",
        })
        db = _fake_db(guidance_values=guid, concepts_standard_mapping=mapping,
                      normalized_concepts_quarterly=concepts)
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            upsert_guidance_records(
                "0001045810", [self._record()], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
        self.assertEqual(
            guid._docs[0]["concept"],
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        )

    def test_agent_provided_concept_stays_authoritative(self):
        from earnings_agents.integrations.guidance import upsert_guidance_records
        guid = _FakeCol()
        db = _fake_db(guidance_values=guid)  # no mapping — resolve would fail
        rec = self._record()
        rec["concept"] = "us-gaap:Revenues"
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            upsert_guidance_records(
                "0001045810", [rec], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
        self.assertEqual(guid._docs[0]["concept"], "us-gaap:Revenues")

    def test_concept_null_only_when_no_row_exists(self):
        """Non-GAAP/custom metrics with no GAAP row keep concept null — the
        resolution genuinely found nothing (never guessed)."""
        from earnings_agents.integrations.guidance import upsert_guidance_records
        guid = _FakeCol()
        db = _fake_db(guidance_values=guid)  # empty mapping + no concepts
        rec = self._record(metric="eps_adjusted")
        rec["standard_label"] = "Adjusted EPS"
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            upsert_guidance_records(
                "0001045810", [rec], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
        self.assertIsNone(guid._docs[0]["concept"])

    def test_concept_fallback_by_label(self):
        """Mapping vocabulary has ~18 labels — a GAAP label that is NOT in it
        (e.g. "Provision for income taxes") still resolves via the row's
        normalized label."""
        from earnings_agents.integrations.guidance import upsert_guidance_records
        guid = _FakeCol()
        concepts = _FakeCol()
        concepts._docs.append({
            "_id": "c9", "cik": "0001045810", "statement_type": "income",
            "concept": "us-gaap:IncomeTaxExpenseBenefit",
            "label": "Provision for income taxes",
        })
        db = _fake_db(guidance_values=guid,
                      normalized_concepts_quarterly=concepts)  # no mapping rows
        rec = self._record(metric="tax_rate")
        rec["standard_label"] = "Provision for income taxes"
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            upsert_guidance_records(
                "0001045810", [rec], accession_number="acc-2",
                filing_period=_q_period(fy=2026, quarter=2),
            )
        self.assertEqual(guid._docs[0]["concept"],
                         "us-gaap:IncomeTaxExpenseBenefit")


# ── Scoring ──────────────────────────────────────────────────────────────────

class TestScore(unittest.TestCase):
    def _doc(self, **kw):
        d = {
            "metric": "revenue", "basis": "gaap", "form": "point",
            "value": 108.0, "value_low": 108.0, "value_high": 108.0,
            "period": {"fiscal_year": 2027, "quarter": 3},
        }
        d.update(kw)
        return d

    def test_point_beat_and_miss(self):
        self.assertEqual(score(self._doc(), 112.3)["outcome"], "beat")
        self.assertEqual(score(self._doc(), 105.0)["outcome"], "miss")
        self.assertEqual(score(self._doc(), 108.0)["outcome"], "inline")
        r = score(self._doc(), 112.3, "acc-x")
        self.assertAlmostEqual(r["delta_abs"], 4.3)
        self.assertAlmostEqual(r["delta_pct"], 3.98, places=2)
        self.assertEqual(r["actual_accession_number"], "acc-x")

    def test_range_bounds(self):
        d = self._doc(form="range", value=109.0, value_low=106.0, value_high=112.0)
        self.assertEqual(score(d, 113.0)["outcome"], "beat")
        self.assertEqual(score(d, 105.0)["outcome"], "miss")
        self.assertEqual(score(d, 109.0)["outcome"], "inline")

    def test_min_and_max(self):
        d = self._doc(form="min", value=10.0, value_low=10.0, value_high=None)
        self.assertEqual(score(d, 11.0)["outcome"], "beat")
        self.assertEqual(score(d, 9.0)["outcome"], "miss")
        self.assertEqual(score(d, 10.0)["outcome"], "miss")  # at least X, == X not a beat
        d = self._doc(form="max", value=5.0, value_low=None, value_high=5.0)
        self.assertEqual(score(d, 4.0)["outcome"], "beat")
        self.assertEqual(score(d, 5.0)["outcome"], "miss")  # up to X, == X not a beat
        self.assertEqual(score(d, 6.0)["outcome"], "miss")

    def test_down_is_good_direction(self):
        d = self._doc(metric="operating_expense", form="range",
                      value=12.0, value_low=11.0, value_high=13.0)
        self.assertEqual(score(d, 10.0)["outcome"], "beat")   # lower expense beats
        self.assertEqual(score(d, 14.0)["outcome"], "miss")

    def test_qualitative_not_scored(self):
        d = self._doc(form="qualitative", value=None, value_low=None, value_high=None)
        self.assertEqual(score(d, 1.0)["outcome"], "qualitative")


class TestScoreForCik(unittest.TestCase):
    def test_scores_arrived_guidance_from_stored_actual(self):
        from earnings_agents.integrations.guidance import score_guidance_for_cik
        guid = _FakeCol()
        guid._docs.append({
            "_id": "g1", "cik": "0001045810", "metric": "revenue",
            "standard_label": "Total Revenues", "statement_type": "income",
            "basis": "gaap", "form": "point", "value": 108.0,
            "value_low": 108.0, "value_high": 108.0,
            "period": {"fiscal_year": 2027, "quarter": 3},
            "is_current": True, "result": None,
        })
        concepts = _FakeCol()
        concepts._docs.append({
            "_id": "c1", "cik": "0001045810", "concept": "us-gaap:Revenues",
            "statement_type": "income",
        })
        values = _FakeCol()
        values._docs.append({
            "cik": "0001045810", "concept_id": "c1", "value": 112.3,
            "accession_number": "acc-actual",
            "reporting_period": {"fiscal_year": 2027, "quarter": 3},
        })
        mapping = _FakeCol()
        mapping._docs.append({
            "standard_label": "Total Revenues", "statement_type": "income",
            "concepts": ["us-gaap:Revenues"], "isActive": True,
        })
        db = _fake_db(guidance_values=guid, normalized_concepts_quarterly=concepts,
                      concept_values_quarterly=values,
                      concepts_standard_mapping=mapping)
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            summary = score_guidance_for_cik(
                "0001045810", _q_period(fy=2027, quarter=3),
            )
        self.assertEqual(summary["scored"], 1)
        self.assertEqual(summary["unresolved"], 0)
        self.assertEqual(guid._docs[0]["result"]["outcome"], "beat")
        self.assertEqual(guid._docs[0]["result"]["actual_accession_number"], "acc-actual")

    def test_pending_guidance_not_scored(self):
        from earnings_agents.integrations.guidance import score_guidance_for_cik
        guid = _FakeCol()
        guid._docs.append({
            "_id": "g2", "cik": "0001045810", "metric": "revenue",
            "standard_label": "Total Revenues", "statement_type": "income",
            "basis": "gaap", "form": "point", "value": 108.0,
            "value_low": 108.0, "value_high": 108.0,
            "period": {"fiscal_year": 2027, "quarter": 4},  # not arrived
            "is_current": True, "result": None,
        })
        db = _fake_db(guidance_values=guid)
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            summary = score_guidance_for_cik("0001045810", _q_period(fy=2027, quarter=3))
        self.assertEqual(summary["checked"], 0)
        self.assertIsNone(guid._docs[0]["result"])


# ── period_arrived helper ────────────────────────────────────────────────────

class TestPeriodArrived(unittest.TestCase):
    def test_quarterly_arrival(self):
        filing = _q_period(fy=2027, quarter=3)
        self.assertTrue(period_arrived({"fiscal_year": 2027, "quarter": 3}, filing))
        self.assertTrue(period_arrived({"fiscal_year": 2027, "quarter": 2}, filing))
        self.assertFalse(period_arrived({"fiscal_year": 2027, "quarter": 4}, filing))
        self.assertFalse(period_arrived({"fiscal_year": 2028, "quarter": 1}, filing))

    def test_annual_arrival(self):
        filing = _q_period(fy=2027, quarter=None, period_type="annual")
        self.assertTrue(period_arrived({"fiscal_year": 2027, "quarter": None}, filing))
        self.assertFalse(period_arrived({"fiscal_year": 2028, "quarter": None}, filing))


# ── save_guidance node — never fails a run ───────────────────────────────────

class TestSaveGuidanceNode(unittest.TestCase):
    def _state(self, **overrides):
        state = {
            "ticker": "NVDA", "company_name": "NVIDIA",
            "status": "saved", "cik": "0001045810",
            "accession_number": "acc-2",
            "detected_period": {
                "period_type": "quarterly", "period_end": "2026-06-30",
                "quarter": 2, "fiscal_year": 2026,
                "period_label": "Three Months Ended June 30, 2026",
            },
            "guidance_records": [{
                "metric": "revenue", "standard_label": "Total Revenues",
                "statement_type": "income", "basis": "gaap", "form": "point",
                "value": 108.0, "value_low": 108.0, "value_high": 108.0,
                "unit": "USD", "scale": "billions", "currency": "USD",
                "period": {"fiscal_year": 2026, "quarter": 3,
                           "period_type": "quarterly", "label": "Q3 FY2026"},
            }],
        }
        state.update(overrides)
        return state

    def test_saves_records_and_leaves_status_saved(self):
        guid = _FakeCol()
        db = _fake_db(guidance_values=guid)
        with mock.patch("earnings_agents.integrations.guidance._get_db", return_value=db):
            out = save_guidance_node(self._state())
        self.assertEqual(out["status"], "saved")
        self.assertEqual(out["guidance_save"]["status"], "saved")
        self.assertEqual(out["guidance_save"]["upserted"], 1)
        self.assertEqual(guid._docs[0]["source"], "llm")

    def test_disabled_config_noops(self):
        with mock.patch.object(_config, "GUIDANCE_ENABLED", False):
            out = save_guidance_node(self._state())
        self.assertEqual(out["guidance_save"]["status"], "skipped")
        self.assertEqual(out["status"], "saved")

    def test_no_records_noops(self):
        out = save_guidance_node(self._state(guidance_records=[]))
        self.assertEqual(out["guidance_save"]["status"], "no_records")

    def test_upsert_error_never_fails_run(self):
        with mock.patch(
            "earnings_agents.integrations.guidance.upsert_guidance_records",
            side_effect=RuntimeError("mongo down"),
        ):
            out = save_guidance_node(self._state())
        self.assertEqual(out["status"], "saved")
        self.assertEqual(out["guidance_save"]["status"], "failed")
        self.assertIn("error", out["guidance_save"])


# ── End-to-end pipeline wiring: __guidance__ → state.guidance_records ──────

class TestPipelineGuidanceWiring(unittest.TestCase):
    def test_pipeline_pops_guidance_into_state(self):
        from unittest.mock import patch
        from earnings_agents.agent import pipeline as P

        state = {
            "ticker": "NVDA", "company_name": "NVIDIA", "status": "extracted",
            "raw_text": "some document text", "cik": "0001045810",
            "detected_period": {
                "period_type": "quarterly", "period_end": "2026-06-30",
                "quarter": 2, "fiscal_year": 2026,
                "period_label": "Three Months Ended June 30, 2026",
            },
            "document_map": [], "target_concepts": [],
            "recent_concept_ids": [], "calculated_concepts": [],
            "fiscal_year_end_code": "0131",
            "company_industry": {},
        }
        final = {
            "__scale__": "millions",
            "us-gaap:Revenues": 1234.0,
            "__currency__": "USD",
            "__guidance__": [{
                "metric": "revenue", "form": "point", "value": 1.23,
                "period": {"fiscal_year": 2026, "quarter": 3,
                           "period_type": "quarterly"},
            }],
        }
        from earnings_agents.agent.period import DetectedPeriod
        period = DetectedPeriod(
            period_type="quarterly", period_end=date(2026, 6, 30),
            quarter=2, fiscal_year=2026, period_label="Test",
        )
        with patch.object(P, "run_agent_loop", return_value=final), \
             patch.object(P, "prescan_document", return_value=("millions", None, [])), \
             patch.object(P, "build_section_index",
                          return_value=({"sections": []}, 0.0)), \
             patch.object(P, "require_detected_period", return_value=period), \
             patch.object(P, "usd_metadata",
                          return_value={"currency": None, "detected_codes": []}), \
             patch.object(P, "map_concepts",
                          return_value=({"c1": 1234.0 * 1_000_000},
                                        {"c1": "us-gaap:Revenues"},
                                        {"us-gaap:Revenues"})), \
             patch.object(P, "semantically_map_unmapped_metrics",
                          return_value=({}, set(), {})), \
             patch.object(P, "build_calc_derivation_block", return_value=("", set())), \
             patch.object(P, "build_no_scale_keys", return_value=set()), \
             patch.object(P, "extract_is_section", return_value=None):
            out = P._run_extraction_pass(state, "plain text", [])

        self.assertEqual(out["status"], "extracted")
        recs = out.get("guidance_records") or []
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["metric"], "revenue")
        self.assertEqual(recs[0]["standard_label"], "Total Revenues")
        self.assertEqual(recs[0]["period"]["fiscal_year"], 2026)
        self.assertEqual(recs[0]["period"]["quarter"], 3)
        self.assertEqual(recs[0]["source"], "llm")
        self.assertEqual(recs[0]["currency"], "USD")
        # the guidance payload must NOT leak into concept metrics
        self.assertNotIn("__guidance__", out["metrics"])

    def test_no_guidance_in_finalize_is_fine(self):
        from unittest.mock import patch
        from earnings_agents.agent import pipeline as P

        state = {
            "ticker": "NVDA", "company_name": "NVIDIA", "status": "extracted",
            "raw_text": "text", "cik": "0001045810",
            "detected_period": {
                "period_type": "quarterly", "period_end": "2026-06-30",
                "quarter": 2, "fiscal_year": 2026,
                "period_label": "Three Months Ended June 30, 2026",
            },
            "document_map": [], "target_concepts": [],
            "recent_concept_ids": [], "calculated_concepts": [],
            "fiscal_year_end_code": "0131", "company_industry": {},
        }
        final = {"__scale__": "as-is", "__currency__": "USD",
                 "us-gaap:Revenues": 1234.0}
        from earnings_agents.agent.period import DetectedPeriod
        period = DetectedPeriod(
            period_type="quarterly", period_end=date(2026, 6, 30),
            quarter=2, fiscal_year=2026, period_label="Test",
        )
        with patch.object(P, "run_agent_loop", return_value=final), \
             patch.object(P, "prescan_document", return_value=("as-is", None, [])), \
             patch.object(P, "build_section_index",
                          return_value=({"sections": []}, 0.0)), \
             patch.object(P, "require_detected_period", return_value=period), \
             patch.object(P, "usd_metadata",
                          return_value={"currency": None, "detected_codes": []}), \
             patch.object(P, "map_concepts",
                          return_value=({}, {}, set())), \
             patch.object(P, "semantically_map_unmapped_metrics",
                          return_value=({}, set(), {})), \
             patch.object(P, "build_calc_derivation_block", return_value=("", set())), \
             patch.object(P, "build_no_scale_keys", return_value=set()), \
             patch.object(P, "extract_is_section", return_value=None):
            out = P._run_extraction_pass(state, "plain text", [])
        self.assertEqual(out.get("guidance_records") or [], [])


if __name__ == "__main__":
    unittest.main()