"""Unit tests for scale/sign parsing and currency-metadata preservation."""
from __future__ import annotations

import unittest

from earnings_agents.agent.derive import build_no_scale_keys
from earnings_agents.agent.loop import _parse_llm_response


class TestParseLlmResponse(unittest.TestCase):
    def test_scales_values_and_preserves_currency(self):
        out = _parse_llm_response(
            '{"__scale__":"millions","__currency__":"USD","[us-gaap:Revenues]": 1234}'
        )
        self.assertEqual(out["__currency__"], "USD")
        self.assertEqual(out["[us-gaap:Revenues]"], 1_234_000_000)

    def test_foreign_currency_survives_but_does_not_convert(self):
        # Currency conversion is not the parser's job: the code must survive
        # untouched so the save gate can reject it.
        out = _parse_llm_response(
            '{"__scale__":"millions","__currency__":"EUR","[us-gaap:Revenues]": 1234}'
        )
        self.assertEqual(out["__currency__"], "EUR")
        self.assertEqual(out["[us-gaap:Revenues]"], 1_234_000_000)

    def test_parenthesized_is_negative(self):
        out = _parse_llm_response('{"__currency__":"USD","[us-gaap:InterestExpense]":"(175,685)"}')
        self.assertEqual(out["[us-gaap:InterestExpense]"], -175_685.0)

    def test_currency_never_scaled_as_number(self):
        out = _parse_llm_response('{"__scale__":"thousands","__currency__":"USD","x": 5}')
        self.assertEqual(out["__currency__"], "USD")
        self.assertEqual(out["x"], 5_000)

    def test_member_tagged_eps_key_never_scaled(self):
        # Regression: member-tagged EPS keys ([custom:Basic|014.001]) carry no
        # "per share" text in the key string, so the key-based guardrail alone
        # scaled 4.85 → 4,850,000 (observed live on PDD).  The label-derived
        # no_scale_keys set must protect them.
        out = _parse_llm_response(
            '{"__scale__":"millions","__currency__":"USD",'
            '"[custom:Basic|014.001]": 4.85, "[us-gaap:Revenues]": 15400}',
            no_scale_keys={"[custom:Basic|014.001]", "custom:Basic|014.001"},
        )
        self.assertEqual(out["[custom:Basic|014.001]"], 4.85)
        self.assertEqual(out["[us-gaap:Revenues]"], 15_400_000_000)


class TestBuildNoScaleKeys(unittest.TestCase):
    def _concept(self, label, key, calculated=False):
        return {
            "_id": key,
            "label": label,
            "concept": key,
            "taxonomy_key": key,
            "calculated": calculated,
            "dimension": False,
            "dimension_concept": False,
        }

    def test_per_ordinary_share_and_per_ads_labels(self):
        # Exact labels from the PDD run — neither matches the old regex.
        concepts = [
            self._concept("Basic (Earnings per ordinary share)", "custom:Basic|014.001"),
            self._concept("Diluted (Earnings per ordinary share)", "custom:Diluted|014.002"),
            self._concept(
                "Basic (Earnings per ADS (4 ordinary shares equal one ADS))",
                "custom:Basic|015.001",
            ),
            self._concept("Revenue", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"),
        ]
        keys = build_no_scale_keys(concepts)
        self.assertIn("[custom:Basic|014.001]", keys)
        self.assertIn("custom:Diluted|014.002", keys)
        self.assertIn("[custom:Basic|015.001]", keys)
        self.assertNotIn("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", keys)

    def test_percentage_and_share_count_labels(self):
        concepts = [
            self._concept("Gross Profit Margin", "us-gaap:GrossProfitMargin"),
            self._concept("Weighted-average shares outstanding", "us-gaap:WeightedAverageNumberOfSharesOutstanding"),
        ]
        keys = build_no_scale_keys(concepts)
        self.assertIn("us-gaap:GrossProfitMargin", keys)
        self.assertIn("[us-gaap:WeightedAverageNumberOfSharesOutstanding]", keys)

    def test_dollar_labels_are_scalable(self):
        concepts = [
            self._concept("Revenue", "us-gaap:Revenues"),
            self._concept("Cost of Revenue", "us-gaap:CostOfRevenue"),
        ]
        self.assertEqual(build_no_scale_keys(concepts), set())


if __name__ == "__main__":
    unittest.main()
