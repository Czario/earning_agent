"""Unit tests for scale/sign parsing and currency-metadata preservation."""
from __future__ import annotations

import unittest

from earnings_agents.agent.derive import build_no_scale_keys
from earnings_agents.agent.loop import _parse_llm_response


class TestParseLlmResponse(unittest.TestCase):
    def test_operations_and_share_count_keys_scale_in_millions(self):
        # Regression (ABBV Q3 2026): "...ContinuingOperations..." falsely matched
        # the case-insensitive camelCase "Ratio" guard, and diluted share counts
        # were never scaled at all.
        out = _parse_llm_response(
            '{"__scale__":"millions",'
            '"[us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest]":4829,'
            '"[us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding]":1772,'
            '"[us-gaap:EarningsPerShareDiluted]":1.5}'
        )
        self.assertEqual(
            out["[us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest]"],
            4_829_000_000,
        )
        self.assertEqual(out["[us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding]"], 1_772_000_000)
        self.assertEqual(out["[us-gaap:EarningsPerShareDiluted]"], 1.5)

    def test_prescan_shares_in_thousands_overrides_dollar_scale(self):
        out = _parse_llm_response(
            '{"__scale__":"millions","[us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding]":1772000}',
            shares_multiplier=1_000,
        )
        self.assertEqual(out["[us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding]"], 1_772_000_000)

    def test_per_value_evidence_scale_is_authoritative(self):
        import json
        ev = {f"a{i}": {"lines": [100 + i, 100 + i], "scale": "thousands"} for i in range(5)}
        ev["seg"] = {"lines": [110, 110], "scale": "millions"}
        d = {f"a{i}": 1000.0 for i in range(5)}
        d.update({"seg": 12500.0, "__scale__": "thousands", "__evidence__": ev})
        out = _parse_llm_response(json.dumps(d))
        self.assertEqual(out["seg"], 12_500_000_000)
        self.assertEqual(out["a0"], 1_000_000)

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

    def test_monetary_concepts_with_ration_rate_employee_margin_shares_labels_scale(self):
        # Regression: words containing "ratio" ("operations", "administration",
        # "integration", "separation", "consideration"), monetary "exchange rate",
        # "employee benefits", "Gross margin" on us-gaap:GrossProfit, "Cost of
        # production", and Balance Sheet "Common stock ... shares issued and
        # outstanding" must NOT be added to no_scale_keys.
        concepts = [
            self._concept("Income from operations", "us-gaap:OperatingIncomeLoss"),
            self._concept(
                "Income from continuing operations before income taxes",
                "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            ),
            self._concept(
                "Selling, general and administration",
                "us-gaap:SellingGeneralAndAdministrativeExpense",
            ),
            self._concept("Acquisition and integration costs", "custom:AcquisitionAndIntegrationCosts"),
            self._concept("Restructuring and separation", "custom:RestructuringAndSeparation"),
            self._concept(
                "Effect of exchange rate changes on cash and cash equivalents",
                "us-gaap:EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            ),
            self._concept(
                "Employee benefits and share-based compensation",
                "us-gaap:EmployeeBenefitsAndShareBasedCompensation",
            ),
            self._concept("Gross margin", "us-gaap:GrossProfit"),
            self._concept("Cost of production", "us-gaap:CostOfGoodsAndServicesSold"),
            self._concept(
                "Common stock, $0.001 par value; 500,000 shares authorized; 120,000 shares issued and outstanding",
                "us-gaap:CommonStockValue",
            ),
            self._concept("U.S. Operations", "us-gaap:Revenues|001.001"),
            self._concept("Growth Markets", "us-gaap:Revenues|001.002"),
        ]
        self.assertEqual(build_no_scale_keys(concepts), set())

        payload = (
            '{"__scale__":"millions","__currency__":"USD",'
            '"[us-gaap:OperatingIncomeLoss]": 4200,'
            '"[us-gaap:SellingGeneralAndAdministrativeExpense]": 1800,'
            '"[us-gaap:EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents]": -45,'
            '"[us-gaap:EmployeeBenefitsAndShareBasedCompensation]": 500,'
            '"[us-gaap:GrossProfit]": 8602,'
            '"[us-gaap:CommonStockValue]": 12}'
        )
        out = _parse_llm_response(payload, no_scale_keys=build_no_scale_keys(concepts))
        self.assertEqual(out["[us-gaap:OperatingIncomeLoss]"], 4_200_000_000)
        self.assertEqual(out["[us-gaap:SellingGeneralAndAdministrativeExpense]"], 1_800_000_000)
        self.assertEqual(
            out["[us-gaap:EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents]"],
            -45_000_000,
        )
        self.assertEqual(out["[us-gaap:EmployeeBenefitsAndShareBasedCompensation]"], 500_000_000)
        self.assertEqual(out["[us-gaap:GrossProfit]"], 8_602_000_000)
        self.assertEqual(out["[us-gaap:CommonStockValue]"], 12_000_000)

    def test_llm_scale_not_downgraded_by_prescan_and_per_value_evidence_scale_honored(self):
        # Agent confirmed table is in millions even though prescan saw "in thousands"
        out = _parse_llm_response(
            '{"__scale__":"millions","__currency__":"USD","[us-gaap:Revenues]": 1234}',
            prescan_dollar_multiplier=1_000,
        )
        self.assertEqual(out["[us-gaap:Revenues]"], 1_234_000_000)

        # Per-value evidence scale overrides when a narrative segment figure is in billions
        out_ev = _parse_llm_response(
            '{"__scale__":"thousands","__currency__":"USD",'
            '"__evidence__":{"[us-gaap:Revenues|seg]":{"lines":[10,12],"scale":"millions"}},'
            '"[us-gaap:Revenues]": 5931000,'
            '"[us-gaap:Revenues|seg]": 631}'
        )
        self.assertEqual(out_ev["[us-gaap:Revenues]"], 5_931_000_000)
        self.assertEqual(out_ev["[us-gaap:Revenues|seg]"], 631_000_000)

    def test_fallback_scale_when_value_has_no_scale_given(self):
        # Most values in __evidence__ have "scale": "millions", while some values
        # have no scale in __evidence__ (or are absent from __evidence__, e.g.
        # computed/derived rows) and __scale__ is omitted or "as-is".
        payload = (
            '{"__scale__":"as-is","__currency__":"USD",'
            '"__evidence__":{'
            '"[us-gaap:Revenues]":{"lines":[100,101],"scale":"millions"},'
            '"[us-gaap:CostOfRevenue]":{"lines":[102,103],"scale":"millions"},'
            '"[us-gaap:OperatingIncomeLoss]":{"lines":[110,111]},'
            '"[us-gaap:NetIncomeLoss]":{"lines":[120,121],"scale":"none"}'
            '},'
            '"[us-gaap:Revenues]": 15400,'
            '"[us-gaap:CostOfRevenue]": -6798,'
            '"[us-gaap:GrossProfit]": 8602,'
            '"[us-gaap:OperatingIncomeLoss]": 4200,'
            '"[us-gaap:NetIncomeLoss]": 3100,'
            '"[us-gaap:EarningsPerShareDiluted]": 2.45}'
        )
        out = _parse_llm_response(payload)
        self.assertEqual(out["[us-gaap:Revenues]"], 15_400_000_000)
        self.assertEqual(out["[us-gaap:CostOfRevenue]"], -6_798_000_000)
        self.assertEqual(out["[us-gaap:GrossProfit]"], 8_602_000_000)
        self.assertEqual(out["[us-gaap:OperatingIncomeLoss]"], 4_200_000_000)
        self.assertEqual(out["[us-gaap:NetIncomeLoss]"], 3_100_000_000)
        self.assertEqual(out["[us-gaap:EarningsPerShareDiluted]"], 2.45)

    def test_statement_local_fallback_scale_from_nearby_lines(self):
        # Income statement (lines 50-80) is in millions; Balance sheet (lines 300-340)
        # is in thousands; a balance sheet row with no scale given at line 315
        # inherits the nearby statement's "thousands" fallback scale.
        payload = (
            '{"__currency__":"USD",'
            '"__evidence__":{'
            '"[us-gaap:Revenues]":{"lines":[50,51],"scale":"millions"},'
            '"[us-gaap:NetIncomeLoss]":{"lines":[70,71],"scale":"millions"},'
            '"[us-gaap:Assets]":{"lines":[305,306],"scale":"thousands"},'
            '"[us-gaap:Liabilities]":{"lines":[325,326],"scale":"thousands"},'
            '"[us-gaap:StockholdersEquity]":{"lines":[315,316]}'
            '},'
            '"[us-gaap:Revenues]": 1200,'
            '"[us-gaap:NetIncomeLoss]": 300,'
            '"[us-gaap:Assets]": 950000,'
            '"[us-gaap:Liabilities]": 450000,'
            '"[us-gaap:StockholdersEquity]": 500000}'
        )
        out = _parse_llm_response(payload)
        self.assertEqual(out["[us-gaap:Revenues]"], 1_200_000_000)
        self.assertEqual(out["[us-gaap:NetIncomeLoss]"], 300_000_000)
        self.assertEqual(out["[us-gaap:Assets]"], 950_000_000)
        self.assertEqual(out["[us-gaap:Liabilities]"], 450_000_000)
        self.assertEqual(out["[us-gaap:StockholdersEquity]"], 500_000_000)


class TestSanitizeToolMessages(unittest.TestCase):
    def test_inserts_missing_tool_message(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from earnings_agents.agent.loop import _sanitize_tool_messages

        messages = [
            HumanMessage(content="Search for revenue"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "t1", "args": {}, "id": "call_1"},
                    {"name": "t2", "args": {}, "id": "call_2"},
                ],
            ),
            ToolMessage(content="result 1", tool_call_id="call_1"),
        ]
        sanitized = _sanitize_tool_messages(messages)
        self.assertEqual(len(sanitized), 4)
        self.assertIsInstance(sanitized[3], ToolMessage)
        self.assertEqual(sanitized[3].tool_call_id, "call_2")

    def test_inserts_missing_invalid_tool_message(self):
        from langchain_core.messages import AIMessage, ToolMessage
        from earnings_agents.agent.loop import _sanitize_tool_messages

        msg = AIMessage(content="")
        msg.invalid_tool_calls = [{"name": "bad_tool", "args": {}, "id": "call_inv_1", "error": "bad"}]
        messages = [msg]
        sanitized = _sanitize_tool_messages(messages)
        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[1], ToolMessage)
        self.assertEqual(sanitized[1].tool_call_id, "call_inv_1")

    def test_leaves_complete_messages_unchanged(self):
        from langchain_core.messages import AIMessage, ToolMessage
        from earnings_agents.agent.loop import _sanitize_tool_messages

        messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "t1", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(content="result 1", tool_call_id="call_1"),
        ]
        sanitized = _sanitize_tool_messages(messages)
        self.assertEqual(len(sanitized), 2)
        self.assertEqual(sanitized[1].tool_call_id, "call_1")


if __name__ == "__main__":
    unittest.main()
