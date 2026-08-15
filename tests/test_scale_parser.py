"""Scale-parser exclusion tests — percentages/per-share/count keys are never
scaled by the ``__scale__`` multiplier, including CamelCase taxonomy keys
(e.g. ``custom:NetInterestMarginCompanyProvided`` — observed live storing
2,080,000 for a 2.08% net interest yield)."""
from __future__ import annotations

import pytest

from earnings_agents.agent.loop import _parse_llm_response

MILLIONS = 1_000_000


@pytest.mark.parametrize("key", [
    "[custom:NetInterestMarginCompanyProvided]",   # observed live bug
    "Net Interest Yield (Company Provided)",
    "[custom:NetInterestYieldCompanyProvided]",
    "Net interest margin",
    "Gross margin",
    "[custom:RevenueGrowthYoY]",
    "[us-gaap:EarningsPerShareBasic]",
    "Weighted average shares used in computation",
    "Number of shares",
    "Effective tax rate",
    "Current ratio",
])
def test_percentage_and_count_keys_are_never_scaled(key):
    result = _parse_llm_response(
        f'{{"__scale__": "millions", "{key}": 2.08, "[us-gaap:Revenue]": 10}}',
        1, MILLIONS,
    )
    assert result[key] == 2.08, f"{key!r} must stay as printed"
    assert result["[us-gaap:Revenue]"] == 10 * MILLIONS  # dollar key still scaled


def test_dollar_keys_containing_ration_are_still_scaled():
    """'Operating expenses' contains the letters r-a-t-i-n — the ratio pattern
    must be word-bounded so dollar lines are never mis-excluded."""
    result = _parse_llm_response(
        '{"__scale__": "thousands", "Operating expenses": 1234}',
        1, 1_000,
    )
    assert result["Operating expenses"] == 1234 * 1_000


def test_share_count_never_scaled():
    result = _parse_llm_response(
        '{"__scale__": "millions", "number of shares": 5000, "[us-gaap:Revenue]": 10}',
        1, MILLIONS,
    )
    assert result["number of shares"] == 5000
    assert result["[us-gaap:Revenue]"] == 10 * MILLIONS
