"""Industry-aware concept prioritisation, identities, and prompt hints.

Classifies companies by SIC code into categories with distinct:
  * Tier overrides — concepts promoted/demoted in criticality
  * Identity checks — industry-specific accounting identities
  * Prompt hints — injection strings for the extraction prompt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── SIC-code → industry category ──────────────────────────────────────────────

# Each entry: (sic_range_low, sic_range_high, category_label)
_SIC_RANGES: list[tuple[int, int, str]] = [
    (6000, 6099, "banking"),
    (6100, 6199, "banking"),       # credit agencies, mortgage banks
    (6200, 6299, "financial"),     # security brokers, exchanges
    (6300, 6399, "insurance"),
    (6400, 6499, "insurance"),     # insurance agents, brokers
    (6500, 6599, "real_estate"),
    (6700, 6799, "financial"),     # holding companies, investment trusts
    (7000, 7099, "hotels"),
    (7200, 7299, "services"),
    (7300, 7371, "services"),
    (7372, 7379, "software"),
    (7380, 7399, "services"),
    (2000, 3999, "manufacturing"),
    (4000, 4999, "transportation"),
    (5000, 5199, "wholesale"),
    (5200, 5999, "retail"),
    (7000, 7099, "services"),
    (7800, 7899, "media"),
    (7900, 7999, "media"),         # amusement, recreation
    (8000, 8099, "healthcare"),
    (8200, 8299, "education"),
    (8700, 8749, "services"),      # engineering, accounting, R&D
    (1300, 1399, "energy"),        # oil & gas
    (4900, 4999, "utilities"),
]


def classify(sic_code: str | None) -> str:
    """Return the industry category for *sic_code*, or ``'general'``."""
    if not sic_code:
        return "general"
    try:
        code = int(sic_code)
    except (ValueError, TypeError):
        return "general"
    for lo, hi, category in _SIC_RANGES:
        if lo <= code <= hi:
            return category
    return "general"


# ── Industry-specific TIER overrides ──────────────────────────────────────────

# For each industry, concepts promoted from TIER2 → TIER1 (critical)
# and TIER3 → TIER2 (expected).  The base TIER1/2/3 registry in
# critical_metrics.py applies to all industries; these overrides add or
# demote concepts for the specified industry.
#
# Keys: ``"concept_pattern"`` or ``"role"`` — matched against the concept label
# or the abstract derivation role (e.g. ``"gross_profit"``, ``"interest_income"``).
# ``action`` is ``"promote"`` (Tier2→Tier1, Tier3→Tier2) or ``"demote"``
# (Tier1→Tier2).

@dataclass
class IndustryProfile:
    category: str
    label: str                         # human-readable
    prompt_hint: str                   # injected into extraction prompt
    # Identity overrides: list of (role_a, op, role_b, result_role, tolerance)
    # e.g. ("interest_income", "+", "other_income_net", "net_interest_income", 0.005)
    extra_identities: list[tuple[str, str, str, str, float]] = field(default_factory=list)
    # Identities to SKIP (not applicable to this industry)
    skip_identities: list[str] = field(default_factory=list)
    # Concepts to promote: ("role", from_tier) → at_least tier
    promote: list[tuple[str, str]] = field(default_factory=list)


_PROFILES: dict[str, IndustryProfile] = {
    "banking": IndustryProfile(
        category="banking",
        label="Banking / Financial Services",
        prompt_hint=(
            "INDUSTRY CONTEXT — This is a BANK / financial-institution earnings "
            "release.  Net Interest Income (after provision) is the primary "
            "revenue component.  Non-interest income (fees, trading, service "
            "charges) is a secondary revenue line.  Interest expense is a major "
            "cost.  Total revenue = Net Interest Income + Non-Interest Income.  "
            "Look for 'Net interest income', 'Interest income', 'Interest expense', "
            "and 'Noninterest income' rows in the primary income statement — these "
            "are the core metrics, not supplemental details."
        ),
        extra_identities=[
            # Net Interest Income = Interest Income − Interest Expense
            ("interest_income", "-", "interest_expense", "gross_profit", 0.005),
        ],
        skip_identities=[
            "Revenue − CoR = GP",   # banks have no COGS; GP means NII
        ],
        promote=[
            ("interest_income", "tier1"),
            ("interest_expense", "tier1"),
            ("other_income_net", "tier2"),
            ("pretax_income", "tier1"),
        ],
    ),
    "software": IndustryProfile(
        category="software",
        label="Software / Technology",
        prompt_hint=(
            "INDUSTRY CONTEXT — This is a SOFTWARE / TECHNOLOGY company earnings "
            "release.  Revenue is primarily from licenses, subscriptions, and "
            "services.  Cost of revenue is typically low (high gross margins).  "
            "Research & Development and Sales & Marketing are the dominant "
            "operating expenses.  Stock-based compensation is a significant "
            "non-cash item often disclosed separately.  Look for segment revenue "
            "breakdowns (product lines, geographies) in supplemental tables."
        ),
        promote=[
            ("rd_expense", "tier2"),
            ("sm_expense", "tier2"),
        ],
    ),
    "insurance": IndustryProfile(
        category="insurance",
        label="Insurance",
        prompt_hint=(
            "INDUSTRY CONTEXT — This is an INSURANCE company earnings release.  "
            "Revenue = Premiums Earned + Investment Income.  Key costs include "
            "Claims/Losses, Policyholder Benefits, and Underwriting Expenses.  "
            "The standard Income Statement structure (Revenue − CoR = GP) does "
            "NOT apply.  Look for 'Premiums earned', 'Net investment income', "
            "'Policyholder benefits', and 'Claims and claim adjustment expenses'."
        ),
        skip_identities=[
            "Revenue − CoR = GP",   # insurance has no COGS
        ],
    ),
    "manufacturing": IndustryProfile(
        category="manufacturing",
        label="Manufacturing / Industrial",
        prompt_hint=(
            "INDUSTRY CONTEXT — This is a MANUFACTURING / INDUSTRIAL company "
            "earnings release.  Cost of Goods Sold (COGS) is the primary direct "
            "cost.  Gross Profit = Revenue − COGS is an important margin metric.  "
            "Depreciation, raw materials, and labour costs drive operating margins."
        ),
        promote=[
            ("cost_of_revenue", "tier1"),
        ],
    ),
    "energy": IndustryProfile(
        category="energy",
        label="Energy / Oil & Gas",
        prompt_hint=(
            "INDUSTRY CONTEXT — This is an ENERGY / OIL & GAS company earnings "
            "release.  Revenue typically includes Production revenue, Midstream "
            "revenue, and Marketing revenue.  Key costs: Production costs, DD&A "
            "(Depreciation, Depletion & Amortisation), Exploration expense.  "
            "Look for production volumes and realised prices in supplemental data."
        ),
    ),
    "real_estate": IndustryProfile(
        category="real_estate",
        label="Real Estate / REIT",
        prompt_hint=(
            "INDUSTRY CONTEXT — This is a REIT / REAL ESTATE company earnings "
            "release.  Key metrics: Rental Revenue, Net Operating Income (NOI), "
            "Funds From Operations (FFO), Adjusted FFO.  Depreciation is a major "
            "non-cash expense.  The standard Revenue − CoR identity may not apply "
            "— REITs often report NOI instead of Gross Profit."
        ),
        skip_identities=[
            "Revenue − CoR = GP",
        ],
    ),
    "healthcare": IndustryProfile(
        category="healthcare",
        label="Healthcare",
        prompt_hint=(
            "INDUSTRY CONTEXT — This is a HEALTHCARE company earnings release.  "
            "Revenue includes patient service revenue, premium revenue, and "
            "pharmaceutical/product sales.  Key costs: Cost of services/products, "
            "R&D (for pharma/biotech), and SG&A.  For managed care, look for "
            "Medical Loss Ratio (MLR) components."
        ),
    ),
    "financial": IndustryProfile(
        category="financial",
        label="Financial Services (non-bank)",
        prompt_hint=(
            "INDUSTRY CONTEXT — This is a FINANCIAL SERVICES company earnings "
            "release.  Revenue includes fee income, investment income, and "
            "transaction-based revenue.  Interest income/expense may be present "
            "but is not the primary business driver."
        ),
    ),
    "retail": IndustryProfile(
        category="retail",
        label="Retail",
        prompt_hint=(
            "INDUSTRY CONTEXT — This is a RETAIL company earnings release.  "
            "Key metrics: Net Sales/Revenue, Cost of Sales (COGS), Gross Profit, "
            "SG&A, Operating Income.  Same-store sales (comps) and store count "
            "are key operational metrics typically disclosed in commentary."
        ),
    ),
}


def get_profile(sic_code: str | None) -> IndustryProfile:
    """Return the industry profile for *sic_code*, or the default general."""
    category = classify(sic_code)
    return _PROFILES.get(category, IndustryProfile(
        category="general",
        label="General",
        prompt_hint="",
    ))


def get_prompt_hint(sic_code: str | None) -> str:
    """Return the industry-specific prompt injection string."""
    return get_profile(sic_code).prompt_hint


def get_skip_identities(sic_code: str | None) -> list[str]:
    """Return identity descriptions to skip for this industry."""
    return get_profile(sic_code).skip_identities


def get_extra_identities(sic_code: str | None) -> list[tuple[str, str, str, str, float]]:
    """Return extra identity checks for this industry."""
    return get_profile(sic_code).extra_identities


def get_promoted_concepts(sic_code: str | None) -> list[tuple[str, str]]:
    """Return ``[(role, target_tier), ...]`` for promoted concepts."""
    return get_profile(sic_code).promote


def load_company_industry(cik: str) -> dict[str, Any] | None:
    """Load the company's industry info from normalize_data.companies.

    Returns ``{'sic_code': ..., 'sic_description': ..., 'category': ...}``
    or ``None`` if the company is not found.
    """
    from earnings_agents.tools.normalize_data_client import _get_client, _NORMALIZE_DB

    db = _get_client()[_NORMALIZE_DB]
    doc = db["companies"].find_one(
        {"cik": cik},
        {"industry.sic_code": 1, "industry.sic_description": 1},
    )
    if not doc:
        return None
    ind = doc.get("industry", {}) or {}
    sic_code = ind.get("sic_code", "")
    return {
        "sic_code": sic_code,
        "sic_description": ind.get("sic_description", ""),
        "category": classify(sic_code),
    }
