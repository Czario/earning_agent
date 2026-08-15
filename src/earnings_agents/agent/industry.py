"""Advisory industry context for the extraction agent.

SIC data from ``normalize_data.companies`` is injected into the extraction
prompt as READ-ONLY context.  It helps the agent interpret filing terminology
(press-release labels often diverge from XBRL labels) but can never:

  • add concepts outside the recent-history extraction target,
  • supply or infer values,
  • override anything the agent read from the filing.
"""
from __future__ import annotations


def normalize_company_industry(industry: dict | None) -> dict | None:
    """Return a clean ``{sic_code, sic_description}`` profile, or ``None``.

    Tolerates ``None``, non-dict values, and whitespace-only fields.
    """
    if not isinstance(industry, dict):
        return None
    sic = str(industry.get("sic_code") or "").strip()
    desc = str(industry.get("sic_description") or "").strip()
    if not sic and not desc:
        return None
    return {"sic_code": sic, "sic_description": desc}


def build_industry_context(industry: dict | None) -> str:
    """Build the advisory industry block for the extraction system prompt.

    Missing/empty industry data → an explicit instruction that no industry
    info is available, so the agent does not improvise one from its priors.
    """
    profile = normalize_company_industry(industry)
    if profile is None:
        return (
            "INDUSTRY CONTEXT:\n"
            "Industry information is unavailable.  Do not infer "
            "industry-specific metrics — use the filing text and the concept "
            "list only."
        )

    sic = profile["sic_code"] or "unknown"
    desc = profile["sic_description"] or "unknown"
    return (
        "INDUSTRY CONTEXT — advisory only:\n"
        f"  SIC: {sic}\n"
        f"  Description: {desc}\n"
        "\n"
        "Use this ONLY to:\n"
        "  • recognize industry-specific terminology used in the filing;\n"
        "  • choose search synonyms for concepts ALREADY in the concept list;\n"
        "  • prioritize relevant income-statement / segment sections.\n"
        "Do NOT:\n"
        "  • add concepts that are not in the concept list;\n"
        "  • infer financial values from the industry;\n"
        "  • assume a metric is reported merely because it is common in this industry;\n"
        "  • override any number, period, scale, or label read from the filing."
    )
