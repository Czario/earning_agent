"""Forward-looking guidance extraction — normalization + validation.

The extraction agent reports guidance in ``finalize_extraction`` under the
metadata key ``__guidance__`` (a JSON list, one object per guidance number).
This module turns each raw record into a ``guidance_values``-shaped document —
the exact schema the admin backend's ``GuidanceService`` reads/writes in the
``guidance_values`` collection (same DB: ``normalize_data``):

  {cik, accession_number, form_type, filing_period, period, period_type,
   metric, standard_label, concept, statement_type, basis, form, value,
   value_low, value_high, plus_minus, plus_minus_unit, unit, scale,
   currency, as_printed, condition, event_type, supersedes, is_current,
   source, edited_by, edited_at, result, created_at, updated_at}

Normalization is deterministic — the same math ``GuidanceService.buildDoc``
applies for manual entries (form → band, midpoint for ranges, ± derivation),
so LLM-extracted and human-entered guidance land in the collection with
identical semantics.  The only difference is ``source="llm"``.

The covered ``period`` is the accuracy-critical field: it must be a FUTURE
period strictly after the filing's reported period (the canonical period-agent
result).  The filing's own guidance header is authoritative when it prints a
fiscal year; otherwise the period is derived as "the next period after this
filing" (reported Q3 FY27 → guidance Q4 FY27; reported Q4 FY26 (annual) →
guidance Q1 FY27).  Anything implausible (past period, nonsense fiscal year,
inverted band) is DROPPED with an issue string — never guessed, never saved.

Monetary values are stored in RAW units — the filing's printed scale
("thousands"/"millions"/"billions") is applied deterministically HERE before
insert, so ``value``/``value_low``/``value_high`` are comparable 1:1 with the
raw actuals in ``concept_values_*`` (e.g. "$61-64 billion" → value
62,500,000,000, low 61,000,000,000, high 64,000,000,000) and `score()` deltas
are exact.  Percentages and per-share/ratio values (EPS, margins, growth,
rates) are NEVER scaled; share counts scale when the filing prints a
magnitude.  The stored ``scale`` field keeps the filing's printed unit as
provenance (``as_printed`` carries the exact quote).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from earnings_agents.agent.loop import _coerce_number
from earnings_agents.agent.currency import is_usd_safe
from earnings_agents.agent.derive import SCALE_MULTIPLIERS
from earnings_agents.agent.period import DetectedPeriod
from earnings_agents.config import GUIDANCE_MAX_RECORDS

logger = logging.getLogger(__name__)

# ── Curated vocabulary (mirrors the admin backend's DTO enums + defaults) ──

# metric → (default standard_label, statement_type, direction)
# direction: "up" = higher actual beats ("revenue", "eps"); "down" = lower
# actual beats ("operating_expense", "capex").  Used by score() only.
# Metrics NOT listed here are still extracted and stored (free-text metric +
# custom standard_label) — they default to "up" for scoring.
METRIC_PROFILES: dict[str, tuple[str | None, str, str]] = {
    "revenue": ("Total Revenues", "income", "up"),
    "eps_diluted": ("Earnings Per Share, Diluted", "income", "up"),
    "eps_basic": ("Earnings Per Share, Basic", "income", "up"),
    "eps_adjusted": (None, "income", "up"),
    "operating_income": ("Operating Income (Loss)", "income", "up"),
    "net_income": ("Net Income (Loss)", "income", "up"),
    "gross_profit": ("Gross Profit", "income", "up"),
    "gross_margin": ("Gross Profit", "income", "up"),
    "operating_margin": (None, "income", "up"),
    "net_margin": (None, "income", "up"),
    "ebit": (None, "income", "up"),
    "ebitda": (None, "income", "up"),
    "adjusted_ebitda": (None, "income", "up"),
    "adjusted_ebit": (None, "income", "up"),
    "operating_expense": (None, "income", "down"),
    "research_development": (None, "income", "down"),
    "sales_marketing": (None, "income", "down"),
    "capex": ("Capital Expenditure", "cashflow", "down"),
    "free_cash_flow": (None, "cashflow", "up"),
    "cash_flow_from_operations": (None, "cashflow", "up"),
    "tax_rate": (None, "income", "down"),
    "revenue_growth": (None, "income", "up"),
    "dividend": (None, "cashflow", "up"),
    "share_count": (None, "income", "up"),
}

FORM_WHITELIST = {
    "point", "range", "min", "max", "plus_minus_pct", "plus_minus_abs",
    "approximate", "percentage_growth", "qualitative", "other",
}
PERIOD_TYPE_WHITELIST = {
    "quarterly", "annual", "multi_year", "ytd", "current_quarter", "other",
}
BASIS_WHITELIST = {"gaap", "non_gaap", "both", "not_applicable"}
EVENT_TYPE_WHITELIST = {
    "initial", "raised", "lowered", "reaffirmed", "narrowed", "widened",
    "withdrawn", "updated", "other",
}
SCALE_WHITELIST = {"thousands", "millions", "billions", "as-is"}

# Metrics/labels that are NEVER magnitude-scaled even when a model tags a
# monetary scale on them: per-share (EPS), percentage/ratio (margin, yield,
# growth, tax/effective rate).  Their numbers are already in the final unit.
_RATIO_OR_PER_SHARE_RE = re.compile(
    r"(eps|per\s?share|/share|margin|yield|growth|ratio|rate|pct|%)",
    re.IGNORECASE,
)


def _monetary_scalable(metric: str | None, standard_label: str | None,
                       unit: str | None, form: str) -> bool:
    """True when a record's values should be scaled to raw units.

    Percentage-unit records are excluded outright; the metric + label guard
    catches per-share/ratio metrics (EPS, margins, growth, rates) that a model
    mistakenly tagged with a monetary scale — 4.85 must never become
    4,850,000,000.  Share counts DO scale when the filing prints a magnitude
    ("2.5 billion shares" with scale "billions"); they carry scale "as-is"
    when already raw.
    """
    if form == "percentage_growth" or unit == "percent":
        return False
    hay = f"{(metric or '').strip()} {(standard_label or '').strip()}"
    return not bool(_RATIO_OR_PER_SHARE_RE.search(hay))


def _num(v: Any) -> float | None:
    """Coerce a raw value (string/number) to a float, or None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return _coerce_number(v)
    return None


def _numeric_tokens(s: str) -> list[float]:
    """All numeric tokens in a display string ("$61-64 billion" → [61, 64]).

    A leading minus is a sign ONLY when not preceded by a digit/comma — the
    hyphen in a range ("61-64", "15-17%") must not be eaten as negation.
    """
    cleaned = (s or "").replace(",", "")
    return [
        float(x)
        for x in re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?", cleaned)
    ]


def _clean_str(v: Any, max_len: int = 500) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    return s[:max_len]


def metric_profile(metric: str) -> tuple[str | None, str, str] | None:
    """Return (standard_label, statement_type, direction) for a curated metric."""
    key = (metric or "").strip().lower()
    return METRIC_PROFILES.get(key)


def period_arrived(covered: dict, filing_period: DetectedPeriod) -> bool:
    """True when the actuals for the covered period should already be stored.

    *covered* is the guidance doc's ``period`` dict.  An annual-covered
    guidance has arrived once the filing reports the same fiscal year (Q4 ==
    annual) or a later one; a quarterly-covered guidance has arrived once the
    filing reports that quarter (or later).
    """
    fy = covered.get("fiscal_year")
    if not isinstance(fy, int) or fy < 1:
        return False
    covered_q = covered.get("quarter")
    if covered_q is None:
        return fy < filing_period.fiscal_year or (
            fy == filing_period.fiscal_year
            and filing_period.period_type == "annual"
        )
    if fy < filing_period.fiscal_year:
        return True
    if fy > filing_period.fiscal_year:
        return False
    filing_q = filing_period.quarter
    if filing_period.period_type == "annual":
        return True  # the annual filing covers the whole year
    return filing_q is not None and covered_q <= filing_q


def _default_next_period(filing: DetectedPeriod) -> dict[str, Any]:
    """Derive 'the next period after this filing' for label-less guidance."""
    if filing.period_type == "quarterly" and filing.quarter in (1, 2, 3):
        nq = filing.quarter + 1
        if nq <= 3:
            return {"fiscal_year": filing.fiscal_year, "quarter": nq,
                    "period_type": "quarterly"}
        # reported Q3 → guidance Q4 of the SAME fiscal year
        return {"fiscal_year": filing.fiscal_year, "quarter": 4,
                "period_type": "quarterly"}
    # annual/Q4 filing → next fiscal year, Q1
    return {"fiscal_year": filing.fiscal_year + 1, "quarter": 1,
            "period_type": "quarterly"}


def _normalize_period(
    raw: Any,
    filing: DetectedPeriod,
    issues: list[str],
) -> dict[str, Any] | None:
    """Validate/normalize the covered period; None + issue when implausible."""
    if isinstance(raw, str):
        # Lenient: models sometimes emit the period as a label string
        # ("Q3 2026", "full year 2026", "FY2027").  Pull year/quarter off
        # the model's own output — never guess.
        m = re.search(r"(20\d{2})", raw)
        qm = re.search(r"Q\s*([1-4])", raw, re.IGNORECASE)
        raw = {
            "fiscal_year": int(m.group(1)) if m else None,
            "quarter": int(qm.group(1)) if qm else None,
            "period_type": "quarterly" if qm else "annual",
        }
    if not isinstance(raw, dict):
        issues.append("period missing — dropped")
        return None

    fy = raw.get("fiscal_year")
    try:
        fy = int(fy) if fy is not None else None
    except (TypeError, ValueError):
        fy = None
    q = raw.get("quarter")
    try:
        q = int(q) if q is not None else None
    except (TypeError, ValueError):
        q = None
    ptype = str(raw.get("period_type") or "").strip().lower()
    if not ptype:
        ptype = "quarterly" if q not in (None, 4) else "annual"

    derived_fy = False
    if fy is None:
        # Derive from 'next period' defaulting when the agent gave only a
        # quarter/type (header had no year).  Never guess a random year.
        if ptype == "annual":
            fy = filing.fiscal_year + 1
            q = None
        elif q is not None and 1 <= q <= 3:
            if filing.period_type == "quarterly" and q > filing.quarter:
                fy = filing.fiscal_year
            else:
                fy = filing.fiscal_year + 1
        elif q == 4:
            fy = filing.fiscal_year
        else:
            issues.append("period has no usable fiscal_year/quarter — dropped")
            return None
        derived_fy = True

    if fy < 1 or fy > 2100:
        issues.append(f"period fiscal_year {fy} implausible — dropped")
        return None
    if ptype in ("annual", "multi_year") or q is None:
        q = None
    elif q not in (1, 2, 3, 4):
        issues.append(f"period quarter {q} invalid — dropped")
        return None

    # FUTURE-only gate: guidance must cover a period strictly after the
    # reported filing period.  A past/same-period record is a misread (the
    # agent pasted an actual) — drop it, never save it.
    covered_fy, covered_q = fy, q
    filing_fy, filing_q = filing.fiscal_year, filing.quarter
    if ptype in ("annual", "multi_year"):
        # Same-fiscal-year annual guidance is future ONLY when it comes from a
        # quarterly filing (the full year is not reported yet — e.g. Q3 filing
        # guiding the FY).  An annual filing already reports the whole FY.
        if covered_fy < filing_fy or (
            covered_fy == filing_fy and filing.period_type != "quarterly"
        ):
            issues.append(
                f"covered period FY{covered_fy} (annual) not after filing "
                f"FY{filing_fy} — dropped"
            )
            return None
    else:
        if covered_fy < filing_fy:
            issues.append(
                f"covered period FY{covered_fy} Q{covered_q} is in the past "
                f"(filing FY{filing_fy}) — dropped"
            )
            return None
        if covered_fy == filing_fy:
            if filing.period_type == "annual":
                # An annual filing already reported the entire fiscal year —
                # same-year quarterly guidance is past/current, not future.
                issues.append(
                    f"covered period FY{covered_fy} Q{covered_q} is not future "
                    "for an annual filing — dropped"
                )
                return None
            if filing_q is not None and covered_q is not None and covered_q <= filing_q:
                issues.append(
                    f"covered period FY{covered_fy} Q{covered_q} not after "
                    f"reported Q{filing_q} — dropped"
                )
                return None

    label = _clean_str(raw.get("label"), 120)
    if not label:
        label = f"Q{q} FY{fy}" if q is not None else f"FY{fy}"
    period_end_date = _clean_str(raw.get("period_end_date"), 40)
    if period_end_date is not None:
        try:
            from earnings_agents.agent.period import parse_iso_period_end
            if parse_iso_period_end(period_end_date) is None:
                period_end_date = None
        except Exception:  # noqa: BLE001
            period_end_date = None

    return {
        "fiscal_year": fy,
        "quarter": q,
        "period_type": ptype if ptype in PERIOD_TYPE_WHITELIST else "other",
        "label": label,
        "period_end_date": period_end_date,
        "_derived_fy": derived_fy,  # internal marker, stripped before save
    }


def _period_type_binary(period_doc: dict[str, Any]) -> str:
    """Derive the top-level ``quarterly | annual`` classification of a
    guidance doc from its normalized ``period`` dict.

    The extended ``period.period_type`` enum (annual/multi_year/ytd/
    current_quarter/other) is preserved inside ``period``; the top-level
    field is the binary class used for filtering/display:

      annual / multi_year        -> "annual"
      quarterly / current_quarter -> "quarterly"
      ytd                         -> "quarterly"
      other / missing             -> fall back on quarter presence
                                     (quarter set -> quarterly, else annual)
    """
    ptype = str((period_doc or {}).get("period_type") or "").strip().lower()
    if ptype in ("annual", "multi_year"):
        return "annual"
    if ptype in ("quarterly", "ytd", "current_quarter"):
        return "quarterly"
    q = (period_doc or {}).get("quarter")
    return "quarterly" if q is not None else "annual"


def normalize_guidance_records(
    raw_records: Any,
    filing_period: DetectedPeriod,
    doc_currency: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize the agent's ``__guidance__`` payload into guidance_values docs.

    Returns ``(records, issues)`` — *records* are stripped of all internal
    keys (``_*`` / ``lines``) except one ``_lines`` evidence key the save node
    may persist for traceability; *issues* are human-readable drop reasons
    (observability only — never a run failure).
    """
    import json

    if isinstance(raw_records, str):
        try:
            raw_records = json.loads(raw_records)
        except json.JSONDecodeError:
            return [], ["__guidance__ is not valid JSON"]
    if isinstance(raw_records, dict):
        # Lenient dict form: {"revenue": {...}, "capex": {...}} — models often
        # key records by metric name instead of a plain list.  Merge the key
        # into each record (already-set metric values win).
        _WRAPPER_KEYS = {"guidance", "items", "records", "list", "forecast", "outlook"}
        merged: list[dict] = []
        for key, val in raw_records.items():
            if isinstance(val, dict):
                rec = dict(val)
                if key not in _WRAPPER_KEYS:
                    rec.setdefault("metric", str(key))
                merged.append(rec)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        rec = dict(item)
                        if key not in _WRAPPER_KEYS:
                            rec.setdefault("metric", str(key))
                        merged.append(rec)
        raw_records = merged
    if not isinstance(raw_records, list):
        return [], ["__guidance__ is not a list"]

    records: list[dict[str, Any]] = []
    issues: list[str] = []
    cap = max(1, GUIDANCE_MAX_RECORDS or 15)

    for i, raw in enumerate(raw_records):
        if i >= cap:
            issues.append(f"guidance list capped at {cap} records")
            break
        if not isinstance(raw, dict):
            issues.append(f"record {i + 1}: not an object — dropped")
            continue

        # metric + curated defaults
        metric = _clean_str(raw.get("metric"), 80) or "other"
        profile = metric_profile(metric)
        default_label, default_statement, _direction = (
            profile if profile else (None, "income", "up")
        )
        standard_label = (
            _clean_str(raw.get("standard_label"), 200)
            or default_label
            or metric
        )
        statement_type = _clean_str(raw.get("statement_type"), 40) or default_statement
        basis = str(raw.get("basis") or "gaap").strip().lower()
        if basis not in BASIS_WHITELIST:
            basis = "gaap"

        form = str(raw.get("form") or "").strip().lower()
        if form not in FORM_WHITELIST:
            form = "point" if raw.get("value") is not None else "qualitative"

        raw_value = raw.get("value")
        value = _num(raw_value)
        pct_hint = isinstance(raw_value, str) and "%" in raw_value
        parsed_low = parsed_high = None
        if value is None and isinstance(raw_value, str):
            # Lenient: models often emit the number as a DISPLAY STRING from
            # the filing ("$61-64 billion", "61 to 64", "approximately
            # $108.0 billion", "between 15-17%").  This parses the model's
            # OWN output field — never the document text, never a guess.
            tokens = _numeric_tokens(raw_value)
            if len(tokens) >= 2 and tokens[0] < tokens[1]:
                parsed_low, parsed_high = tokens[0], tokens[1]
                value = (parsed_low + parsed_high) / 2
                if form in ("", "point", "qualitative", "other"):
                    form = "range"
            elif len(tokens) == 1:
                value = tokens[0]
                if form in ("", "point", "qualitative", "other"):
                    form = "approximate"
        low = _num(raw.get("value_low"))
        high = _num(raw.get("value_high"))
        if low is None and parsed_low is not None:
            low = parsed_low
        if high is None and parsed_high is not None:
            high = parsed_high
        plus_minus = _num(raw.get("plus_minus"))
        pm_unit = _clean_str(raw.get("plus_minus_unit"), 20)

        # form → band (same math as GuidanceService.buildDoc)
        if form == "plus_minus_pct" and value is not None and plus_minus is not None:
            low = value * (1 - plus_minus / 100)
            high = value * (1 + plus_minus / 100)
            pm_unit = pm_unit or "percent"
        elif form == "plus_minus_abs" and value is not None and plus_minus is not None:
            low = value - plus_minus
            high = value + plus_minus
            pm_unit = pm_unit or "absolute"
        elif low is not None and high is not None and value is None:
            value = (low + high) / 2  # range midpoint
        elif form in ("min", "max") and low is None and high is None and value is not None:
            if form == "min":
                low, high = value, None
            else:
                low, high = None, value

        if form == "qualitative":
            value = value if value is not None and low is None else None
            if value is not None:
                issues.append(f"record {i + 1}: qualitative with a number — kept as qualitative")
                value = None
        elif value is None and low is None and high is None and plus_minus is None:
            issues.append(f"record {i + 1} ({metric}): no number at all — dropped")
            continue

        if low is not None and high is not None and low > high:
            issues.append(f"record {i + 1} ({metric}): value_low > value_high — dropped")
            continue

        if form == "percentage_growth":
            unit = "percent"
            scale = "as-is"
        else:
            unit = _clean_str(raw.get("unit"), 20)
            scale = _clean_str(raw.get("scale"), 20)
            if scale and scale not in SCALE_WHITELIST:
                scale = None
        if pct_hint and form not in ("percentage_growth",):
            unit = "percent"  # "between 15-17%" parsed from a display string
        if unit is None and value is not None and form not in ("percentage_growth",):
            unit = "USD"  # monetary default — matches the backend default
        currency = (_clean_str(raw.get("currency"), 10) or "").upper() or None
        if not currency:
            currency = (doc_currency or "").upper() or "USD"

        # per-record currency gate (belt-and-braces; doc-level gate already ran)
        monetary = form not in ("percentage_growth",)
        if monetary and value is not None and not is_usd_safe(currency):
            issues.append(
                f"record {i + 1} ({metric}): currency {currency} not USD — "
                "non-USD monetary guidance not saved"
            )
            continue

        # ── Magnitude scaling → store RAW units (with zeros) ─────────────────
        # Monetary guidance values land in `guidance_values` in the same raw
        # unit concept_values_* stores actuals (e.g. 62,500,000,000 for
        # "$61-64 billion"), so scoring (delta vs the stored actual) is
        # apples-to-apples.  The filing's printed scale is applied HERE once —
        # never scaled again downstream; the stored `scale` field keeps the
        # printed unit as provenance.  Percentages and per-share/ratio values
        # are never scaled (they arrive scale "as-is"); share counts scale
        # when the filing prints a magnitude ("2.5 billion shares").
        multiplier = SCALE_MULTIPLIERS.get(scale or "")
        if multiplier and _monetary_scalable(metric, standard_label, unit, form):
            if value is not None:
                value *= multiplier
            if low is not None:
                low *= multiplier
            if high is not None:
                high *= multiplier
            # plus_minus_abs is in the same monetary unit — scale it too;
            # plus_minus_pct is a percentage and stays untouched.
            if plus_minus is not None and pm_unit != "percent":
                plus_minus *= multiplier
        # value sanity on FINAL stored units: reject absurd magnitudes
        if value is not None and abs(value) > 1e16:
            issues.append(
                f"record {i + 1} ({metric}): value {value} implausible — dropped"
            )
            continue

        period = _normalize_period(raw.get("period"), filing_period, issues)
        if period is None:
            continue
        period_doc = {k: v for k, v in period.items() if not k.startswith("_")}
        # Top-level binary class (quarterly|annual) — derived from period, never
        # accepted as raw input so the two can never drift apart.
        period_type = _period_type_binary(period_doc)

        lines = raw.get("lines")
        lines_out = None
        if isinstance(lines, (list, tuple)) and len(lines) == 2:
            try:
                lines_out = [int(lines[0]), int(lines[1])]
            except (TypeError, ValueError):
                lines_out = None

        event_type = str(raw.get("event_type") or "initial").strip().lower()
        if event_type not in EVENT_TYPE_WHITELIST:
            event_type = "initial"

        doc: dict[str, Any] = {
            "metric": metric,
            "standard_label": standard_label,
            "concept": _clean_str(raw.get("concept"), 300),
            "statement_type": statement_type,
            "basis": basis,
            "form": form,
            "value": value,
            "value_low": low,
            "value_high": high,
            "plus_minus": plus_minus,
            "plus_minus_unit": pm_unit,
            "unit": unit,
            "scale": scale,
            "currency": currency or "USD",
            "as_printed": _clean_str(raw.get("as_printed"), 500),
            "condition": _clean_str(raw.get("condition"), 300),
            "event_type": event_type,
            "period": period_doc,
            "period_type": period_type,
            "is_current": True,
            "source": "llm",
            "result": None,
        }
        if lines_out is not None:
            doc["_lines"] = lines_out  # internal evidence — save node strips it
        records.append(doc)

    return records, issues


def score(guidance_doc: dict[str, Any], actual_value: float,
          actual_accession: str | None = None) -> dict[str, Any]:
    """Score one guidance doc against a stored actual.

    Mirrors the plan's ``score()``: lo/hi from the stored band, outcome per
    metric direction (revenue/EPS up-is-good, opex/capex down-is-good).
    Floor/ceiling forms (min/max) are hard limits — crossing the bound is a
    miss regardless of metric direction; point/range/approximate use the
    band with the metric-direction flip.  Returns the doc's ``result``.
    """
    metric = str(guidance_doc.get("metric") or "").strip().lower()
    profile = metric_profile(metric)
    up_is_good = profile[2] != "down" if profile else True

    value = _num(guidance_doc.get("value"))
    form = str(guidance_doc.get("form") or "point").lower()
    if form == "qualitative" or value is None:
        return {
            "actual_value": actual_value,
            "actual_accession_number": actual_accession,
            "delta_abs": None,
            "delta_pct": None,
            "outcome": "qualitative",
        }

    lo = _num(guidance_doc.get("value_low"))
    hi = _num(guidance_doc.get("value_high"))

    # Floor/ceiling — absolute bounds, no metric-direction flip.
    if form == "min":
        outcome = "beat" if actual_value > value else "miss"
    elif form == "max":
        outcome = "beat" if actual_value < value else "miss"
    else:
        if lo is None and hi is None:
            lo = hi = value
        if actual_value > hi:
            outcome = "beat" if up_is_good else "miss"
        elif actual_value < lo:
            outcome = "miss" if up_is_good else "beat"
        else:
            outcome = "inline"

    delta_abs = actual_value - value
    delta_pct = (
        (delta_abs / abs(value)) * 100 if value not in (None, 0) else None
    )
    return {
        "actual_value": actual_value,
        "actual_accession_number": actual_accession,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
        "outcome": outcome,
    }