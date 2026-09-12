"""One-off backfill: top-level ``period_type`` (quarterly|annual) on legacy
``guidance_values`` docs.

Before the top-level ``period_type`` field existed, guidance docs carried only
the nested ``period.period_type`` extended enum.  This script derives and
persists the binary classification for every doc missing it:

  annual / multi_year     -> "annual"
  quarterly / ytd / current_quarter -> "quarterly"
  other / missing         -> fall back on quarter presence (set -> quarterly)

Idempotent and safe to re-run: docs that already carry the field are skipped.
Run with the same MONGODB_URI as the worker:

    uv run python scripts/backfill_guidance_period_type.py
"""
from __future__ import annotations

import sys
from typing import Any

from earnings_agents.integrations.normalize import _get_client, _NORMALIZE_DB


def derive(period: dict[str, Any] | None) -> str:
    ptype = str((period or {}).get("period_type") or "").strip().lower()
    if ptype in ("annual", "multi_year"):
        return "annual"
    if ptype in ("quarterly", "ytd", "current_quarter"):
        return "quarterly"
    q = (period or {}).get("quarter")
    return "quarterly" if q is not None else "annual"


def main() -> int:
    db = _get_client()[_NORMALIZE_DB]
    col = db["guidance_values"]
    docs = col.find({"period_type": {"$exists": False}})
    n_missing = n_updated = 0
    for d in docs:
        n_missing += 1
        period_type = derive(d.get("period"))
        col.update_one(
            {"_id": d["_id"]},
            {"$set": {"period_type": period_type}},
        )
        n_updated += 1
    print(f"guidance_values: {n_updated}/{n_missing} legacy doc(s) backfilled "
          f"with top-level period_type")
    return 0 if n_updated == n_missing else 1


if __name__ == "__main__":
    sys.exit(main())