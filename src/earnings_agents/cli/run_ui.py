"""Shared Rich progress presentation for one company earnings run.

Used by BOTH the multi-ticker CLI (``cli/earnings.py::_run_company_parallel``)
and the historical backfill CLI (``cli/backfill.py``) so a backfill shows the
identical stage / step-summary / ``[llm]`` / ``[tool]`` / ``[db]`` trace that a
single ``uv run earnings --ticker X`` run shows.

``RunPresenter`` installs the thread-local hooks callbacks, renders a
terminal-width-aware progress description, and prints the per-node step
summaries.  It works with a live ``rich.progress.Progress`` (the CLI) or with a
plain ``Console`` (headless / tests).
"""
from __future__ import annotations

from typing import Any, Optional

from rich.console import Console
from rich.markup import escape as _escape

# Human-readable stage names for the progress description.
NODE_LABELS: dict[str, str] = {
    "fetch_filing_node":              "fetch",
    "detect_period_node":             "period",
    "check_period_node":              "check period",
    "load_company_concepts_node":     "load concepts",
    "detect_document_type_node":      "detect type",
    "mongodb_save_node":              "save",
    "save_guidance_node":             "guidance",
    "calculate_q4_node":              "q4",
    "agent_document_pipeline_node":   "agent pipeline",
}

SHORT_STAGE: dict[str, str] = {
    "fetch":          "fetch",
    "period":         "period",
    "check period":   "check",
    "load concepts":  "cncpt",
    "detect type":    "type",
    "save":           "save",
    "guidance":       "guid",
    "q4":             "q4",
    "agent pipeline": "agent",
}


def format_result_detail(state: dict) -> str:
    """Return the compact period/concept detail shown after a finished run.

    e.g. ``"FY2026 Q2, 71 concepts"``.  Empty string when nothing is known.
    """
    label = ""
    try:
        from earnings_agents.agent.period import (
            format_period_label,
            require_detected_period,
        )
        label = format_period_label(require_detected_period(state))
    except Exception:  # noqa: BLE001 — presentation only
        label = ""
    n = len(state.get("concept_metrics") or {})
    if label and n:
        return f"{label}, {n} concept(s)"
    if label:
        return label
    if n:
        return f"{n} concept(s)"
    return ""


def fmt_duration(ms: float) -> str:
    """Format a millisecond duration as a human-readable string.

    Examples: 450 ms → "0.5s", 5300 ms → "5.3s", 196200 ms → "3m 16s".
    """
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    return f"{m}m {s - m * 60:.0f}s"


def format_step_line(node_name: str, state: dict) -> Optional[str]:
    """Return a one-line human-readable summary for a completed node.

    Returns ``None`` when no summary is needed.
    """
    if node_name == "load_company_concepts_node":
        if state.get("status") in ("skipped", "failed"):
            reason = (state.get("error") or "no concepts")[:70]
            return f"  [load concepts]  skipped — {reason}"
        n = len(state.get("target_concepts") or [])
        try:
            from earnings_agents.agent.period import require_detected_period
            pt = require_detected_period(state).period_type
        except Exception:
            pt = "unknown"
        return f"  [load concepts]  {n} concepts  ({pt})"

    if node_name == "detect_document_type_node":
        if state.get("status") == "failed":
            return f"  [detect type]    failed — {(state.get('error') or '')[:60]}"
        return f"  [detect type]    {state.get('file_type') or '?'}"

    if node_name == "detect_period_node":
        if state.get("status") in ("failed", "skipped"):
            return f"  [period]         ✗  {(state.get('error') or '')[:70]}"
        try:
            from earnings_agents.agent.period import require_detected_period
            period = require_detected_period(state)
            label = period.period_label or period.period_end.isoformat()
            return f"  [period]         ✓  {period.period_type}  {label}"
        except Exception as exc:  # defensive: hooks must not break the CLI
            return f"  [period]         ✗  invalid result — {str(exc)[:50]}"

    if node_name == "check_period_node":
        s = state.get("status", "?")
        if s == "failed":
            return f"  [check period]   ✗  {(state.get('error') or '')[:70]}"
        if state.get("_pending_replace"):
            lbl = state.get("_replace_period_label") or "existing period"
            return f"  [check period]   re-extract {lbl} — replace after successful save"
        return "  [check period]   new period — proceeding"

    if node_name == "mongodb_save_node":
        s = state.get("status", "?")
        if s == "failed":
            return f"  [save]           ✗  {(state.get('error') or 'failed')[:70]}"
        # Success is already fully reported by the in-node [save]/[db] lines —
        # no extra summary line.
        return None

    if node_name == "agent_document_pipeline_node":
        metrics = state.get("metrics") or {}
        n_metrics = len([k for k in metrics if not k.startswith("__")])
        if state.get("status") == "failed":
            return f"  [agent pipe]  failed — {(state.get('error') or '')[:60]}"
        concept_metrics = state.get("concept_metrics") or {}
        derived_ids = set(state.get("derived_concept_ids") or [])
        target_concepts = state.get("target_concepts") or []
        n_total = len(target_concepts) + len(state.get("calculated_concepts") or [])
        n_mapped = len(concept_metrics) - len(derived_ids)
        n_derived = len(derived_ids)
        n_absent = len([c for c in target_concepts if c["_id"] not in concept_metrics])
        lines = [f"  [agent pipe]  → {n_metrics} metrics extracted"]
        if n_total:
            lines.append(
                f"  [agent pipe]  concepts: "
                f"{n_mapped} mapped  {n_derived} derived  {n_absent} not in filing"
                f"  (of {n_total})"
            )
        return "\n".join(lines)

    return None


class RunPresenter:
    """Renders one filing's progress exactly like the single-ticker CLI.

    ``progress``/``task`` are a live Rich progress bar and its task id.  When
    omitted, a plain ``Console`` is used (headless runs, tests) and the same
    lines are printed without a spinner.
    """

    def __init__(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        progress: Any | None = None,
        task: Any | None = None,
    ) -> None:
        self.name = name
        self.label = label or f"[cyan]{name}[/]"
        self.progress = progress
        self.task = task
        self._console: Console = progress.console if progress is not None else Console()
        self._completed: list[str] = []
        self._stage = "resolving\u2026"
        self._detail = ""
        self._extract_pass = 0
        self._last_chunk_done = 0
        # Total LLM API requests fired for this filing (same counter the
        # single-ticker CLI reports as "(N LLM calls)").
        self.llm_calls = 0
        # When True, the per-call [llm]/[tool]/[db] lines are suppressed (the
        # backfill CLI's --quiet); stage/step summaries still print.
        self.quiet = False

    # ── binding ────────────────────────────────────────────────────────
    def bind(self, progress: Any, task: Any, label: Optional[str] = None) -> None:
        """Point the presenter at a new live progress row and reset per-run state.

        The backfill CLI calls this once per filing so each filing gets its own
        progress row while printing the same stage/step trace.
        """
        self.progress = progress
        self.task = task
        if progress is not None:
            self._console = progress.console
        if label is not None:
            self.label = label
        self._completed.clear()
        self._stage = "resolving\u2026"
        self._detail = ""
        self._extract_pass = 0
        self._last_chunk_done = 0
        self.llm_calls = 0

    # ── printing ────────────────────────────────────────────────────────
    def _update(self) -> None:
        if self.progress is not None and self.task is not None:
            self.progress.update(self.task, description=self._render())

    def _print(self, markup: str) -> None:
        self._console.print(markup)

    def _render(self) -> str:
        """Build a terminal-width-aware progress description.

        Completed stages are abbreviated (e.g. "✓disc ✓type ✓fetch").  When the
        full string exceeds the available width, the breadcrumb is replaced
        with a compact count ("(3✓)") and, if still too long, the active-stage
        detail is truncated to fit.
        """
        try:
            term_w = self._console.width
        except Exception:  # noqa: BLE001
            term_w = 80
        # Overhead: spinner(2) + bar(36) + timer(10) + padding(6) ≈ 54 chars
        max_chars = max(30, term_w - 54)

        active_stage = self._stage
        detail = self._detail
        active_text = f"{active_stage}: {detail}" if detail else active_stage

        done_parts = [f"\u2713{SHORT_STAGE.get(s, s[:5])}" for s in self._completed]
        done_plain = " ".join(done_parts)

        sep = "  \u2192 "
        full_plain = (
            f"{self.name}  {done_plain}{sep}{active_text}"
            if done_parts
            else f"{self.name}  \u2192 {active_text}"
        )
        if len(full_plain) <= max_chars:
            done_markup = f"[green]{done_plain}[/]" if done_parts else ""
            return f"{self.label}  {done_markup}{sep}[bold]{active_text}[/]"

        n = len(self._completed)
        count_plain = f"{self.name}  ({n}\u2713){sep}{active_text}"
        if len(count_plain) <= max_chars:
            count_markup = f"[green]({n}\u2713)[/]" if n > 0 else ""
            return f"{self.label}  {count_markup}{sep}[bold]{active_text}[/]"

        avail = max_chars - len(self.name) - len(active_stage) - 8
        short_detail = detail[:max(0, avail)] if detail else ""
        trunc_ellipsis = "\u2026" if detail and len(detail) > avail else ""
        active_trunc = (
            f"{active_stage}: {short_detail}{trunc_ellipsis}" if short_detail else active_stage
        )
        count_markup = f"[green]({n}\u2713)[/]  " if n > 0 else ""
        return f"{self.label}  {count_markup}[bold]\u2192 {active_trunc}[/]"

    # ── hooks callbacks ─────────────────────────────────────────────────
    def node_cb(
        self,
        node_name: str,
        event: str,
        _ticker: str,
        node_state: Any = None,
        elapsed_ms: float | None = None,
    ) -> None:
        stage = NODE_LABELS.get(node_name, node_name.replace("_node", ""))
        if event == "start":
            if node_name == "agent_document_pipeline_node":
                self._extract_pass += 1
                if self._extract_pass > 1:
                    stage = f"re-extract #{self._extract_pass}"
            self._stage = stage
            self._detail = ""
            self._update()
            self._print(f"[dim]{self.name}[/]  [bold cyan]▶ {_escape(stage)}[/]")
            return
        if event != "end":
            return
        finished = self._stage
        if finished not in self._completed:
            self._completed.append(finished)
        self._update()
        if node_state is not None:
            line = format_step_line(node_name, node_state)
            if line:
                dur = f"  [dim]{fmt_duration(elapsed_ms)}[/]" if elapsed_ms is not None else ""
                parts = line.split("\n")
                for i, part in enumerate(parts):
                    suffix = dur if i == len(parts) - 1 else ""
                    self._print(f"[dim]{self.name}[/]{_escape(part)}{suffix}")

    def detail_cb(self, detail: str) -> None:
        self._detail = detail
        self._update()
        # Print one line per chunk completion.  The detail string looks like:
        #   "chunks 1✓ 2✓ 3⟳ (2/4)"
        if "chunks " in detail and ("✓" in detail or "✗" in detail):
            import re as _re
            m = _re.search(r"\((\d+)/(\d+)\)", detail)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                if done > self._last_chunk_done:
                    self._last_chunk_done = done
                    pass_label = (
                        f"re-extract #{self._extract_pass}" if self._extract_pass > 1 else "extract"
                    )
                    self._print(
                        f"[dim]{self.name}[/]  {_escape(f'[{pass_label}]')}  "
                        f"chunk {done}/{total}  " + _escape(detail.split("(")[0].strip())
                    )

    def call_cb(self, msg: str) -> None:
        if self.quiet:
            return
        esc_msg = _escape(msg)
        # Highlight LLM-related messages (direct calls and LLM-triggered steps)
        # distinctly so they stand out from DB/HTTP operational messages.
        is_llm = " [llm]" in msg or ("llm" in msg.lower() and "\u2192" in msg)
        if is_llm:
            if "\u2192 calling llm" in msg:
                self.llm_calls += 1
            self._print(f"[dim]{self.name}[/][bold yellow]{esc_msg}[/]")
        elif "[industry]" in msg:
            # Single industry-context line per extraction pass (SIC injected).
            self._print(f"[dim]{self.name}[/][bold cyan]{esc_msg}[/]")
        else:
            self._print(f"[dim]{self.name}[/]{esc_msg}")

    # ── installation ────────────────────────────────────────────────────
    def install(self) -> None:
        """Bind all three hooks callbacks to this presenter (thread-local)."""
        from earnings_agents.hooks import (
            set_call_callback,
            set_detail_callback,
            set_node_callback,
        )
        set_node_callback(self.node_cb)
        set_detail_callback(self.detail_cb)
        set_call_callback(self.call_cb)

    @staticmethod
    def uninstall() -> None:
        from earnings_agents.hooks import (
            set_call_callback,
            set_detail_callback,
            set_node_callback,
        )
        set_node_callback(None)
        set_detail_callback(None)
        set_call_callback(None)

    # ── completion ──────────────────────────────────────────────────────
    def result_description(self, result: dict) -> tuple[str, bool]:
        """Return ``(rich_description, ok)`` for a finished run's progress row."""
        status = str(result.get("status") or "?")
        llm_tag = f"  [dim]({self.llm_calls} LLM calls)[/]" if self.llm_calls else ""
        detail = format_result_detail(result)
        detail_tag = f"  [dim]({_escape(detail)})[/]" if detail else ""
        if status == "saved":
            return f"[green]{self.name}[/]  saved \u2713{llm_tag}{detail_tag}", True
        if status == "failed":
            return f"[red]{self.name}[/]  failed \u2717{llm_tag}{detail_tag}", False
        return f"[yellow]{self.name}[/]  {status}{detail_tag}", False