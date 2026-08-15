"""Agent tool-calling loop (ReAct-style).

Runs the agent loop: system prompt → LLM with bound tools → parse tool calls →
execute tools → feed results back → repeat until finalize or max steps.

Shared by both the pipeline node and the lightweight extractor.
"""
from __future__ import annotations

import json
import logging
import re
from itertools import count
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from earnings_agents.agent.prompts import FINALIZE_DESCRIPTION
from earnings_agents.config import LLM_PROVIDER
from earnings_agents.hooks import report_call
from earnings_agents.llm import build_chat_llm

logger = logging.getLogger(__name__)

# ── Response parsing ────────────────────────────────────────────────────────

_PCT_OR_PER_SHARE_PATTERNS = re.compile(
    r"(%|percent|margin|yield|growth"
    r"|\bratio\b|\bratios\b|\beps\b|per share|\byoy\b|\bpct\b|\brate\b"
    r"|employee|headcount|basis points|percentage points"
    r"|\bper\s+(?:basic|diluted|basic\s+and\s+diluted|common)\s+share\b"
    r"|per\w*share"
    r"|production|deliveries|delivered"
    r"|(?:super)?charger.{0,12}(?:station|connector)"
    r"|\bstations?\b|\bconnectors?\b"
    r"|\bdays.{0,5}supply\b|\blease count\b"
    r"|\bactive\b.{0,20}\bsubscriptions?\b|\bfsd subscriptions?\b)",
    re.IGNORECASE,
)

_SHARE_COUNT_PATTERN = re.compile(
    r"\bnumber of shares\b|\bshares used\b|\bweighted.{0,15}average.{0,15}shares\b",
    re.IGNORECASE,
)

_SHARE_COUNT_RAW_MAX = 100_000_000
_TABLE_RAW_MAX = 10_000_000
_IMPLAUSIBLE_ABS_USD = _TABLE_RAW_MAX * 1_000_000


def _parse_llm_response(
    response: str,
    shares_multiplier: int = 1,
    prescan_dollar_multiplier: int = 0,
) -> dict[str, Any] | None:
    """Strip markdown fences, parse JSON, and apply the __scale__ multiplier."""
    from earnings_agents.agent.derive import SCALE_MULTIPLIERS

    cleaned = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]
    end_brace = cleaned.rfind("}")
    if end_brace >= 0:
        cleaned = cleaned[: end_brace + 1]
    try:
        parsed: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    scale_str = str(parsed.pop("__scale__", "as-is")).lower()
    llm_multiplier = SCALE_MULTIPLIERS.get(scale_str, 1)
    if llm_multiplier > 1 and prescan_dollar_multiplier > 1:
        multiplier = min(llm_multiplier, prescan_dollar_multiplier)
    elif prescan_dollar_multiplier > 1:
        multiplier = prescan_dollar_multiplier
    else:
        multiplier = llm_multiplier if llm_multiplier > 1 else 1

    table_raw_max = _IMPLAUSIBLE_ABS_USD // multiplier if multiplier > 1 else _TABLE_RAW_MAX
    if multiplier > 1 or shares_multiplier > 1:
        for k, v in list(parsed.items()):
            if v is None or not isinstance(v, (int, float)):
                continue
            is_share_count = bool(_SHARE_COUNT_PATTERN.search(k))
            if is_share_count and shares_multiplier > 1:
                if abs(v) < _SHARE_COUNT_RAW_MAX:
                    parsed[k] = v * shares_multiplier
            elif (
                not is_share_count
                and not _PCT_OR_PER_SHARE_PATTERNS.search(k)
                and multiplier > 1
                and abs(v) < table_raw_max
            ):
                if "gross margin" in k.lower() and abs(v) <= 100:
                    continue
                parsed[k] = v * multiplier

    return parsed


def run_agent_loop(
    system_prompt: str,
    initial_message: str,
    tools: list,
    *,
    ticker: str = "?",
    dollar_multiplier: int = 1,
    max_steps: int | None = None,
    finalize_name: str = "finalize_extraction",
    finalize_description: str = FINALIZE_DESCRIPTION,
    parse_final_result: Callable[[str], dict[str, Any] | None] | None = None,
    recovery_regex: re.Pattern | None = None,
) -> dict[str, Any] | None:
    """Run the tool-calling agent loop — shared by extraction and period detection.

    The loop is OPEN-ENDED by default: it runs until the agent calls the
    terminal finalize tool (or the LLM call fails).  No step cap.

    Args:
        system_prompt: Full system prompt.
        initial_message: First human message to the agent.
        tools: List of LangChain tools; the terminal finalize tool is appended here.
        ticker: For logging.
        dollar_multiplier: Scale multiplier for the default (extraction) parser.
        max_steps: Optional hard step cap (None = open-ended, the default).
        finalize_name: Name of the terminal tool (extraction: ``finalize_extraction``,
            period detection: ``finalize_period``).
        finalize_description: Docstring for the terminal tool.
        parse_final_result: Parser for the terminal tool's JSON string.  Defaults
            to the scale-aware extraction parser bound to *dollar_multiplier*.
        recovery_regex: Pattern used to recover a final JSON blob from the last AI
            message when the agent never called the terminal tool.  Defaults to the
            ``__scale__``-keyed extraction pattern.

    Returns:
        The parsed final result dict,
        or ``None`` if the agent failed to produce a result.
    """
    if parse_final_result is None:
        parse_final_result = lambda s: _parse_llm_response(s, 1, dollar_multiplier)  # noqa: E731
    if recovery_regex is None:
        recovery_regex = re.compile(r'\{[^{}]*"__scale__"[^{}]*\}', re.DOTALL)

    # Add finalize tool
    from langchain_core.tools import tool as _lc_tool

    @_lc_tool
    def finalize_tool(result_json: str) -> str:
        """Call this when done. Pass a JSON string with all results."""
        return result_json
    finalize_tool.name = finalize_name
    finalize_tool.description = finalize_description
    all_tools = list(tools) + [finalize_tool]

    # Build chat model
    try:
        chat_llm = build_chat_llm()
    except ValueError as exc:
        logger.warning("Agent loop unavailable for %s: %s", ticker, exc)
        return None

    llm_with_tools = chat_llm.bind_tools(all_tools)
    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=initial_message),
    ]

    final_result: dict[str, Any] | None = None

    steps = count(1) if not max_steps else range(1, max_steps + 1)
    for step in steps:
        logger.debug("Agent step %d for %s", step, ticker)
        # Matches the "→ calling llm" convention consumed by the CLI printer
        # and worker progress publisher (LLM-call counting + highlighting).
        report_call(
            f"  [llm]  agent step {step}  → calling llm  "
            f"({LLM_PROVIDER or 'llm'})"
        )

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as exc:
            logger.error("Agent LLM call failed at step %d for %s: %s", step, ticker, exc)
            break

        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            content = getattr(response, "content", "") or ""
            if "finalize" in content.lower() or "{" in content:
                break
            messages.append(HumanMessage(
                content="Use your tools to explore the document (search, read_lines, "
                f"get_document_info). When done, call {finalize_name}."
            ))
            continue

        tool_messages: list[ToolMessage] = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id", "")

            logger.info(
                "Agent step %d for %s → tool: %s(%s)",
                step, ticker, tool_name,
                str(tool_args)[:120],
            )
            # Surface tool calls on the CLI progress display and worker
            # event stream (both consume report_call).  Finalize args carry
            # the full metrics JSON — too large for a progress line, and
            # already logged at INFO above.
            if tool_name == finalize_name:
                report_call(f"  [tool]  step {step} → {finalize_name}()")
            else:
                args_brief = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
                if len(args_brief) > 110:
                    args_brief = args_brief[:110] + "…"
                report_call(f"  [tool]  step {step} → {tool_name}({args_brief})")

            if tool_name == finalize_name:
                try:
                    raw_result = tool_args.get("result_json", "")
                    result_str = raw_result if isinstance(raw_result, str) else json.dumps(raw_result)
                except Exception:
                    result_str = str(tool_args)

                # Log the raw JSON for debugging
                logger.info("Agent finalize JSON for %s: %s", ticker, result_str[:1000])

                parsed = parse_final_result(result_str)
                if parsed is None:
                    report_call(
                        f"  [tool]  ✗ {finalize_name} — could not parse result JSON"
                    )
                if parsed is not None:
                    final_result = parsed
                    logger.info(
                        "Agent finalize for %s: %d keys extracted",
                        ticker,
                        len([k for k in final_result if not k.startswith("__")]),
                    )
                tool_messages.append(ToolMessage(
                    content=(
                        "Finalize received. Extraction complete."
                        if parsed is not None
                        else "Failed to parse result JSON. Ensure it's valid JSON "
                             "and dates use YYYY-MM-DD format."
                    ),
                    tool_call_id=tool_call_id,
                ))
                messages.extend(tool_messages)
                break
            else:
                tool_fn = next((t for t in all_tools if t.name == tool_name), None)
                if tool_fn is not None:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                else:
                    result = f"Unknown tool: {tool_name}"

                if isinstance(result, str) and result.startswith(("Tool error", "Unknown tool")):
                    report_call(f"  [tool]  ✗ {result[:120]}")

                if isinstance(result, str) and len(result) > 8000:
                    result = result[:8000] + "\n... (truncated)"
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

        messages.extend(tool_messages)
        if final_result is not None:
            break

    # Fallback: try to recover JSON from last message
    if final_result is None:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = getattr(msg, "content", "") or ""
                json_match = recovery_regex.search(content)
                if json_match:
                    parsed = parse_final_result(json_match.group())
                    if parsed is not None:
                        final_result = parsed
                        logger.info("Agent loop: recovered %d keys from final message",
                                    len([k for k in final_result if not k.startswith("__")]))
                        break

    if final_result is None:
        logger.error("Agent for %s: no result produced", ticker)
        return None

    return final_result
