"""LangGraph workflow — agent pipeline.

Graph:
    fetch_filing → detect_period → check_period → load_company_concepts
        → agent_document_pipeline → mongodb_save → calculate_q4 → END

The reporting period is decided ONLY by the period agent (detect_period); if it
fails the run fails — no deterministic period inference anywhere.

``calculate_q4`` is the post-save Q4 derivation (income statement only): after
an ANNUAL filing is saved to ``concept_values_annual`` it derives
Q4 = Annual − (Q1 + Q2 + Q3) and inserts the quarterly values into
``concept_values_quarterly`` (see ``nodes/q4.py`` / ``integrations/q4.py``).
It runs only after a successful save and never fails the run.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from earnings_agents.agent.period import detect_period_node
from earnings_agents.agent.pipeline import agent_document_pipeline_node
from earnings_agents.nodes.check import check_period_node
from earnings_agents.nodes.concepts import load_company_concepts_node
from earnings_agents.nodes.fetch import fetch_filing_node
from earnings_agents.nodes.q4 import calculate_q4_node
from earnings_agents.nodes.save import mongodb_save_node
from earnings_agents.state import EarningsAgentState
from earnings_agents.hooks import with_hooks

logger = logging.getLogger(__name__)

# Statuses that short-circuit the remaining pipeline.
_SHORT_CIRCUIT = ("failed", "skipped")


def _route_after(next_node: str):
    """Route to *next_node*, or END when the run failed / skipped / is stale."""

    def _route(state: EarningsAgentState) -> str:
        if state.get("status") in _SHORT_CIRCUIT:
            return "__end__"
        return next_node

    return _route


def _route_after_save(state: EarningsAgentState) -> str:
    """Run the post-save Q4 derivation only after a successful save.

    The node itself no-ops for quarterly periods, disabled config, or missing
    CIK/concepts — this route only avoids invoking it when the save failed.
    """
    if state.get("status") == "saved":
        return "calculate_q4"
    return "__end__"


def build_graph():
    """Compile and return the LangGraph earnings scraping workflow."""
    graph = StateGraph(EarningsAgentState)

    graph.add_node("fetch_filing", with_hooks(fetch_filing_node))
    graph.add_node("detect_period", with_hooks(detect_period_node))
    graph.add_node("check_period", with_hooks(check_period_node))
    graph.add_node("load_company_concepts", with_hooks(load_company_concepts_node))
    graph.add_node("agent_document_pipeline", with_hooks(agent_document_pipeline_node))
    graph.add_node("mongodb_save", with_hooks(mongodb_save_node))
    graph.add_node("calculate_q4", with_hooks(calculate_q4_node))

    graph.set_entry_point("fetch_filing")

    for src, dst in [
        ("fetch_filing", "detect_period"),
        ("detect_period", "check_period"),
        ("check_period", "load_company_concepts"),
        ("load_company_concepts", "agent_document_pipeline"),
        ("agent_document_pipeline", "mongodb_save"),
    ]:
        route = _route_after(dst)
        graph.add_conditional_edges(src, route, {dst: dst, "__end__": END})

    # Q4 derivation runs only after a successful save (annual filings only —
    # the node no-ops otherwise).  Never fails the run.
    graph.add_conditional_edges(
        "mongodb_save",
        _route_after_save,
        {"calculate_q4": "calculate_q4", "__end__": END},
    )

    graph.add_edge("calculate_q4", END)

    return graph.compile()
