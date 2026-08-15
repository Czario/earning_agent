"""LangGraph workflow — agent pipeline.

Graph:
    fetch_filing → detect_period → check_period → load_company_concepts
        → agent_document_pipeline → mongodb_save → END

The reporting period is decided ONLY by the period agent (detect_period); if it
fails the run fails — no deterministic period inference anywhere.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from earnings_agents.agent.period import detect_period_node
from earnings_agents.agent.pipeline import agent_document_pipeline_node
from earnings_agents.nodes.check import check_period_node
from earnings_agents.nodes.concepts import load_company_concepts_node
from earnings_agents.nodes.fetch import fetch_filing_node
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


def build_graph():
    """Compile and return the LangGraph earnings scraping workflow."""
    graph = StateGraph(EarningsAgentState)

    graph.add_node("fetch_filing", with_hooks(fetch_filing_node))
    graph.add_node("detect_period", with_hooks(detect_period_node))
    graph.add_node("check_period", with_hooks(check_period_node))
    graph.add_node("load_company_concepts", with_hooks(load_company_concepts_node))
    graph.add_node("agent_document_pipeline", with_hooks(agent_document_pipeline_node))
    graph.add_node("mongodb_save", with_hooks(mongodb_save_node))

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

    graph.add_edge("mongodb_save", END)

    return graph.compile()
