# agents/researcher.py
from __future__ import annotations

from typing import List, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import END, START, StateGraph

from rag.ingest import chunk_and_embed
from state import Evidence, SubQuestion

search_tool = DuckDuckGoSearchRun()


class ResearchState(TypedDict):
    """Per-sub-question state used inside the research subgraph."""

    question: str
    evidence: List[Evidence]


def research_node(state: ResearchState) -> dict:
    """Search the web for state['question'], chunk + embed the result, keep top-k evidence."""
    question = state["question"]
    raw_text = search_tool.invoke(question)
    evidence = chunk_and_embed(raw_text, source="web_search", url="duckduckgo", query=question)
    return {"evidence": evidence}


research_graph = StateGraph(ResearchState)
research_graph.add_node("research", research_node)
research_graph.add_edge(START, "research")
research_graph.add_edge("research", END)
research_subgraph = research_graph.compile()


def run_research_for_subquestion(payload: ResearchState) -> dict:
    """Send() target - runs the subgraph for one sub-question, packages result for the parent graph."""
    result = research_subgraph.invoke(payload)
    sub_question: SubQuestion = {
        "question": payload["question"],
        "evidence": result["evidence"],
        "answer": None,
        "confidence": None,
    }
    return {"research_results": [sub_question]}