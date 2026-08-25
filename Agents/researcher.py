# agents/researcher.py
from __future__ import annotations

from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages

from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import END, START, StateGraph
from config import llm
from rag.grader import grade_relevance
from rag.ingest import chunk_and_embed
from state import Evidence, SubQuestion
from langgraph.prebuilt import ToolNode , tools_condition
from tools.search_tools import fetch_url, web_search, wikipedia_search

search_tool = DuckDuckGoSearchRun()

tools = [fetch_url, web_search, wikipedia_search]
llm_with_tools = llm.bind_tools(tools)

MAX_RETRIES = 2

class ResearchState(TypedDict):
    """Per-sub-question state used inside the research subgraph."""

    question: str
    evidence: List[Evidence]
    messages: Annotated[list[BaseMessage], add_messages]
    retries: int

RESEARCH_SYSTEM_PROMPT = """You are the research agent for Veritas. You are given ONE
sub-question. Use the available tools to find evidence that answers it. Prefer
wikipedia_search for stable, encyclopedic facts, and web_search for anything current or
time-sensitive. Call fetch_url if a search result points to a specific promising page. Call
tools as many times as needed, then stop once you have enough to answer confidently."""

def agent_node(state: ResearchState) -> dict:
    """Let the LLM decide which tool(s) to call - or stop - for this sub-question."""
    messages = state["messages"]
    if not messages:
        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
 
 
def extract_evidence_node(state: ResearchState) -> dict:
    """Once the agent stops calling tools, chunk + embed every tool result collected so far."""
    question = state["question"]
    evidence: List[Evidence] = []
 
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            evidence.extend(
                chunk_and_embed(
                    raw_text=str(msg.content),
                    source=msg.name or "tool",
                    url=msg.name or "unknown",
                    query=question,
                )
            )
 
    return {"evidence": evidence}


def grade_evidence_node(state: ResearchState) -> dict:
    """Grade each evidence chunk for relevance to the sub-question; irrelevant chunks are dropped."""
    question = state["question"]
    graded: List[Evidence] = []

    for ev in state["evidence"]:
        score = grade_relevance(question, ev["content"])
        if score == "yes":
            graded.append({**ev, "relevance_score": 1.0})

    return {"evidence": graded}


def route_after_grading(state: ResearchState) -> str:
    """If grading left at least one relevant chunk, stop. Otherwise retry (up to MAX_RETRIES)."""
    if state["evidence"]:
        return "sufficient"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "sufficient"  # out of retries - proceed with whatever we have (possibly empty)
    return "retry"


def prepare_retry_node(state: ResearchState) -> dict:
    """All evidence graded irrelevant - nudge the agent to search differently and loop back."""
    nudge = HumanMessage(
        content=(
            "None of the evidence you found was relevant enough. Try a broader or differently "
            "phrased search query, or try a different tool (e.g. web_search instead of wikipedia_search)."
        )
    )
    return {"messages": [nudge], "retries": state.get("retries", 0) + 1}


research_graph = StateGraph(ResearchState)

research_graph.add_node("agent", agent_node)
research_graph.add_node("tools", ToolNode(tools))
research_graph.add_node("extract_evidence", extract_evidence_node)
research_graph.add_node("grade_evidence", grade_evidence_node)
research_graph.add_node("prepare_retry", prepare_retry_node)

research_graph.add_edge(START, "agent")
research_graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "extract_evidence"})
research_graph.add_edge("tools", "agent")
research_graph.add_edge("extract_evidence", "grade_evidence")
research_graph.add_conditional_edges("grade_evidence", route_after_grading, {"sufficient": END, "retry": "prepare_retry"})
research_graph.add_edge("prepare_retry", "agent")

research_subgraph = research_graph.compile()

def run_research_for_subquestion(payload: dict) -> dict:
    """Send() target - runs the subgraph for one sub-question, packages result for the parent graph."""
    result = research_subgraph.invoke({"question": payload["question"], "evidence": [], "messages": [], "retries": 0})
    sub_question: SubQuestion = {
        "question": payload["question"],
        "evidence": result["evidence"],
        "answer": None,
        "confidence": None,
    }
    return {"research_results": [sub_question]}