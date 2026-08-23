from __future__ import annotations

import sqlite3
from typing import List

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import config
from Agents.planner import planner_node
from state import SubQuestion, VeritasState

from langgraph.types import Send
from Agents.researcher import run_research_for_subquestion


def fan_out_to_research(state: VeritasState):
    """Dispatch one research subgraph run per approved sub-question, in parallel."""
    return [Send("research_subgraph", {"question": sq["question"], "evidence": []}) for sq in state["sub_questions"]]


def merge_research_node(state: VeritasState) -> dict:
    """Copy fanned-out research results into sub_questions once all parallel branches complete."""
    return {"sub_questions": state["research_results"]}

def human_review_plan_node(state: VeritasState) -> dict:
    """
    Pause the graph after planning so a human can approve or edit the sub-questions
    before any research (and search/API spend) begins.

    Resume with:
      "approve"   -> keep sub_questions produced by the planner
      list[str]   -> replace sub_questions with this edited list of question strings
    """
    current_questions = [sq["question"] for sq in state["sub_questions"]]

    decision = interrupt(
        {
            "reason": "review_sub_questions",
            "claim": state["claim"],
            "sub_questions": current_questions,
        }
    )

    if isinstance(decision, list):
        edited: List[SubQuestion] = [
            {"question": q, "evidence": [], "answer": None, "confidence": None}
            for q in decision
        ]
        return {"sub_questions": edited}

    return {}


conn = sqlite3.connect(database=config.DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(VeritasState)
graph.add_node("planner", planner_node)
graph.add_node("human_review_plan", human_review_plan_node)
graph.add_node("research_subgraph", run_research_for_subquestion)
graph.add_node("merge_research", merge_research_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "human_review_plan")
graph.add_conditional_edges("human_review_plan", fan_out_to_research, ["research_subgraph"])
graph.add_edge("research_subgraph", "merge_research")
graph.add_edge("merge_research", END)
veritas_graph = graph.compile(checkpointer=checkpointer)

#hlpers
def retrieve_all_threads() -> list[str]:
    """List every thread_id (investigation) that has at least one checkpoint."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def start_investigation(claim: str, thread_id: str) -> dict:
    """Kick off a new investigation for `claim` under a fresh thread_id."""
    run_config = {"configurable": {"thread_id": thread_id}}
    initial_state: VeritasState = {
        "messages": [],
        "claim": claim,
        "sub_questions": [],
        "contradictions": [],
        "verdict": None,
        "confidence_score": None,
        "citations": [],
    }
    return veritas_graph.invoke(initial_state, config=run_config)


def resume_investigation(thread_id: str, decision) -> dict:
    """Resume a paused investigation (e.g. after human review of the sub-question plan)."""
    run_config = {"configurable": {"thread_id": thread_id}}
    return veritas_graph.invoke(Command(resume=decision), config=run_config)