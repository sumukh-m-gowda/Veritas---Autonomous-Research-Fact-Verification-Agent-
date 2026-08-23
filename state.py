from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class Evidence(TypedDict):
    """A single piece of retrieved evidence for one sub-question."""

    source: str  # e.g. "web_search", "wikipedia", "faiss"
    url: str
    content: str
    relevance_score: Optional[float]  # filled in by the CRAG grader (Step 6)


class SubQuestion(TypedDict):
    """One sub-question the planner breaks the claim into, plus its own research trail."""

    question: str
    evidence: List[Evidence]
    answer: Optional[str]
    confidence: Optional[float]


class VeritasState(TypedDict):
    """
    Shared state that flows through the whole Veritas graph.

    messages       - raw LLM/tool-call history (needed once we add tool calling in Step 5)
    claim          - the original claim or question the user submitted
    sub_questions  - planner output; each one accumulates its own evidence + answer
    contradictions - filled in by the verifier (Step 7) when sub-answers disagree
    verdict        - final synthesizer output, e.g. "True" / "False" / "Unverified"
    confidence_score - synthesizer's confidence in the verdict, 0-1
    citations      - flat list of source URLs backing the final verdict
    """

    messages: Annotated[list[BaseMessage], add_messages]
    claim: str
    sub_questions: List[SubQuestion]
    contradictions: List[str]
    verdict: Optional[str]
    confidence_score: Optional[float]
    citations: List[str]
    research_results: Annotated[List[SubQuestion], operator.add]  