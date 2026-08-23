from __future__ import annotations

from typing import List
 
from pydantic import BaseModel, Field
 
from config import llm
from state import SubQuestion

class PlannerOutput(BaseModel):
    """Structured output schema for the planner LLM call (Pydantic - output schema only, not graph state)."""
 
    sub_questions: List[str] = Field(
        description="3 to 5 focused, independently-researchable sub-questions that "
        "together are enough to verify or refute the claim."
    )

 
PLANNER_SYSTEM_PROMPT = """You are the planning agent for Veritas, a fact-verification system.
Given a claim or question, break it down into 3 to 5 specific, independently-researchable
sub-questions. Each sub-question should target one verifiable fact (a date, a number, an event,
an attribution) rather than being a vague restatement of the claim. Do not try to answer the
sub-questions yourself - only produce the questions."""

def planner_node(state : dict) -> dict:
    """
    Break the incoming claim into focused sub-questions.
 
    Reads: state['claim']
    Writes: state['sub_questions'] (SubQuestion dicts, evidence empty, ready for Step 4 research)
    """
    claim = state["claim"]
 
    structured_llm = llm.with_structured_output(PlannerOutput)
    result: PlannerOutput = structured_llm.invoke(
        [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Claim: {claim}"},
        ]
    )
 
    sub_questions: List[SubQuestion] = [
        {"question": q, "evidence": [], "answer": None, "confidence": None}
        for q in result.sub_questions
    ]
 
    return {"sub_questions": sub_questions}

