from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from config import llm


class RelevanceGrade(BaseModel):
    """Structured grade for whether one evidence chunk helps answer a sub-question."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the evidence chunk is relevant to answering the sub-question, else 'no'."
    )


GRADER_SYSTEM_PROMPT = """You are a strict relevance grader for Veritas, a fact-verification
system. Given a sub-question and one retrieved evidence chunk, decide if the chunk contains
information that helps answer the sub-question. Grade 'no' if the chunk is off-topic, too vague,
or only mentions keywords without real substance."""


def grade_relevance(question: str, evidence_content: str) -> str:
    """Grade one evidence chunk as 'yes' or 'no' relevant to the sub-question. Returns the score."""
    structured_llm = llm.with_structured_output(RelevanceGrade)
    result: RelevanceGrade = structured_llm.invoke(
        [
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Sub-question: {question}\n\nEvidence chunk:\n{evidence_content}"},
        ]
    )
    return result.binary_score