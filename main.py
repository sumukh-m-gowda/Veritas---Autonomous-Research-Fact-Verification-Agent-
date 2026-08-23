from __future__ import annotations

import uuid

from langgraph.types import Command

from graph import veritas_graph


def run_cli():
    """Simple CLI loop to run a Veritas investigation with human-in-the-loop plan review."""
    thread_id = str(uuid.uuid4())
    claim = input("Enter a claim to investigate: ")

    run_config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [],
        "claim": claim,
        "sub_questions": [],
        "contradictions": [],
        "verdict": None,
        "confidence_score": None,
        "citations": [],
    }
    result = veritas_graph.invoke(initial_state, config=run_config)

    interrupts = result.get("__interrupt__", [])
    if interrupts:
        payload = interrupts[0].value
        print(f"\nProposed sub-questions for: {payload['claim']}")
        for i, q in enumerate(payload["sub_questions"], 1):
            print(f"  {i}. {q}")

        decision = input(
            "\nType 'approve' to continue, or type your first edited question "
            "(then keep entering questions, blank line to finish):\n"
        )
        if decision.strip().lower() == "approve":
            resume_value = "approve"
        else:
            lines = [decision]
            while True:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            resume_value = [line.strip() for line in lines if line.strip()]

        result = veritas_graph.invoke(Command(resume=resume_value), config=run_config)

    print("\nFinal sub-questions:")
    for sq in result["sub_questions"]:
        print("-", sq["question"])


if __name__ == "__main__":
    run_cli()