# Veritas — Execution Flow (what actually runs, in order, and why)

This traces ONE real run: user submits a claim via Streamlit, approves the
sub-questions as-is, and research completes. Read top to bottom — every
step names the exact file, function, and what state looks like before/after.

---

## Trigger: user types a claim and clicks "Investigate"

**File:** `streamlit_app.py`
```python
if st.button("Investigate") and claim:
    result = start_investigation(claim, st.session_state["thread_id"])
```
→ calls into `graph.py`.

---

## STEP 1 — `start_investigation(claim, thread_id)` — `graph.py`

Builds a completely empty `VeritasState`:
```python
{
  "messages": [], "claim": "<the claim>", "sub_questions": [],
  "contradictions": [], "verdict": None, "confidence_score": None,
  "citations": [], "research_results": [],
}
```
Calls `veritas_graph.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})`.
This hands control to the compiled graph, which begins at `START`.

---

## STEP 2 — `planner_node(state)` — `Agents/planner.py`

**Reads:** `state["claim"]`
**Does:**
1. Calls `llm.with_structured_output(PlannerOutput)`, invoked with the
   `PLANNER_SYSTEM_PROMPT` + the claim.
2. Gemini returns something matching `PlannerOutput` —
   `{"sub_questions": ["When was...", "Who was...", "What was..."]}`.
3. Converts each string into a full `SubQuestion` dict — `evidence: []`,
   `answer: None`, `confidence: None`.

**Writes:** `{"sub_questions": [SubQuestion, SubQuestion, SubQuestion]}`

**Why this runs first:** research can't start on a vague claim — it needs
concrete, checkable questions to search for.

Graph edge taken next: `planner → human_review_plan` (unconditional).

---

## STEP 3 — `human_review_plan_node(state)` — `graph.py`

**Reads:** `state["claim"]`, `state["sub_questions"]`
**Does:**
```python
current_questions = [sq["question"] for sq in state["sub_questions"]]
decision = interrupt({"reason": "review_sub_questions", "claim": ..., "sub_questions": current_questions})
```
**`interrupt(...)` FREEZES EXECUTION HERE.** The graph does not continue.
`veritas_graph.invoke(...)` from Step 1 returns immediately, right now, back
to `streamlit_app.py`, with the result dict containing a
`"__interrupt__"` key holding this payload.

**Back in `streamlit_app.py`:**
```python
interrupts = result.get("__interrupt__", [])
st.session_state["pending_interrupt"] = interrupts[0].value if interrupts else None
```
The UI renders the sub-question review box. Nothing in the graph is
running right now — it's frozen, waiting, possibly for minutes or hours
(state is safely persisted via `checkpointer`).

**User clicks "Approve as-is":**
```python
result = resume_investigation(st.session_state["thread_id"], "approve")
```
→ `graph.py`'s `resume_investigation`:
```python
veritas_graph.invoke(Command(resume="approve"), config={"configurable": {"thread_id": thread_id}})
```
This loads the checkpoint, and `interrupt(...)` from before now "returns"
the string `"approve"` — `human_review_plan_node` resumes exactly where it
left off:
```python
decision = "approve"  # <- this is what interrupt() just returned
if isinstance(decision, list):   # False, it's a string
    ...
return {}   # no changes - sub_questions stay as the planner made them
```

*(If the user had instead edited the questions and clicked "Submit edited
sub-questions", `decision` would be a `list[str]`, and this function would
rebuild `sub_questions` from that edited list instead — same code path,
different outcome.)*

**Writes:** `{}` (in the approve case) — `sub_questions` unchanged.

Graph edge taken next: a **conditional** edge via `fan_out_to_research`.

---

## STEP 4 — `fan_out_to_research(state)` — `graph.py`

**Reads:** `state["sub_questions"]` (3 items, say)
**Does:**
```python
return [Send("research_subgraph", {"question": sq["question"], "evidence": [], "messages": []}) for sq in state["sub_questions"]]
```
Returns a LIST of 3 `Send` objects — NOT a single next-node name. This
tells LangGraph: "run the node named `research_subgraph`, THREE separate
times, in PARALLEL, each with its own small input dict."

**This is the fan-out point** — from here, three independent execution
threads run simultaneously, each going through Steps 5a–5f below
INDEPENDENTLY, for its own one sub-question.

---

## STEP 5 — Inside EACH parallel branch: `run_research_for_subquestion(payload)` — `Agents/researcher.py`

This function is the node registered under the name `"research_subgraph"`
in the parent graph. `payload` is `{"question": "...", "evidence": [], "messages": []}`
— just ONE sub-question's worth of input, courtesy of `Send`.

```python
result = research_subgraph.invoke({"question": payload["question"], "evidence": [], "messages": [], "retries": 0})
```
This invokes the COMPILED SUBGRAPH (a separate, smaller state machine) —
everything in Steps 5a-5f below happens INSIDE this one `.invoke()` call.

### STEP 5a — `agent_node(state)` — first call

**Reads:** `state["messages"]` (empty — first call)
**Does:** builds `[SystemMessage(RESEARCH_SYSTEM_PROMPT), HumanMessage(question)]`,
calls `llm_with_tools.invoke(messages)`. Gemini decides: "I should search
for this" and returns an `AIMessage` with `.tool_calls` populated, e.g.
`[{"name": "wikipedia_search", "args": {"query": "Eiffel Tower completion date"}}]`.

**Writes:** `{"messages": [that AIMessage]}` — reducer appends it.

**Routing:** `tools_condition` inspects the last message, sees
`.tool_calls` is non-empty, returns `"tools"` → graph goes to the `"tools"`
node.

### STEP 5b — `ToolNode(tools)` — `Agents/researcher.py` (prebuilt LangGraph node)

**Reads:** the last `AIMessage`'s `.tool_calls`.
**Does:** actually calls `wikipedia_search.invoke({"query": "..."})` →
`tools/search_tools.py`'s function runs → either returns real Wikipedia
text, or (if it fails) a graceful error string thanks to the `try/except`.
**Writes:** `{"messages": [ToolMessage(content=<result>, name="wikipedia_search")]}`

**Routing:** unconditional edge back to `"agent"`.

### STEP 5c — `agent_node(state)` — second call

**Reads:** `state["messages"]` — now has `[System, Human, AI(tool_call), Tool]`.
Non-empty, so skips rebuilding, uses history as-is.
**Does:** `llm_with_tools.invoke(messages)`. This time, suppose Gemini
decides it has enough — returns an `AIMessage` with `.content` set and NO
`.tool_calls`.
**Writes:** `{"messages": [that AIMessage]}`.

**Routing:** `tools_condition` sees no `.tool_calls` → returns the special
`END` signal → but the graph's mapping redirects `END` to
`"extract_evidence"` instead of actually ending (see the
`{"tools": "tools", END: "extract_evidence"}` mapping in `graph.py`/
`Agents/researcher.py`'s subgraph wiring).

### STEP 5d — `extract_evidence_node(state)`

**Reads:** the ENTIRE `state["messages"]` list.
**Does:** loops through, finds the ONE `ToolMessage` (from Step 5b), calls
`chunk_and_embed(raw_text=<wikipedia text>, source="wikipedia_search", url="wikipedia_search", query=question)`
→ `rag/ingest.py`: splits into ~500-char chunks, embeds them + the
question, keeps the top-3 most similar chunks.
**Writes:** `{"evidence": [Evidence, Evidence, Evidence]}` (each with
`relevance_score: None` still).

**Routing:** unconditional edge to `"grade_evidence"`.

### STEP 5e — `grade_evidence_node(state)`

**Reads:** `state["question"]`, `state["evidence"]` (3 chunks).
**Does:** for EACH chunk, calls `grade_relevance(question, chunk_content)`
→ `rag/grader.py` → one structured-output LLM call per chunk, returns
`"yes"` or `"no"`. Suppose 2 come back `"yes"`, 1 comes back `"no"`.
**Writes:** `{"evidence": [the 2 "yes" chunks, each stamped relevance_score: 1.0]}`
— the `"no"` chunk is dropped entirely.

**Routing:** `route_after_grading` checks `state["evidence"]` — it's
non-empty (2 chunks survived) → returns `"sufficient"` → maps to `END`.

*(If ALL chunks had graded `"no"`: `route_after_grading` would return
`"retry"` instead → routes to `prepare_retry_node`, which appends a nudge
message and increments `retries`, then loops back to Step 5a's
`agent_node` to search again — up to `MAX_RETRIES` times before giving up
and proceeding with whatever it has, even if empty.)*

### STEP 5f — subgraph reaches `END`

`research_subgraph.invoke(...)` (called back in Step 5's opening) returns
`{"question": "...", "evidence": [2 graded Evidence dicts], "messages": [...], "retries": 0}`.

**Back in `run_research_for_subquestion`:**
```python
sub_question: SubQuestion = {
    "question": payload["question"],
    "evidence": result["evidence"],
    "answer": None,
    "confidence": None,
}
return {"research_results": [sub_question]}
```
This ONE branch's final output. Because `research_results` is
`Annotated[..., operator.add]` on the PARENT `VeritasState`, this gets
CONCATENATED with the other 2 parallel branches' outputs, not overwritten.

*(All three parallel branches — one per sub-question — run through Steps
5a-5f independently and simultaneously. LangGraph waits for ALL of them to
finish before continuing.)*

---

## STEP 6 — `merge_research_node(state)` — `graph.py`

Runs once all 3 parallel branches are done.
**Reads:** `state["research_results"]` — now a list of 3 fully-researched,
graded `SubQuestion` dicts (accumulated via the reducer across the 3
branches).
**Writes:** `{"sub_questions": state["research_results"]}` — copies the
accumulated results into the field every future node will actually read.

Graph edge taken next: `merge_research → END`.

---

## STEP 7 — Graph run complete

`veritas_graph.invoke(Command(resume="approve"), ...)` (from Step 3's
resume call) FINALLY returns, all the way back to `streamlit_app.py`:
```python
result = resume_investigation(...)
st.session_state["pending_interrupt"] = None
st.session_state["result"] = result
```
The UI renders:
```python
st.subheader("Sub-questions")
for sq in st.session_state["result"].get("sub_questions", []):
    st.write("•", sq["question"])
```
— showing each sub-question. (Evidence itself isn't displayed in the UI
yet — it's in `result["sub_questions"][i]["evidence"]` if you want to add
that.)

---

## One-paragraph summary of the whole path

```
User submits claim
  → planner_node (LLM: claim → sub-questions)
  → human_review_plan_node (interrupt/pause for human approval)
  → fan_out_to_research (Send × N, parallel)
      → [per sub-question, in parallel:]
         agent_node ⇄ ToolNode  (LLM picks & runs search tools, loops until satisfied)
         → extract_evidence_node (chunk + embed tool results, keep top-k similar)
         → grade_evidence_node (LLM grades each chunk yes/no relevant)
         → (if all "no" and retries left: prepare_retry_node → back to agent_node)
         → END (subgraph done for this sub-question)
  → merge_research_node (combine all parallel results back into sub_questions)
  → END (whole investigation done)
```

Everything after this point (answering each sub-question from its graded
evidence, cross-checking for contradictions, producing a final verdict with
citations) is NOT built yet — that's Steps 7+ in the roadmap.
