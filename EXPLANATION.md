# Veritas — Complete Explanation & Revision Guide

You're about to step away for a while. This file exists so that when you
come back, you don't have to re-derive anything — just re-read this,
top to bottom, and it should all click back into place.

Read Part 1 even if you think you remember LangGraph — it's the minimum
vocabulary every later section assumes you have fresh.

---

# PART 1 — Fundamentals (re-read this first, every time)

## 1.1 What problem is Veritas actually solving?

Given a claim like *"The Eiffel Tower was completed in 1889 as the entrance
to the World's Fair"*, a plain LLM will just answer from memory — no way to
check if it's right, no sources, no way to know if it's confidently wrong.
Veritas instead: breaks the claim into checkable parts, actually searches
for evidence on each part, checks whether that evidence is any good, checks
whether different sources agree, and only then commits to a verdict with
citations attached. Every file in this project exists to serve ONE of those
sub-steps.

## 1.2 What is LangGraph, in one paragraph?

LangGraph lets you build a program as a **graph of nodes**, where each node
is a plain Python function `(state) -> partial_state_update`, and the edges
between nodes decide execution order. Instead of one big function calling
other functions directly, nodes only ever communicate by reading/writing a
shared **state** object. This makes complex agent logic (loops, retries,
parallel branches, pausing for human input) something you can draw as a
diagram instead of tangled function calls.

## 1.3 The five LangGraph concepts every file in this project uses

**(a) State (`TypedDict`)** — a typed dictionary describing what data
exists at any point in the graph. `VeritasState` (whole pipeline) and
`ResearchState` (just the research subgraph) are the two you have.

**(b) Node** — a plain function taking `state` and returning a dict of the
fields it wants to update. It does NOT return the whole state — just what
changed. Example: `planner_node` only returns `{"sub_questions": [...]}`,
nothing else, even though `state` has 8 fields total.

**(c) Reducers (`Annotated[type, reducer_function]`)** — normally, when a
node returns a value for a field, LangGraph OVERWRITES the old value. If a
field is declared `Annotated[list[X], some_reducer]`, LangGraph instead
calls `some_reducer(old_value, new_value)` to COMBINE them. You use this
for exactly two things in this project:
  - `messages: Annotated[list[BaseMessage], add_messages]` — appends new
    messages onto conversation history instead of replacing it.
  - `research_results: Annotated[List[SubQuestion], operator.add]` — list
    concatenation, needed because parallel branches (see next point) all
    write to this field at once and need to be combined, not overwrite
    each other.

**(d) Fan-out with `Send`** — normally one node leads to one next node. When
you need to run the SAME node N times in parallel with N different inputs
(e.g. "research subgraph, once per sub-question"), a node instead returns a
LIST of `Send(node_name, custom_input_dict)` objects. LangGraph runs all of
them concurrently and waits for all to finish before continuing.

**(e) `interrupt()` / `Command(resume=...)`** — human-in-the-loop. Calling
`interrupt(payload)` inside a node FREEZES the whole graph right there and
hands `payload` back to whoever called `.invoke(...)`. Nothing else runs
until someone later calls `.invoke(Command(resume=some_value), config=...)`
on the SAME `thread_id` — at which point `interrupt(...)` "returns"
`some_value`, as if it had just been waiting the whole time. This requires
a **checkpointer** (we use `SqliteSaver`) to persist where execution paused.

## 1.4 Two other patterns used throughout

**Structured output** (`llm.with_structured_output(SomeModel)`) — forces
the LLM to return data matching a Pydantic schema instead of free text.
Used by the planner (to get a clean list of sub-questions) and the CRAG
grader (to get a clean `"yes"`/`"no"`).

**Tool calling** (`llm.bind_tools([...])`, `ToolNode`, `tools_condition`)
— gives the LLM a menu of Python functions (decorated with `@tool`) it can
request to call. The LLM doesn't execute them — it just asks. `ToolNode`
actually runs the requested function(s). `tools_condition` checks "did the
LLM ask for a tool, or is it done?" and routes accordingly. This is what
makes the research agent adaptive instead of hardcoded to one search call.

## 1.5 Two kinds of Python typed structures used — don't confuse them

- **`TypedDict`** (in `state.py`) — describes GRAPH STATE. Plain dict at
  runtime, type hints are just for your editor/type-checker.
- **`BaseModel` (Pydantic)** — describes ONE LLM CALL's forced output shape
  (`PlannerOutput`, `RelevanceGrade`). Pydantic actually VALIDATES data at
  runtime (raises errors on bad data); TypedDict does not.

---

# PART 2 — Every file, every class, every function

Read these in this exact order — later files depend on earlier ones.

## 2.1 `config.py` — shared LLM/embeddings client, project-wide

| What | Purpose |
|---|---|
| `REQUIRED_ENV_VARS` check | Crashes immediately and clearly if `GEMINI_API_KEY` is missing, instead of failing confusingly deep inside some agent later. |
| `llm` | The ONE `ChatGoogleGenerativeAI` (`gemini-2.5-flash`, `temperature=0`) instance every agent imports. `temperature=0` = deterministic, no creative randomness — important for a fact-checker. |
| `embeddings` | The ONE `GoogleGenerativeAIEmbeddings` instance. Turns text into vectors for similarity search (used in `rag/ingest.py`). Different model from `llm` — one predicts text, the other measures text similarity. |
| `DATA_DIR`, `DB_PATH`, `FAISS_DIR` | Shared file paths so SQLite and any vector stores always agree on location. |

## 2.2 `state.py` — the shape of data flowing through the whole graph

**`Evidence(TypedDict)`** — one retrieved chunk: `source`, `url`, `content`,
`relevance_score` (filled by the CRAG grader, `None` until then).

**`SubQuestion(TypedDict)`** — one planner question plus everything
discovered about it: `question`, `evidence: List[Evidence]`, `answer`
(not yet used — reserved for a future sub-answering step), `confidence`
(same). Nesting `Evidence` inside `SubQuestion` is what lets each
sub-question be researched and graded independently.

**`VeritasState(TypedDict)`** — the full pipeline state:
- `messages` — `Annotated[..., add_messages]`. Conversation-style history
  for the top-level graph (not heavily used yet at this stage; the
  research subgraph has its OWN separate `messages` field for its
  tool-calling loop).
- `claim` — the original input.
- `sub_questions` — the working list every downstream node reads/writes.
- `contradictions` — reserved for a future verifier step.
- `verdict`, `confidence_score`, `citations` — reserved for a future
  synthesizer step. All `None`/empty right now — that's expected, not
  a bug.
- `research_results` — `Annotated[..., operator.add]`. Scratch field ONLY
  used during the parallel research fan-out; `merge_research_node` copies
  it into `sub_questions` right after.

## 2.3 `Agents/planner.py` — turns a claim into sub-questions

**`PlannerOutput(BaseModel)`** — Pydantic schema forcing the LLM's output
into `{"sub_questions": ["...", "...", ...]}`.

**`PLANNER_SYSTEM_PROMPT`** — instructs the LLM to produce SPECIFIC,
checkable sub-questions (dates, numbers, attributions), not vague
restatements of the claim.

**`planner_node(state) -> dict`**
1. Reads `state["claim"]`.
2. Calls `llm.with_structured_output(PlannerOutput)`, invoked with a
   system+user message pair (plain dicts, not `BaseMessage` objects — fine
   for a single one-shot call).
3. Converts each returned question string into a full `SubQuestion` dict
   (`evidence: []`, `answer: None`, `confidence: None` — those get filled
   by later nodes, not this one).
4. Returns `{"sub_questions": [...]}` — a partial update, nothing else in
   state gets touched.

## 2.4 `tools/search_tools.py` — the tools the research agent can call

`_ddg = DuckDuckGoSearchRun()` and `_wiki = WikipediaQueryRun(...)` are
created once at import time — the underlying search clients. Underscore
prefix = "internal, don't use these directly, use the `@tool`-wrapped
versions below."

**`web_search(query: str) -> str`** — `@tool`-decorated. Docstring is not
just documentation — the LLM reads it to decide WHEN to pick this tool vs.
the others. Wrapped in `try/except`: any failure (network error, API
hiccup) returns a descriptive error STRING instead of crashing, so the
agent can see the failure and try something else instead of the whole
investigation dying.

**`wikipedia_search(query: str) -> str`** — same pattern, for stable
encyclopedic facts. Also wrapped in `try/except` — this is the one that
actually broke once during testing (Wikipedia's API returned malformed
JSON), which is exactly why the error handling exists.

**`fetch_url(url: str) -> str`** — fetches a specific page's text (e.g. one
found via search), capped at 5000 characters, same `try/except` pattern.

## 2.5 `rag/ingest.py` — chunking + embedding + similarity filtering

`splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
— created once. 500-char chunks so embeddings represent focused passages,
not one giant diluted blob. 50-char overlap so a sentence at a chunk
boundary doesn't get cut in half and lose meaning on both sides.

**`chunk_and_embed(raw_text, source, url, query, k=3) -> list[dict]`**
1. Splits `raw_text` into `Document` chunks (each carrying source/url as
   metadata).
2. If nothing came out (empty text), returns `[]` immediately — defensive,
   handles the case where a tool failed and returned an empty/error string.
3. Embeds ALL chunks into a **throwaway, in-memory** FAISS index (this is
   NOT persistent long-term memory — just a temporary ranking tool for this
   one call).
4. `similarity_search(query, k=k)` — embeds the sub-question too, finds the
   `k` closest chunks by vector distance. This is what filters "everything
   the tool returned" down to "only what's actually relevant to THIS
   sub-question."
5. Reshapes results into `Evidence`-shaped dicts, `relevance_score: None`
   (filled later by the CRAG grader).

## 2.6 `rag/grader.py` — CRAG: is this evidence actually relevant?

**`RelevanceGrade(BaseModel)`** — Pydantic schema forcing a
`binary_score: Literal["yes", "no"]`.

**`GRADER_SYSTEM_PROMPT`** — instructs the LLM to be STRICT: grade "no" if
a chunk is off-topic, vague, or just keyword-matches without real
substance.

**`grade_relevance(question, evidence_content) -> str`**
- One structured-output call per evidence chunk. Returns `"yes"` or
  `"no"`. Called once for EVERY chunk in `grade_evidence_node` (see
  `Agents/researcher.py` below) — meaning grading cost scales with how much
  evidence was retrieved.

## 2.7 `Agents/researcher.py` — the biggest file: tool-calling + CRAG loop

This file defines an entire **subgraph** — a self-contained mini state
machine that runs once per sub-question (via `Send`, wired in `graph.py`).

```python
tools = [fetch_url, web_search, wikipedia_search]
llm_with_tools = llm.bind_tools(tools)
MAX_RETRIES = 2
```
`llm_with_tools` is a NEW llm-like object aware of these 3 tools' names,
docstrings, and parameter schemas — it can REQUEST them, not call them
directly. `MAX_RETRIES` caps the corrective search loop so a stubborn
sub-question can't loop forever.

**`ResearchState(TypedDict)`** — this subgraph's own, smaller state:
`question`, `evidence`, `messages` (its own tool-calling conversation,
separate from the parent's `messages`), `retries` (plain int, no reducer
needed — only one branch runs at a time inside this subgraph, no parallel
writes to worry about).

**`agent_node(state) -> dict`** — the "brain" of the loop. First call:
`state["messages"]` is empty, so it builds `[SystemMessage(...),
HumanMessage(question)]`. Every call after that: history already exists
(thanks to the `add_messages` reducer), so it's used as-is. Either way,
`llm_with_tools.invoke(messages)` is called, and the single new response
message is returned (the reducer appends it — you never manage the full
list by hand).

**`extract_evidence_node(state) -> dict`** — runs once the LLM stops
requesting tools. Walks the ENTIRE accumulated `messages` list, pulls out
just the `ToolMessage` instances (actual tool RESULTS — ignores
system/human/AI messages), and chunk+embeds each one via
`rag.ingest.chunk_and_embed`. **Important nuance:** this rebuilds `evidence`
from ALL `ToolMessage`s in history every time it runs — including ones from
a PRIOR retry attempt. Functionally correct (old irrelevant chunks just get
re-graded "no" again), just means retries cost a few extra grading calls.

**`grade_evidence_node(state) -> dict`** — NEW in Step 6. Calls
`grade_relevance(question, chunk_content)` on every chunk in `evidence`,
keeps only the ones graded `"yes"` (stamping `relevance_score: 1.0` on
survivors), drops the rest.

**`route_after_grading(state) -> str`** — conditional edge function.
Returns `"sufficient"` if ANY evidence survived grading, OR if
`retries >= MAX_RETRIES` (give up gracefully, proceed with whatever we
have — even if empty). Otherwise returns `"retry"`.

**`prepare_retry_node(state) -> dict`** — only runs when ALL evidence was
graded irrelevant and retries remain. Appends a `HumanMessage` nudging the
agent to search differently (broader query, or a different tool), and
increments `retries`. Returns straight back to `agent_node`.

**The subgraph wiring:**
```python
research_graph.add_edge(START, "agent")
research_graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "extract_evidence"})
research_graph.add_edge("tools", "agent")
research_graph.add_edge("extract_evidence", "grade_evidence")
research_graph.add_conditional_edges("grade_evidence", route_after_grading, {"sufficient": END, "retry": "prepare_retry"})
research_graph.add_edge("prepare_retry", "agent")
```
Drawn as a path: `agent ⇄ tools` (tool-calling loop) `→ extract_evidence →
grade_evidence` → either done, or `prepare_retry → agent` again (corrective
loop). `tools_condition` and `ToolNode` are PREBUILT LangGraph functions —
you didn't write their internal logic, just wired them in.

**`run_research_for_subquestion(payload) -> dict`** — the bridge function,
this is what's ACTUALLY registered as a node in the PARENT graph
(`graph.py`). Takes the `Send` payload, invokes the whole subgraph fresh
(`retries: 0` to start), repackages the result as a full `SubQuestion`, and
returns `{"research_results": [sub_question]}` — landing in the parent's
`operator.add`-reducer field so parallel branches combine correctly.

## 2.8 `graph.py` — assembles everything above into one runnable graph

**`fan_out_to_research(state) -> list[Send]`** — for every sub-question,
builds a `Send("research_subgraph", {question, evidence: [], messages: []})`.
Returning a LIST of `Send`s (not a single next-node string) is what makes
LangGraph run all sub-questions' research in PARALLEL.

**`merge_research_node(state) -> dict`** — runs once ALL parallel branches
finish (LangGraph waits automatically). Copies the now-complete
`research_results` list into `sub_questions` — the field name every
downstream node actually reads.

**`human_review_plan_node(state) -> dict`** — the HITL checkpoint. Calls
`interrupt({...})` with the claim + proposed questions. If resumed with a
list of strings, rebuilds `sub_questions` from that edited list; if resumed
with `"approve"` (or anything else non-list), returns `{}` (no change).

**`conn` / `checkpointer`** — a raw `sqlite3` connection
(`check_same_thread=False`, required because Streamlit/LangGraph may touch
it from different threads) wrapped in `SqliteSaver`. This is what makes
`interrupt()`/resume possible — checkpoints are how the graph remembers
where it paused.

**Graph assembly:**
```python
graph.add_edge(START, "planner")
graph.add_edge("planner", "human_review_plan")
graph.add_conditional_edges("human_review_plan", fan_out_to_research, ["research_subgraph"])
graph.add_edge("research_subgraph", "merge_research")
graph.add_edge("merge_research", END)
veritas_graph = graph.compile(checkpointer=checkpointer)
```
Note: the node NAME `"research_subgraph"` (a string) is bound to the
function `run_research_for_subquestion` — the actual compiled subgraph
object (also confusingly named `research_subgraph` in
`Agents/researcher.py`) is invoked INSIDE that function, not wired in here
directly as a node object. Two similarly-named but different things.

**`retrieve_all_threads() -> list[str]`** — scans every saved checkpoint,
collects unique `thread_id`s. Powers the "Past Investigations" sidebar.

**`start_investigation(claim, thread_id) -> dict`** — builds a fully empty
`VeritasState` (all 8 fields at their "nothing yet" value, including
`research_results: []`) and starts a fresh graph run.

**`resume_investigation(thread_id, decision) -> dict`** — the only way to
un-pause a graph that hit `interrupt()`. `Command(resume=decision)` tells
LangGraph to load the checkpoint for `thread_id` and feed `decision` back
into the waiting `interrupt(...)` call.

## 2.9 `main.py` — CLI entrypoint (no Streamlit needed)

Builds its OWN `initial_state` dict by hand (doesn't call
`start_investigation()` — a known duplication between this file and
`graph.py`; keep them in sync manually if you add new state fields later).
Runs the graph, checks for `result.get("__interrupt__", [])`, prints the
proposed sub-questions, reads either `"approve"` or a list of edited
questions from `input()`, resumes via `Command(resume=...)`, prints the
final sub-questions.

## 2.10 `streamlit_app.py` — the UI

Only ever imports from `graph.py` (`start_investigation`,
`resume_investigation`, `retrieve_all_threads`) — never touches `Agents/`
directly. Uses `st.session_state` (Streamlit's way of persisting variables
across reruns — Streamlit re-executes the ENTIRE script on every click)
to track `thread_id`, `past_threads`, `pending_interrupt`, `result`. The
`if st.session_state["pending_interrupt"]:` block is the UI counterpart to
`human_review_plan_node`'s `interrupt()` — renders an editable text box +
"Approve as-is" / "Submit edited sub-questions" buttons.

---

# PART 3 — What's genuinely NOT built yet (don't panic if you see `None`s)

- `SubQuestion["answer"]`/`["confidence"]` — always `None`. No node
  answers a sub-question from its graded evidence yet.
- `VeritasState["contradictions"]`/`["verdict"]`/`["confidence_score"]`/
  `["citations"]` — always empty/`None`. No verifier or synthesizer node
  exists yet.
- `Evidence["url"]` is really "which tool produced this," not a real
  clickable link — worth revisiting once citations matter.
- No SQLite table for clean, queryable investigation results (only the
  checkpointer, which is for resuming, not querying).
- No short-term/long-term memory, no tracing, no MCP server, no eval
  harness.

These are all future steps, not things currently broken.
