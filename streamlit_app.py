import uuid

import streamlit as st

from graph import resume_investigation, retrieve_all_threads, start_investigation

# **************************************** utility functions *************************


def generate_thread_id():
    """Create a fresh thread_id for a new investigation."""
    return str(uuid.uuid4())


def reset_investigation():
    """Start a blank investigation session."""
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["pending_interrupt"] = None
    st.session_state["result"] = None


# **************************************** Session Setup ******************************

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "past_threads" not in st.session_state:
    st.session_state["past_threads"] = retrieve_all_threads()

if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

if "result" not in st.session_state:
    st.session_state["result"] = None


# **************************************** Sidebar UI *********************************

st.sidebar.title("Veritas")

if st.sidebar.button("New Investigation"):
    reset_investigation()

st.sidebar.header("Past Investigations")
for thread_id in st.session_state["past_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        st.session_state["result"] = None
        st.session_state["pending_interrupt"] = None


# **************************************** Main UI ************************************

st.title("Veritas — Fact Verification Agent")

claim = st.text_input("Enter a claim or question to investigate")

if st.button("Investigate") and claim:
    result = start_investigation(claim, st.session_state["thread_id"])
    if st.session_state["thread_id"] not in st.session_state["past_threads"]:
        st.session_state["past_threads"].append(st.session_state["thread_id"])

    interrupts = result.get("__interrupt__", [])
    st.session_state["pending_interrupt"] = interrupts[0].value if interrupts else None
    st.session_state["result"] = result

# ---- Human review of sub-questions (HITL) ----
if st.session_state["pending_interrupt"]:
    payload = st.session_state["pending_interrupt"]
    st.subheader("Review sub-questions before research begins")
    st.caption(f"Claim: {payload['claim']}")

    edited_text = st.text_area(
        "One sub-question per line — edit, remove, or add as needed",
        value="\n".join(payload["sub_questions"]),
        height=150,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve as-is"):
            result = resume_investigation(st.session_state["thread_id"], "approve")
            st.session_state["pending_interrupt"] = None
            st.session_state["result"] = result
    with col2:
        if st.button("Submit edited sub-questions"):
            edited_list = [q.strip() for q in edited_text.split("\n") if q.strip()]
            result = resume_investigation(st.session_state["thread_id"], edited_list)
            st.session_state["pending_interrupt"] = None
            st.session_state["result"] = result

# ---- Show final state ----
if st.session_state["result"] and not st.session_state["pending_interrupt"]:
    st.subheader("Sub-questions")
    for sq in st.session_state["result"].get("sub_questions", []):
        st.write("•", sq["question"])