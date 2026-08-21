"""
app.py
------
ClinicalIQ | Apollo Health Clinic -- Streamlit Frontend + Human-in-the-Loop.

Ported from WealthDesk's s13 app.py onto ClinicalIQ's supervisor graph
(clinicaliq/agent.py): classify -> {documents_agent | services_agent |
escalate | decline} -> compliance_agent -> END. Also ports WealthDesk's s13
token-streaming hook (nodes.py's _stream_callback) for a typewriter effect.

Run from inside s01/:
    streamlit run app.py
"""
import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from clinicaliq.agent import build_graph  # noqa: E402
from clinicaliq.config import SAFE_COMPLIANCE_RESPONSE  # noqa: E402
import clinicaliq.nodes as _nodes  # noqa: E402


# ---------------------------------------------------------------------------
# Token streaming -- bridges llm.stream() in nodes.py to the Streamlit UI
# ---------------------------------------------------------------------------

class _StreamingState:
    """Accumulates tokens pushed by nodes.py via _nodes._stream_callback and
    renders them into a Streamlit placeholder, producing a typewriter effect.

    app.py sets _nodes._stream_callback to an instance of this class before
    graph.invoke(); the respond node calls it once per token from llm.stream().
    """

    def __init__(self, placeholder, token_delay: float = 0.0) -> None:
        self._placeholder = placeholder
        self._text = ""
        self._delay = token_delay  # optional per-token sleep (seconds); 0 = full speed

    def __call__(self, token: str) -> None:
        self._text += token
        # Overwrite the same placeholder each token -- avoids flicker from adding
        # a new element per call. The "▌" cursor signals more text is coming.
        self._placeholder.markdown(self._text + "▌")
        if self._delay > 0:
            time.sleep(self._delay)

    @property
    def text(self) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Pure helper functions (no Streamlit calls -- unit tested in s01/tests/)
# ---------------------------------------------------------------------------

def build_input_state(message: str) -> dict:
    """The dict graph.invoke() expects as its first argument for one turn."""
    return {
        "customer_message":  message,  # the patient's input for this turn
        "response":          "",       # filled by the specialist agent
        "specialist":        "",       # which agent handled the query
        "retrieved_docs":    [],       # RAG chunks -- reset each turn to avoid leakage
        "compliance_status": "",       # PASS, REVISED, or FAIL:... -- set by the compliance agent
    }


def get_thread_config(thread_id: str) -> dict:
    """LangGraph needs a thread ID to keep memory across turns in the same session."""
    return {"configurable": {"thread_id": thread_id}}


def compliance_badge(status: str) -> str:
    if status == "PASS":
        return "✅ Compliant"
    if status == "REVISED":
        return "⚠️ Revised"
    if status.startswith("FAIL"):
        return "❌ Violation"
    return ""  # escalated / declined routes have no compliance status


def needs_human_review(result: dict) -> bool:
    """True when the Compliance Agent revised the response -- the human operator
    must approve before it is shown to the patient."""
    return result.get("compliance_status", "") == "REVISED"


def format_route_label(result: dict) -> str:
    sp    = result.get("specialist", "—")    # e.g. "documents_agent", "services_agent"
    cs    = result.get("compliance_status", "")
    badge = compliance_badge(cs)
    label = f"Route: {sp}"
    if badge:
        label += f" | {badge}"
    return label


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        st.session_state.graph     = build_graph(checkpointer=MemorySaver())
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages  = []
        st.session_state.routes    = []


def _sidebar() -> None:
    with st.sidebar:
        st.header("🏥 ClinicalIQ")
        st.caption("Apollo Health Clinic Patient Assistant")
        st.divider()
        if st.button("🔄 New Conversation", use_container_width=True):
            for key in ["graph", "thread_id", "messages", "routes", "pending_hitl"]:
                st.session_state.pop(key, None)
            st.rerun()
        if "thread_id" in st.session_state:
            st.caption(f"Session: {st.session_state.thread_id[:8]}…")
        st.divider()
        st.subheader("Agents")
        st.markdown(
            "- **Supervisor** — classifies query\n"
            "- **Documents Agent** — handles policy & procedure questions\n"
            "- **Services Agent** — handles doctor & service queries (live data)\n"
            "- **Compliance Agent** — patient-safety rules check\n"
            "- **Human-in-the-Loop** — reviews revisions"
        )

        st.divider()
        # Slows down streaming so the typewriter cursor is visible during demos.
        # 0 = full speed (production default). Stored in session_state so the
        # slider position persists across re-runs instead of resetting to 0.
        st.subheader("Demo settings")
        st.session_state["token_delay"] = st.slider(
            "Token delay (ms)",
            min_value=0, max_value=100, value=st.session_state.get("token_delay", 0),
            step=5,
            help="Slow down token streaming so the cursor effect is visible. 0 = full speed.",
        )

        st.divider()
        # INSTRUCTOR TOOL: Force HITL Demo -- the production LLM rarely emits
        # a banned phrase naturally, so this injects a canned compliance-revised
        # response directly into session_state (bypassing the LLM) so the full
        # HITL approval flow can be demonstrated reliably in class.
        st.subheader("Instructor tools")
        if st.button("⚠️ Force HITL Demo", use_container_width=True,
                     help="Injects a canned compliance-revised response to demonstrate the HITL approval form."):
            st.session_state.messages.append({
                "role": "user",
                "content": "What medicine should I take for my fever?",
            })
            st.session_state.pending_hitl = {
                "response": (
                    "Please speak with our nurse about medication options -- she can "
                    "review your symptoms and recommend what's appropriate.\n\n"
                    "If your fever is very high, persistent, or comes with difficulty "
                    "breathing or confusion, please call 112 or go to the nearest "
                    "emergency room.\n\n"
                    "ClinicalIQ | Apollo Health Clinic"
                ),
                "route_label": "Route: services_agent | ⚠️ Revised",
            }
            st.rerun()


def _render_history() -> None:
    messages = st.session_state.get("messages", [])
    routes   = st.session_state.get("routes",   [])
    assistant_idx = 0
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if assistant_idx < len(routes):
                st.caption(routes[assistant_idx])
            assistant_idx += 1


def _handle_hitl() -> bool:
    if "pending_hitl" not in st.session_state:
        return False
    pending = st.session_state.pending_hitl
    st.warning(
        "⚠️ **Compliance Review Required** — The Compliance Agent revised this response. "
        "Please review and approve before sending to the patient."
    )
    with st.form("hitl_approval"):
        edited    = st.text_area("Review and edit the response if needed:", value=pending["response"], height=220)
        col1, col2 = st.columns(2)
        approved  = col1.form_submit_button("✅ Approve & Send", use_container_width=True)
        discarded = col2.form_submit_button("❌ Discard",         use_container_width=True)
    if approved:
        st.session_state.messages.append({"role": "assistant", "content": edited})
        st.session_state.routes.append(pending["route_label"])
        del st.session_state.pending_hitl
        st.rerun()
    elif discarded:
        # Discard: the revised text never reaches the patient. Show the neutral
        # SAFE_COMPLIANCE_RESPONSE fallback instead of leaving the chat blank.
        st.session_state.messages.append({
            "role": "assistant",
            "content": SAFE_COMPLIANCE_RESPONSE,
        })
        st.session_state.routes.append(pending["route_label"])
        del st.session_state.pending_hitl
        st.rerun()
    return True


def main() -> None:
    st.set_page_config(page_title="ClinicalIQ | Apollo Health Clinic", page_icon="🏥", layout="wide")
    st.title("🏥 ClinicalIQ | Apollo Health Clinic")
    st.caption("AI-powered patient guidance assistant — Streamlit UI + Human-in-the-Loop")

    _init_session()
    _sidebar()
    _render_history()

    hitl_active = _handle_hitl()

    if not hitl_active:
        prompt = st.chat_input("Ask about appointments, departments, or test preparation…")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Open the assistant bubble before the response is ready so the
            # streaming placeholder has somewhere to render into, token by token.
            with st.chat_message("assistant"):
                placeholder = st.empty()

            delay_ms = st.session_state.get("token_delay", 0)
            streamer = _StreamingState(placeholder, token_delay=delay_ms / 1000)
            _nodes._stream_callback = streamer
            try:
                result = st.session_state.graph.invoke(
                    build_input_state(prompt),
                    config=get_thread_config(st.session_state.thread_id),
                )
            finally:
                # Always clear the callback -- even on exception -- so the next
                # invocation doesn't accidentally use a stale streamer.
                _nodes._stream_callback = None

            route_label = format_route_label(result)

            if needs_human_review(result):
                # Compliance revised the draft -- don't show the streamed text;
                # the operator must review it in the HITL form first.
                placeholder.empty()
                st.session_state.pending_hitl = {
                    "response":    result["response"],
                    "route_label": route_label,
                }
                st.rerun()
            else:
                # Replace the streaming placeholder with the final clean text
                # (removes the ▌ cursor) and record it in history.
                placeholder.markdown(result["response"])
                st.caption(route_label)
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                st.session_state.routes.append(route_label)


if __name__ == "__main__":
    main()
