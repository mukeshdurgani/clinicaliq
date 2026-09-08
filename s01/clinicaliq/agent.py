"""
clinicaliq/agent.py
-------------------
Graph construction and the terminal loop.

Run the agent from the session folder:
    cd s01/
    python -m clinicaliq.agent

S14 guard + US-11 supervisor graph (mirrors WealthDesk's s14 pattern):
    START --> guard --> route_guard --> {classify | blocked}
    blocked  --> END   (static response, skips classify/compliance entirely)
    classify --> route_supervisor -->
        {call_documents_agent | call_services_agent | escalate | decline}
    call_documents_agent --> call_compliance_agent --> END
    call_services_agent  --> call_compliance_agent --> END
    escalate --> END   (static response, skips compliance)
    decline  --> END   (static response, skips compliance)

Session 12 pattern (mirrors WealthDesk's s12): call_compliance_agent invokes
the Compliance Agent subgraph, which critiques the specialist's draft and, on
a FAIL, revises just the flagged violation instead of hard-replacing the
whole response.
"""
import sqlite3
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .nodes import (
    blocked, call_compliance_agent, call_documents_agent, call_services_agent,
    classify, decline, escalate, guard, route_guard, route_supervisor,
)
from .state import ClinicalIQState
from .config import CHECKPOINT_DB, ESCALATE_RESPONSE, MCP_SERVER_PATH


def build_graph(checkpointer=None):
    """Build and compile the ClinicalIQ supervisor graph (US-11 + S14 guard)."""
    builder = StateGraph(ClinicalIQState)
    builder.add_node("guard",                             guard)    # S14: PII/injection/LlamaGuard input guard
    builder.add_node("blocked",                           blocked)  # S14: canned response for a guard-blocked turn
    builder.add_node("classify",                         classify)
    builder.add_node("call_documents_agent [subgraph]",  call_documents_agent)
    builder.add_node("call_services_agent [subgraph]",   call_services_agent)
    builder.add_node("escalate",                          escalate)
    builder.add_node("decline",                           decline)
    builder.add_node("call_compliance_agent [subgraph]",  call_compliance_agent)  # US-08: critique-revise guardrail on a specialist's draft

    builder.set_entry_point("guard") # START
    builder.add_conditional_edges("guard", route_guard, {
        "classify": "classify",
        "blocked":  "blocked",
    })
    builder.add_edge("blocked", END)

    builder.add_conditional_edges("classify", route_supervisor, {
        "call_documents_agent": "call_documents_agent [subgraph]",
        "call_services_agent":  "call_services_agent [subgraph]",
        "escalate":             "escalate",
        "decline":              "decline",
    })

    builder.add_edge("call_documents_agent [subgraph]", "call_compliance_agent [subgraph]")
    builder.add_edge("call_services_agent [subgraph]",  "call_compliance_agent [subgraph]")
    builder.add_edge("call_compliance_agent [subgraph]", END)
    # escalate/decline return static, pre-approved strings -- skip the compliance check
    builder.add_edge("escalate", END)
    builder.add_edge("decline", END)

    return builder.compile(checkpointer=checkpointer)  # optional checkpointer for persistence


# Module-level graph instance required by langgraph.json for LangGraph Studio.
# run() uses this directly rather than building a second copy.
graph = build_graph()


# ---------------------------------------------------------------------------
# Terminal loop (provided -- no changes needed)
# ---------------------------------------------------------------------------

def run(user_id: str = "cli") -> None:
    import os
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph = build_graph(checkpointer=SqliteSaver(conn))  # terminal app opts into disk persistence explicit

    # ASI03:2026 Identity & Privilege Abuse -- prefix thread_id with user_id so
    # checkpoints from different patients are namespaced in the shared SQLite
    # store. Prevents cross-patient memory access even if an attacker guesses
    # a UUID (default "cli" is the single-user terminal case; a multi-user
    # front end would pass its own authenticated user_id here).
    thread_id = f"{user_id}::{uuid4()}"
    config    = {"configurable": {"thread_id": thread_id}}

    if not MCP_SERVER_PATH.exists():
        print(f"[ClinicalIQ] WARNING: MCP server not found at {MCP_SERVER_PATH}")

    print("=" * 55)
    print("  ClinicalIQ | Apollo Health Clinic")
    print("  Tools  : via MCP (s01/mcp_server.py)")
    print("  Type 'quit' to exit")
    print("=" * 55)

    print(f"  Session: {thread_id[:8]}...")  # sanity check -- confirms config actually reached graph.invoke()
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        project = os.getenv("LANGSMITH_PROJECT", "batch1-wealthdesk")
        print(f"  Tracing : LangSmith ({project})")
    print("=" * 55)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nClinicalIQ: Session ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nClinicalIQ: Thank you for choosing Apollo Health Clinic. Goodbye!")
            break

        # "response": "" is a placeholder to satisfy the TypedDict contract.
        # The specialist agent (call_documents_agent/call_services_agent)
        # overwrites it; graph.invoke() returns the full merged state.
        result = _graph.invoke(
            {"customer_message": user_input, "response": "",
             "compliance_status": "", "specialist": "", "blocked_reason": ""},
            config=config,
        )

        blocked_reason = result.get("blocked_reason", "")
        if blocked_reason:
            print(f"\n[Guard: BLOCKED ({blocked_reason})]")
        else:
            route      = result.get("query_type", "?")
            specialist = result.get("specialist", "?")
            docs       = result.get("retrieved_docs", [])
            compliance = result.get("compliance_status", "")
            response   = result["response"]
            print(f"\n[Routed: {route} -> {specialist}]", end="")
            if docs and response != ESCALATE_RESPONSE:
                sources = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
                print(f"  [Retrieved {len(docs)} chunk(s) from: {', '.join(sorted(sources))}]", end="")
            if compliance:
                print(f"  [Compliance: {compliance}]", end="")
            print()
        print(f"\nClinicalIQ: {result['response']}")


if __name__ == "__main__":
    run()
