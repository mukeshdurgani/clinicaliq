"""
clinicaliq/agent.py
-------------------
Graph construction and the terminal loop.

Run the agent from the session folder:
    cd s01/
    python -m clinicaliq.agent

Session 1 graph:
    START --> respond --> END
"""
import sqlite3
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .nodes import respond,classify,decline,route_query,escalate,retrieve_docs
from .state import ClinicalIQState
from .config import CHECKPOINT_DB, ESCALATE_RESPONSE

# ---------------------------------------------------------------------------
# TODO 5 of 5 -- build_graph
# ---------------------------------------------------------------------------
# Implement build_graph() so it:
#
#   1. Creates a StateGraph:
#        builder = StateGraph(ClinicalIQState)
#
#   2. Registers the respond node:
#        builder.add_node("respond", respond)
#
#   3. Sets the entry point (first node to run):
#        builder.set_entry_point("respond")
#
#   4. Connects respond → END (the graph exits after one response):
#        builder.add_edge("respond", END)
#
#   5. Compiles and returns the graph:
#        return builder.compile()
#
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Build and compile the ClinicalIQ LangGraph graph."""
    # START -> Classify -> based on route_query decide next node to be executed.
    builder = StateGraph(ClinicalIQState)
    builder.add_node("classify", classify) #Naming of node
    builder.add_node("decline", decline)
    builder.add_node("escalate", escalate)
    builder.add_node("respond", respond)
    builder.add_node("retrieve_docs", retrieve_docs)

    builder.set_entry_point("classify") # START
    builder.add_conditional_edges("classify", route_query, {
        "retrieve_docs": "retrieve_docs",
        "decline": "decline",
        "escalate": "escalate"
    })

    builder.add_edge("retrieve_docs", "respond")
    builder.add_edge("respond", END)
    builder.add_edge("escalate", END)
    builder.add_edge("decline", END)

    return builder.compile(checkpointer=checkpointer)  # optional checkpointer for persistence


# Module-level graph instance required by langgraph.json for LangGraph Studio.
# run() uses this directly rather than building a second copy.
graph = build_graph()


# ---------------------------------------------------------------------------
# Terminal loop (provided -- no changes needed)
# ---------------------------------------------------------------------------

def run() -> None:
    import os
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))  # terminal app opts into disk persistence explicit
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}
    print("=" * 55)
    print("  ClinicalIQ | Apollo Health Clinic")
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
        # respond() overwrites it; graph.invoke() returns the full merged state.
        #result = graph.invoke({"customer_message": user_input, "response": ""})
        
        result = _graph.invoke({"customer_message": user_input, "response": ""}, config=config,)
        route = result.get("query_type", "?")
        docs = result.get("retrieved_docs", [])
        response = result["response"]
        print(f"\n[Routed: {route}]")
        if docs and response != ESCALATE_RESPONSE:
            sources = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
            print(f"  [Retrieved {len(docs)} chunk(s) from: {', '.join(sorted(sources))}]")
        else:
            print()
        print(f"\nClinicalIQ: {result['response']}")


if __name__ == "__main__":
    run()
