"""
clinicaliq/nodes.py
-------------------
Node functions for the ClinicalIQ graph.

Each node is a plain Python function:
  - Input : the full ClinicalIQState (read-only)
  - Output: a dict containing ONLY the keys this node changed
             (LangGraph merges it into the state automatically)
"""
import re
import sqlite3
import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langsmith import traceable
from .config import (
    SYSTEM_PROMPT,CLASSIFY_SYSTEM_PROMPT,ESCALATE_RESPONSE,DECLINE_RESPONSE,
    EMBED_MODEL,VECTORSTORE_DIR, RETRIEVAL_K,RETRIEVAL_SCORE_THRESHOLD,
    ESCALATE_ROUTING_ENABLED, COMPLIANCE_BANNED_PHRASES, DB_PATH,
    SAFE_COMPLIANCE_RESPONSE,
)
from .state import ClinicalIQState
from .tools import classifier_llm, llm_with_tools, _run_tool
vectorstore = None  # shared across calls; initialised once by _init_vectorstore()

# ---------------------------------------------------------------------------
# US-08: Compliance Review Filter
# ---------------------------------------------------------------------------
# Compiled regex automaton: one NFA pass over the response text regardless of
# how many banned phrases are in the list (see config.py for the list itself
# and the PRD rules it covers).
_BANNED_PATTERN: re.Pattern = re.compile(
    "|".join(re.escape(p) for p in COMPLIANCE_BANNED_PHRASES),
    re.IGNORECASE,
)


def _normalize_for_check(text: str) -> str:
    # LLMs often output Unicode punctuation that looks identical to ASCII but
    # breaks substring matching. Replace all Unicode hyphen/dash variants with
    # an ASCII hyphen so phrases match regardless of which one the LLM used.
    text = unicodedata.normalize("NFKC", text)
    for ch in "‐‑‒–—―−":  # hyphen variants + minus sign
        text = text.replace(ch, "-")
    return text.lower()


def _extract_prices(text: str) -> list:
    """Extract all 'Rs. X' / 'Rs X' amounts from text as floats."""
    matches = re.findall(r"rs\.?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return [float(m) for m in matches]


def _load_valid_prices() -> set:
    # Reads SQLite directly (not via the MCP server) -- the compliance check
    # needs every valid consultation fee / service / package price as a set
    # of numbers for comparison, not the MCP tools' formatted display strings.
    try:
        conn     = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        fees     = conn.execute("SELECT consultation_fee FROM doctors").fetchall()
        services = conn.execute("SELECT price FROM services").fetchall()
        packages = conn.execute("SELECT price FROM health_packages").fetchall()
        conn.close()
        return {row[0] for row in fees + services + packages}
    except Exception:
        return set()


@traceable(name="apollo_compliance_check")
def _check_compliance(draft: str) -> tuple:
    """Checks the LLM draft response for:
      a) banned diagnosis / medication / outcome-promise phrases (PRD rules 1, 2, 5)
      b) a quoted Rs. amount that isn't a real fee/price in clinic_data.db

    Returns (True, "PASS") if both checks pass, else (False, "<reason>").
    """
    normalized = _normalize_for_check(draft)

    match = _BANNED_PATTERN.search(normalized)
    if match:
        return False, f"banned phrase: '{match.group()}'"

    mentioned_prices = _extract_prices(normalized)
    if mentioned_prices:
        valid_prices = _load_valid_prices()
        if valid_prices:
            for price in mentioned_prices:
                if price not in valid_prices:
                    return False, f"hallucinated price: Rs. {price:g} not in database"

    return True, "PASS"

#BLOCKLIST = [
#    "ignore all previous",
#    "forget everything",
#    "you are now",
#    "disregard your system",
#    "act as",
#    "jailbreak",
#]

# ---------------------------------------------------------------------------
# TODO 4 of 5 -- respond node
# ---------------------------------------------------------------------------
# Implement the respond() function so it:
#
#   1. Builds a messages list:
#        messages = [
#            SystemMessage(content=SYSTEM_PROMPT),
#            HumanMessage(content=state["customer_message"]),
#        ]
#
#   2. Calls the LLM inside a try / except block:
#        result = llm.invoke(messages)
#
#   3. On success  → return {"response": result.content}
#      On exception → print the error with a [ClinicalIQ] prefix
#                      and return a safe fallback string so the
#                      agent never crashes mid-conversation.
#
# ---------------------------------------------------------------------------

def _init_vectorstore() -> None:
    global vectorstore
    if vectorstore is not None:  # already loaded — skip the 90 MB model reload
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)  # loads ~90 MB model from ~/.cache/huggingface/
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),  # opens chroma.sqlite3 on disk — does NOT load all chunks into memory
            embedding_function=embeddings,            # same model used at ingest time — must match or retrieval breaks
        )
    except Exception as e:
        print(f"[ClinicalIQ] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")
 
 
def classify(state: ClinicalIQState) -> dict:
    """Call the LLM and return the agent's reply."""
    valid_types = {"IN_SCOPE", "OUT_OF_SCOPE"}
    # --- ESCALATE routing toggle -------------------------------------------
    # See ESCALATE_ROUTING_ENABLED in config.py. When False, CLASSIFY_SYSTEM_PROMPT
    # never mentions ESCALATE, but as a belt-and-braces measure this also stops
    # a stray "ESCALATE" reply from the LLM being accepted -- it would fall
    # through to the `query_type not in valid_types` check below and reset to
    # IN_SCOPE, same as any other unrecognised reply.
    if ESCALATE_ROUTING_ENABLED:
        valid_types.add("ESCALATE")
    # -------------------------------------------------------------------------

    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=state["customer_message"]),
    ]

    try:
       result = classifier_llm.invoke(messages)
       query_type = result.content.strip().upper()
       if query_type not in valid_types:
          query_type = "IN_SCOPE"
    except Exception as e:
        print(f"[ClinicalIQ] Classification error: {e}")
        query_type = "IN_SCOPE"
 
    return {"query_type": query_type, "retrieved_docs": []}

def retrieve_docs(state: ClinicalIQState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        # similarity_search_with_relevance_scores returns a list of (doc, score) tuples.
        #   doc.page_content : the raw chunk text (e.g. "PAN card and Aadhaar are required...")
        #   doc.metadata     : dict — e.g. {"source": "home_loan_guide.md"}
        #   score            : cosine similarity 0–1; higher = more relevant to the query
        results   = vectorstore.similarity_search_with_relevance_scores(
            state["customer_message"], k=RETRIEVAL_K
        )
        retrieved = []
        for doc, score in results:  # doc = LangChain Document object; score = float 0–1
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                retrieved.append(
                    f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                )
            else:
                print(
                    f"[ClinicalIQ] Chunk skipped (score {score:.2f} < {RETRIEVAL_SCORE_THRESHOLD}): "
                    f"{doc.metadata.get('source', 'unknown')}"
                )
    except Exception as e:
        print(f"[ClinicalIQ] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}


def respond(state: ClinicalIQState) -> dict:
    """Call the LLM (with MCP tools bound) and return the agent's reply.

    retrieved_docs is now one of two grounding sources, not the only one --
    doctor availability/pricing questions are grounded via query_doctor/
    query_service (see tools.py) instead of ChromaDB, so an empty
    retrieved_docs no longer forces escalation; it just means no policy
    document context is injected for this turn.
    """
    history = state.get("history",[])
    retrieved = state.get("retrieved_docs", [])

    if retrieved:
        context_block  = "\n\n---\n\n".join(retrieved)
        system_content = (
            SYSTEM_PROMPT
            + "\n\nThe following sections from Apollo Health Clinic's policy documents "
            "are relevant to the customer's question. Use this information in your answer:\n\n"
            + context_block
        )
    else:
        system_content = SYSTEM_PROMPT

    messages = [
        SystemMessage(content=system_content)
    ]
    for turn in history:
       if turn["role"] == "user":
          messages.append(HumanMessage(content=turn["content"]))
       else:
          messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm_with_tools.invoke(messages)

        if result.invalid_tool_calls:
            for itc in result.invalid_tool_calls:
                print(f"[ClinicalIQ] Invalid tool call ignored: {itc.get('name', 'unknown')} -- {itc.get('error', 'parse error')}")

        # Manual tool-calling loop: the MCP tools (query_doctor, query_service)
        # are bound to the LLM but run here, not inside LangGraph, so each
        # result can be logged and fed back as a ToolMessage before asking the
        # LLM for its next step.
        max_tool_rounds = 5
        tool_rounds     = 0
        while result.tool_calls and tool_rounds < max_tool_rounds:
            messages.append(result)
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(f"[ClinicalIQ] MCP tool: {tc['name']}({tc['args']}) -> {str(tool_output)[:80]}")
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
            tool_rounds += 1
            result = llm_with_tools.invoke(messages)

        response_text = result.content or ""
    except Exception as e:
        print(f"[ClinicalIQ] LLM error: {e}")
        return {"response": "I am temporarily unavailable. Please try again in a moment."}

    new_history = history + [{"role": "user", "content": state["customer_message"]}, {"role": "assistant", "content": response_text}]
    return {"response": response_text, "history": new_history}


def check_compliance(state: ClinicalIQState) -> dict:
    """Post-hoc guardrail on respond()'s draft (US-08). Runs after respond(),
    before the graph ends -- escalate()/decline() return static, pre-approved
    strings and skip this node entirely (see agent.py's graph wiring)."""
    draft          = state["response"]
    passed, reason = _check_compliance(draft)

    if not passed:
        print(f"[ClinicalIQ] Compliance FAIL: {reason}")
        return {
            "response":          SAFE_COMPLIANCE_RESPONSE,
            "compliance_status": f"FAIL: {reason}",
        }

    print("[ClinicalIQ] Compliance PASS")
    return {"compliance_status": "PASS"}


def escalate(state: ClinicalIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}
 
def decline(state: ClinicalIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}
 
def route_query(state: ClinicalIQState)->str:
   query_type = state.get("query_type","SIMPLE")
   # --- ESCALATE routing toggle --------------------------------------------
   # See ESCALATE_ROUTING_ENABLED in config.py / classify() above. Turning the
   # toggle off means classify() can never produce "ESCALATE", which makes this
   # branch dead code automatically -- no need to touch it separately.
   if query_type == "ESCALATE":
      return "escalate"
   # -------------------------------------------------------------------------
   if query_type == "OUT_OF_SCOPE":
      return "decline"
   return "retrieve_docs"
    
    
    
    #try:
    #    result = llm.invoke(messages)
    #    return {"response": result.content}
    #except Exception as exc:
    #    print(f"[ClinicalIQ] LLM call failed: {exc}")
    #    return {"response": "I am temporarily unavailable. Please try again in a moment."}

    