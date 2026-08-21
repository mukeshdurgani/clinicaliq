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
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph
from langsmith import traceable
from .config import (
    SYSTEM_PROMPT,DOCS_SYSTEM_PROMPT,CLASSIFY_SYSTEM_PROMPT,ESCALATE_RESPONSE,DECLINE_RESPONSE,
    EMBED_MODEL,VECTORSTORE_DIR, RETRIEVAL_K,RETRIEVAL_SCORE_THRESHOLD,
    ESCALATE_ROUTING_ENABLED, ESCALATE_KEYWORD_PATTERNS, COMPLIANCE_BANNED_PHRASES, DB_PATH,
    SAFE_COMPLIANCE_RESPONSE, PROMPT_INJECTION_BLOCKLIST, MIN_QUERY_LENGTH, MAX_QUERY_LENGTH,
)
from .state import ClinicalIQState
from .tools import classifier_llm, llm, llm_with_tools, _run_tool

# ---------------------------------------------------------------------------
# Streamlit UI: token streaming hook (ported from WealthDesk's s13 nodes.py)
#
# Set by app.py to a callable before graph.invoke() when the Streamlit UI
# wants to display tokens as they arrive. None = silent (tests, CLI) --
# _doc_respond() and _services_respond() fall back to llm.invoke() in that case.
# ---------------------------------------------------------------------------
_stream_callback: Optional[Callable[[str], None]] = None

vectorstore = None  # shared across calls; initialised once by _init_vectorstore()

# ---------------------------------------------------------------------------
# US-08: Compliance Review Filter -- Compliance Agent (critique-revise loop)
# ---------------------------------------------------------------------------
# check_compliance() flags a violation; revise_response() then asks the LLM to
# rewrite just the flagged violation instead of hard-replacing the whole draft
# with SAFE_COMPLIANCE_RESPONSE (see call_compliance_agent()/create_compliance_agent()
# further down -- mirrors WealthDesk's s12 Compliance Agent).
#
# Compiled regex automaton: one NFA pass over the response text regardless of
# how many banned phrases are in the list (see config.py for the list itself
# and the PRD rules it covers).
_BANNED_PATTERN: re.Pattern = re.compile(
    "|".join(re.escape(p) for p in COMPLIANCE_BANNED_PHRASES),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Deterministic ESCALATE pre-filter -- see ESCALATE_KEYWORD_PATTERNS in
# config.py for what this catches and why (classifier LLM reliability).
# Only compiled when the toggle is on: with ESCALATE_ROUTING_ENABLED False,
# ESCALATE is not a valid classify() outcome at all, so there is nothing for
# this pattern to route to -- classify() below checks the same flag before
# ever consulting it.
# ---------------------------------------------------------------------------
_ESCALATE_PATTERN: "re.Pattern | None" = (
    re.compile("|".join(ESCALATE_KEYWORD_PATTERNS), re.IGNORECASE)
    if ESCALATE_ROUTING_ENABLED else None
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
    """Supervisor node: classify the query into SERVICES / POLICY / OUT_OF_SCOPE
    (+ ESCALATE, see toggle below) so route_supervisor() can send it to the
    right specialist agent."""
    msg = state["customer_message"].strip()

    # --- Input guardrails (ported from WealthDesk's s03 nodes.py) -----------
    # Cheap string checks before spending an LLM call: reject empty/near-empty
    # input (nothing to classify) and pathologically long input, and
    # short-circuit obvious prompt-injection phrasing straight to OUT_OF_SCOPE
    # rather than trusting the classifier LLM (or a specialist's system
    # prompt) to police it. Runs first, ahead of even the ESCALATE pre-filter
    # below -- an injection attempt should never be treated as a genuine
    # medical query just because it happens to share wording with one.
    if not msg or len(msg) < MIN_QUERY_LENGTH or len(msg) > MAX_QUERY_LENGTH:
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}
    if any(phrase in msg.lower() for phrase in PROMPT_INJECTION_BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}
    # -------------------------------------------------------------------------

    valid_types = {"SERVICES", "POLICY", "OUT_OF_SCOPE"}
    # --- ESCALATE routing toggle -------------------------------------------
    # See ESCALATE_ROUTING_ENABLED in config.py. When False, CLASSIFY_SYSTEM_PROMPT
    # never mentions ESCALATE, but as a belt-and-braces measure this also stops
    # a stray "ESCALATE" reply from the LLM being accepted -- it would fall
    # through to the `query_type not in valid_types` check below and reset to
    # POLICY, same as any other unrecognised reply.
    if ESCALATE_ROUTING_ENABLED:
        valid_types.add("ESCALATE")

        # --- Deterministic ESCALATE pre-filter ------------------------------
        # Safety net for classifier LLM misses on obvious emergency/symptom/
        # medication phrasing (see ESCALATE_KEYWORD_PATTERNS in config.py --
        # this is what caught "What medicine should I take for my fever?"
        # slipping through as SERVICES during manual testing). Checked BEFORE
        # the classifier LLM call, and short-circuits it entirely on a match --
        # the safety-critical route should not depend on an LLM call at all
        # when the query is this unambiguous.
        if _ESCALATE_PATTERN.search(state["customer_message"]):
            return {"query_type": "ESCALATE", "retrieved_docs": []}
        # ---------------------------------------------------------------------
    # -------------------------------------------------------------------------

    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=state["customer_message"]),
    ]

    try:
       result = classifier_llm.invoke(messages)
       query_type = result.content.strip().upper()
       if query_type not in valid_types:
          # Fall back to POLICY (not SERVICES) on a misclassification -- it
          # never triggers a live tool call, so a bad reply degrades to an
          # unnecessary ChromaDB lookup rather than an unnecessary DB write path.
          query_type = "POLICY"
    except Exception as e:
        print(f"[ClinicalIQ] Classification error: {e}")
        query_type = "POLICY"

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


# ---------------------------------------------------------------------------
# US-11: Documents Agent + Services Agent (specialist subgraphs)
# ---------------------------------------------------------------------------
# Mirrors WealthDesk's s10 supervisor pattern: classify() routes to one of two
# independently-compiled subgraphs instead of a single respond() node that did
# both RAG context injection and tool-calling. The Documents Agent only ever
# sees ChromaDB context (no MCP tools bound -- see DOCS_SYSTEM_PROMPT comment
# in config.py). The Services Agent only ever calls MCP tools (no ChromaDB
# context -- doctor/service data is live and must come from query_doctor/
# query_service, never from retrieved policy chunks).

def _doc_respond(state: ClinicalIQState) -> dict:
    """Documents Agent's respond step: ChromaDB context + plain LLM, no tools."""
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    if retrieved:
        context_block  = "\n\n---\n\n".join(retrieved)
        system_content = (
            DOCS_SYSTEM_PROMPT
            + "\n\nThe following sections from Apollo Health Clinic's policy documents "
            "are relevant to the customer's question. Use this information in your answer:\n\n"
            + context_block
        )
    else:
        system_content = DOCS_SYSTEM_PROMPT

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        # When the Streamlit UI is active, app.py sets _stream_callback to a
        # _StreamingState instance before graph.invoke() -- use llm.stream() so
        # each token can be pushed to the UI placeholder as it arrives.
        if _stream_callback is not None:
            response_text = ""
            for chunk in llm.stream(messages):
                if chunk.content:
                    response_text += chunk.content
                    _stream_callback(chunk.content)
        else:
            response_text = llm.invoke(messages).content or ""
    except Exception as e:
        print(f"[ClinicalIQ] Documents Agent LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [{"role": "user", "content": state["customer_message"]}, {"role": "assistant", "content": response_text}]
    return {"response": response_text, "history": new_history}


def _services_respond(state: ClinicalIQState) -> dict:
    """Services Agent's respond step: MCP tools (query_doctor/query_service),
    no ChromaDB context. Same multi-round tool-calling loop the old respond()
    used, unchanged."""
    history  = state.get("history", [])
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
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
        used_tools      = False
        while result.tool_calls and tool_rounds < max_tool_rounds:
            used_tools = True
            messages.append(result)
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(f"[ClinicalIQ] MCP tool: {tc['name']}({tc['args']}) -> {str(tool_output)[:80]}")
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
            tool_rounds += 1
            result = llm_with_tools.invoke(messages)

        if used_tools:
            # Tool results are in `messages` now -- generate the final answer
            # with the plain (non tool-bound) LLM so it can be streamed. Mirrors
            # WealthDesk's rates agent: always a dedicated post-tool call rather
            # than reusing the tool-bound decision call's (unstreamed) content.
            if _stream_callback is not None:
                response_text = ""
                for chunk in llm.stream(messages):
                    if chunk.content:
                        response_text += chunk.content
                        _stream_callback(chunk.content)
            else:
                response_text = llm.invoke(messages).content or ""
        else:
            # Direct response (no tool call) -- already computed, no re-invoke needed.
            response_text = result.content or ""
    except Exception as e:
        print(f"[ClinicalIQ] Services Agent LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [{"role": "user", "content": state["customer_message"]}, {"role": "assistant", "content": response_text}]
    return {"response": response_text, "history": new_history}


def create_documents_agent():
    """Compile the Documents Agent as a standalone subgraph: retrieve_docs -> respond -> END."""
    builder = StateGraph(ClinicalIQState)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("document_respond", _doc_respond)
    builder.set_entry_point("retrieve_docs")
    builder.add_edge("retrieve_docs", "document_respond")
    builder.add_edge("document_respond", END)
    return builder.compile()


def create_services_agent():
    """Compile the Services Agent as a standalone subgraph: respond -> END (tool
    calls happen inside _services_respond, not as separate graph nodes)."""
    builder = StateGraph(ClinicalIQState)
    builder.add_node("services_respond", _services_respond)
    builder.set_entry_point("services_respond")
    builder.add_edge("services_respond", END)
    return builder.compile()


_documents_agent = create_documents_agent()
_services_agent  = create_services_agent()


def call_documents_agent(state: ClinicalIQState) -> dict:
    """Supervisor node that invokes the Documents Agent subgraph and merges
    its result back into supervisor state."""
    print("[ClinicalIQ] Supervisor -> Documents Agent")
    result = _documents_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "POLICY"),
        "retrieved_docs":    [],
        "compliance_status": "",
        "specialist":        "",
    })
    return {
        "response":       result["response"],
        "retrieved_docs": result.get("retrieved_docs", []),
        "history":        result.get("history", state.get("history", [])),
        "specialist":     "documents_agent",
    }


def call_services_agent(state: ClinicalIQState) -> dict:
    """Supervisor node that invokes the Services Agent subgraph and merges
    its result back into supervisor state."""
    print("[ClinicalIQ] Supervisor -> Services Agent")
    result = _services_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "SERVICES"),
        "retrieved_docs":    [],
        "compliance_status": "",
        "specialist":        "",
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "services_agent",
    }


def check_compliance(state: ClinicalIQState) -> dict:
    """Compliance Agent's first node (US-08): runs _check_compliance() on the
    draft returned by either specialist agent and sets compliance_status only.
    Does NOT overwrite the response on FAIL -- revise_response() rewrites the
    flagged violation instead of hard-replacing it with a generic message."""
    draft          = state["response"]
    passed, reason = _check_compliance(draft)

    if not passed:
        print(f"[ClinicalIQ] Compliance FAIL: {reason}")
        return {"compliance_status": f"FAIL: {reason}"}

    print("[ClinicalIQ] Compliance PASS")
    return {"compliance_status": "PASS"}


def revise_response(state: ClinicalIQState) -> dict:
    """Compliance Agent's second node: instead of hard-replacing a failing
    draft with SAFE_COMPLIANCE_RESPONSE, ask the LLM to rewrite just the
    flagged violation while keeping the rest of the response helpful."""
    draft  = state["response"]
    reason = state.get("compliance_status", "violation").replace("FAIL: ", "")

    prompt = (
        "You are an Apollo Health Clinic compliance officer reviewing an AI "
        "patient-guidance response.\n\n"
        f"The response was flagged for: {reason}\n\n"
        "Rewrite it to fix the violation while keeping the response helpful.\n\n"
        "Rules:\n"
        "  1. Never diagnose a condition, name or recommend a medication, or "
        "promise a treatment outcome (e.g. 'guaranteed to cure', '100% effective')\n"
        "  2. Only state Rs. amounts that appeared in the original -- do not add new ones\n"
        "  3. Keep the rewritten response under 150 words\n"
        "  4. End with 'ClinicalIQ | Apollo Health Clinic'\n\n"
        f"Original response:\n{draft}\n\n"
        "Compliant rewrite:"
    )

    try:
        result       = llm.invoke([HumanMessage(content=prompt)])
        revised_text = result.content.strip() or SAFE_COMPLIANCE_RESPONSE
    except Exception as e:
        print(f"[ClinicalIQ] Compliance Agent revision error: {e}")
        revised_text = SAFE_COMPLIANCE_RESPONSE

    print("[ClinicalIQ] Compliance Agent: response revised")
    return {
        "response":          revised_text,
        "compliance_status": "REVISED",
    }


def route_compliance(state: ClinicalIQState) -> str:
    """Route to revise if check_compliance flagged a violation."""
    return "revise" if state.get("compliance_status", "").startswith("FAIL") else END


def create_compliance_agent():
    """Compile the Compliance Agent as a standalone subgraph:
    check_compliance -> (revise -> END) | END."""
    builder = StateGraph(ClinicalIQState)
    builder.add_node("check_compliance", check_compliance)
    builder.add_node("revise",           revise_response)
    builder.set_entry_point("check_compliance")
    builder.add_conditional_edges(
        "check_compliance",
        route_compliance,
        {"revise": "revise", END: END},
    )
    builder.add_edge("revise", END)
    return builder.compile()


_compliance_agent = create_compliance_agent()


def call_compliance_agent(state: ClinicalIQState) -> dict:
    """Supervisor node that invokes the Compliance Agent subgraph on the
    specialist's draft response, before the graph ends -- escalate()/decline()
    return static, pre-approved strings and skip this node entirely (see
    agent.py's graph wiring)."""
    print("[ClinicalIQ] Supervisor -> Compliance Agent")
    result = _compliance_agent.invoke({
        "customer_message":  state["customer_message"],
        "response":          state["response"],
        "history":           state.get("history", []),
        "query_type":        state.get("query_type", ""),
        "retrieved_docs":    state.get("retrieved_docs", []),
        "specialist":        state.get("specialist", ""),
        "compliance_status": "",
    })
    return {
        "response":          result["response"],
        "compliance_status": result.get("compliance_status", "PASS"),
    }


def escalate(state: ClinicalIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history, "specialist": "escalated"}

def decline(state: ClinicalIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history, "specialist": "declined"}

def route_supervisor(state: ClinicalIQState) -> str:
   """Supervisor routing function: sends the classified query to the right
   specialist agent (or escalate/decline)."""
   query_type = state.get("query_type", "POLICY")
   # --- ESCALATE routing toggle --------------------------------------------
   # See ESCALATE_ROUTING_ENABLED in config.py / classify() above. Turning the
   # toggle off means classify() can never produce "ESCALATE", which makes this
   # branch dead code automatically -- no need to touch it separately.
   if query_type == "ESCALATE":
      return "escalate"
   # -------------------------------------------------------------------------
   if query_type == "OUT_OF_SCOPE":
      return "decline"
   if query_type == "SERVICES":
      return "call_services_agent"
   return "call_documents_agent"
