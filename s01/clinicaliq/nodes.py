"""
clinicaliq/nodes.py
-------------------
Node functions for the ClinicalIQ graph.

Each node is a plain Python function:
  - Input : the full ClinicalIQState (read-only)
  - Output: a dict containing ONLY the keys this node changed
             (LangGraph merges it into the state automatically)
"""
from langchain_core import messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import SYSTEM_PROMPT,CLASSIFY_SYSTEM_PROMPT,ESCALATE_RESPONSE,DECLINE_RESPONSE
from .state import ClinicalIQState
from .tools import llm, classifier_llm


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

def classify(state: ClinicalIQState) -> dict:
    """Classify the query so route_query can pick the next node."""
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=state["customer_message"]),
    ]

    try:
       result = classifier_llm.invoke(messages)
       query_type = result.content.strip().upper()
       if query_type not in {"SIMPLE","COMPLEX","OUT_OF_SCOPE"}:
          query_type = "SIMPLE"
    except Exception as e:
        print(f"[ClinicalIQ] Classification error: {e}")
        query_type = "SIMPLE"
 
    return {"query_type": query_type}

def respond(state: ClinicalIQState) -> dict:
    """Call the LLM and return the agent's reply."""
    # raise NotImplementedError("TODO 4: implement respond() in clinicaliq/nodes.py")
    messages = [
      SystemMessage(content=SYSTEM_PROMPT),
      # HumanMessage(content=state["customer_message"])
    ]
    history = state.get("history", [])
    for turn in history:
      if turn["role"] == "user":
        messages.append(HumanMessage(content=turn["content"]))
      else:
        messages.append(AIMessage(content=turn["content"]))
        
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
      result = llm.invoke(messages)
      response_text = result.content
      # return {"response": result.content}
    except Exception as e:
      print(f"[ClinicalIQ] LLM error {e}")
      return {"response": "I am temporarily unavailable; please try again in some time"}
    
    new_history = history + [{"role": "user", "content": state["customer_message"]}, {"role": "assistant", "content": response_text}]
    
    return {"response": response_text, "history": new_history}
  
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
   if query_type == "COMPLEX":
      return "escalate"
   if query_type == "OUT_OF_SCOPE":
      return "decline"
   return "respond"
    
    
    
    #try:
    #    result = llm.invoke(messages)
    #    return {"response": result.content}
    #except Exception as exc:
    #    print(f"[ClinicalIQ] LLM call failed: {exc}")
    #    return {"response": "I am temporarily unavailable. Please try again in a moment."}

    