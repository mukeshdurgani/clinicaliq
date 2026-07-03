"""
clinicaliq/state.py
-------------------
The shared state that flows through the LangGraph graph.

Every node reads from this state and writes back a partial update.
Only define the shape here -- no logic.
"""
from typing import TypedDict


# ---------------------------------------------------------------------------
# TODO 3 of 5 -- State definition
# ---------------------------------------------------------------------------
# Define ClinicalIQState as a TypedDict with exactly two fields:
#
#   customer_message : str   -- the question the patient typed
#   response         : str   -- the answer ClinicalIQ will return
#
# Pattern:
#   class ClinicalIQState(TypedDict):
#       field_name: type
#
# ---------------------------------------------------------------------------

class ClinicalIQState(TypedDict):
    customer_message: str
    response: str


# Guard: raises at import time if the fields haven't been defined yet.
if "customer_message" not in ClinicalIQState.__annotations__:
    raise NotImplementedError(
        "TODO 3: define 'customer_message: str' and 'response: str' "
        "in ClinicalIQState in clinicaliq/state.py"
    )
