"""
clinicaliq/tools.py
-------------------
LLM client setup.

Provided in full for Session 1 -- no changes needed here.
In later sessions this file will grow to include @tool functions
that let the agent query live databases.
"""
import os

from langchain_groq import ChatGroq

from .config import MAX_TOKENS, MODEL_NAME, TEMPERATURE, classifier_MAX_TOKENS, classifier_TEMPERATURE

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Did you copy .env.example to .env and fill in your key?\n"
        "  Windows:  copy .env.example .env\n"
        "  Mac/Linux: cp .env.example .env"
    )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=classifier_TEMPERATURE,
    max_tokens=classifier_MAX_TOKENS,
)
