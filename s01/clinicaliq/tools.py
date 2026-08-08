"""
clinicaliq/tools.py
-------------------
LLM clients + MCP tool loading for ClinicalIQ.

US-06 Part 2: query_doctor and query_service are no longer defined here as
local @tool functions -- they are loaded from the standalone MCP server at
../mcp_server.py via langchain-mcp-adapters. The graph and agent code are
otherwise unchanged; only how the tools are sourced changes.
"""
import asyncio
import os
import sys

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import (
    MAX_TOKENS, MCP_SERVER_PATH, MODEL_NAME, TEMPERATURE, TOOL_MODEL_NAME,
    classifier_MAX_TOKENS, classifier_TEMPERATURE,
)

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
    model=TOOL_MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=classifier_TEMPERATURE,
    max_tokens=classifier_MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# MCP client -- connects to mcp_server.py as a STDIO subprocess
# ---------------------------------------------------------------------------
# MultiServerMCPClient takes a dict mapping a server name we choose ("clinicaliq")
# to a connection config. For a stdio server: "command" is the interpreter to run
# it with (sys.executable -- the same Python running this process) and "args" is
# the script path.
_mcp_client = MultiServerMCPClient({
    "clinicaliq": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(MCP_SERVER_PATH)],
    }
})

# get_tools() is async and returns ready-to-bind LangChain tools -- their
# names/descriptions/schemas come from the server's own @mcp.tool() docstrings,
# no second copy of them needed here. Bridged to sync with asyncio.run() once,
# at module import time (this starts and stops the server subprocess once just
# to discover its tool schemas).
mcp_tools      = asyncio.run(_mcp_client.get_tools())
_tool_registry = {t.name: t for t in mcp_tools}
llm_with_tools = llm.bind_tools(mcp_tools)


def _extract_text(result) -> str:
    """MCP tool results come back as a list of content blocks, e.g.
    [{"type": "text", "text": "...", "id": "..."}]. Join the text blocks
    into the plain string the rest of the agent code expects."""
    if isinstance(result, list):
        return "\n".join(
            block.get("text", "") for block in result if isinstance(block, dict)
        )
    return str(result)


def _run_tool(tool_name: str, tool_args: dict) -> str:
    # tools loaded via langchain-mcp-adapters are async-only (they negotiate a
    # fresh MCP session per call), so bridge with asyncio.run() per call.
    if tool_name not in _tool_registry:
        return f"Unknown tool: {tool_name}"
    try:
        result = asyncio.run(_tool_registry[tool_name].ainvoke(tool_args))
        return _extract_text(result)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"
