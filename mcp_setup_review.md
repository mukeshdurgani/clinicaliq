# MCP Setup Review — ClinicalIQ vs WealthDesk (s07/s08)

The equivalent already exists in ClinicalIQ and works.

**Nothing to build** — `s01/mcp_server.py` and `s01/clinicaliq/tools.py` already mirror the
wealthdesk s07/s08 pattern, adapted to this project's domain:

| WealthDesk | ClinicalIQ | Status |
|---|---|---|
| `s07/starter/mcp_server.py` (`query_rates`, `query_branch`) | `s01/mcp_server.py` (`query_doctor`, `query_service`) | Implemented |
| `s08/starter/wealthdesk/tools.py` (MultiServerMCPClient wiring) | `s01/clinicaliq/tools.py` (same wiring) | Implemented |

Smoke test — `clinicaliq/tools.py` successfully spawns the MCP server subprocess and loads
both tools:

```
Tools loaded: ['query_doctor', 'query_service']
```

## Notes on how it's wired here vs. WealthDesk's cleaner split

- Everything landed in the single `s01/` folder rather than separate `s07`/`s08` folders —
  this project's session boundaries got collapsed (per `CLAUDE.md`, "only `s01/` exists so
  far" and features get added incrementally in place).
- `nodes.py:respond()` runs a manual tool-calling loop (up to 5 rounds) around
  `llm_with_tools`, feeding `ToolMessage`s back — same shape as what WealthDesk's `nodes.py`
  would do in s08/s09, just already present here.
- `config.py` uses `TOOL_MODEL_NAME = "openai/gpt-oss-20b"` specifically because Groq's
  llama-3.x models emit tool calls in a format Groq's own API rejects — noted in a comment,
  worth keeping in mind if you ever revert the model.
- All four files (`agent.py`, `config.py`, `nodes.py`, `tools.py`) show as modified/untracked
  in git — this is in-progress uncommitted work, not yet committed.
