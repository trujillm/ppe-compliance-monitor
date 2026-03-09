"""Postgres MCP tool loader.

Connects to a postgres-mcp server (read-only / restricted mode) over SSE
and exposes its tools as LangChain tools for the ReAct agent.
"""

import os

from langchain_mcp_adapters.client import MultiServerMCPClient

from logger import get_logger

log = get_logger(__name__)

POSTGRES_MCP_URL = os.environ["POSTGRES_MCP_URL"]

_mcp_client = MultiServerMCPClient(
    {
        "postgres": {
            "url": POSTGRES_MCP_URL,
            "transport": "sse",
        }
    }
)


async def load_tools():
    """Load all postgres-mcp tools as LangChain tools."""
    tools = await _mcp_client.get_tools()
    log.info(
        "Loaded %d tools from postgres-mcp: %s",
        len(tools),
        [t.name for t in tools],
    )
    return tools
