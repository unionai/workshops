"""
LangGraph research agent with tool calling.

The LLM decides when to search and what to search for using Tavily as a bound tool.

Graph: agent →(tool calls?)→ tools → agent →(loop)→ END
"""

import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient
import flyte

log = logging.getLogger(__name__)


def build_research_graph(openai_api_key: str, tavily_api_key: str, max_searches: int = 3):
    """Build a research agent that uses Tavily search as a tool."""
    tavily = TavilyClient(api_key=tavily_api_key)

    @tool
    @flyte.trace
    async def web_search(query: str) -> str:
        """Search the web for information on a topic. Use this to find current facts, data, and sources."""
        log.info(f"Searching: {query}")
        results = tavily.search(query=query, max_results=3)
        formatted = ""
        for r in results.get("results", []):
            formatted += f"- {r['title']}: {r['content'][:300]}\n  {r['url']}\n\n"
        return formatted or "No results found."

    tools = [web_search]
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=openai_api_key).bind_tools(tools)

    system_prompt = (
        f"You are a research agent. Your job is to thoroughly research a topic by searching the web. "
        f"Use the web_search tool up to {max_searches} times to gather information from different angles. "
        f"After gathering enough information, write a clear research summary with key findings and sources."
    )

    async def agent(state: MessagesState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    async def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "__end__": "__end__",
    })
    graph.add_edge("tools", "agent")

    return graph.compile()