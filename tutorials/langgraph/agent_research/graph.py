"""
LangGraph research agent — a ReAct (Reason + Act) loop with web search.

This is a hand-built ReAct loop using StateGraph — not LangGraph's prebuilt
create_react_agent — so you can see exactly how the cycle works:

    agent → (tool calls?) → tools → agent → (loop) → END

The planner and synthesizer in workflow.py are simple single-shot LLM calls —
no graph needed. The research agent is where LangGraph shines, because it
needs a reasoning loop with tool calling and conditional routing.
"""

import logging

import flyte
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from tools.search import create_search_tool

log = logging.getLogger(__name__)


def build_research_graph(
    openai_api_key: str,
    tavily_api_key: str,
    max_searches: int = 3,
    model: str = "gpt-4.1-nano",
):
    """Build a ReAct research agent that uses Tavily search as a tool."""

    # ------------------------------------------------------------------
    # 1. Set up tools and LLM
    # ------------------------------------------------------------------
    web_search = create_search_tool(tavily_api_key)
    tools = [web_search]
    llm = ChatOpenAI(model=model, api_key=openai_api_key).bind_tools(tools)

    system_prompt = (
        f"You are a research agent. Your job is to thoroughly research a topic by searching the web. "
        f"Use the web_search tool up to {max_searches} times to gather information from different angles. "
        f"After gathering enough information, write a clear research summary with key findings and sources."
    )

    # ------------------------------------------------------------------
    # 2. Define the agent node
    # ------------------------------------------------------------------
    # The "Reason" step of ReAct. The LLM sees the full message history
    # and decides to either call a tool or return a final answer.
    @flyte.trace
    async def agent(state: MessagesState) -> MessagesState:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                log.info(f"[Research] Tool call: {tc['name']}({tc['args']})")
        elif response.content:
            log.info(f"[Research] Response: {response.content[:200]}")

        return {"messages": [response]}

    # ------------------------------------------------------------------
    # 3. Define the routing function
    # ------------------------------------------------------------------
    # After the agent responds, check if it wants to call tools.
    # If yes → route to "tools" node. If no → we're done.
    @flyte.trace
    async def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            log.info(f"[Research] Routing → tools ({len(last.tool_calls)} call(s))")
            return "tools"
        log.info("[Research] Routing → done (final answer)")
        return "__end__"

    # ------------------------------------------------------------------
    # 4. Build the graph
    # ------------------------------------------------------------------
    #   ┌──────────┐     tool_calls?     ┌───────┐
    #   │  agent   │ ──── yes ─────────→ │ tools │
    #   │ (reason) │ ◄───────────────── │ (act)  │
    #   └──────────┘                     └───────┘
    #        │ no tool_calls
    #        ▼
    #      END
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
