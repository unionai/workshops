"""
LangGraph research agent with tool calling.

This is a hand-built ReAct (Reason + Act) loop using StateGraph — not LangGraph's
prebuilt create_react_agent — so you can see exactly how the cycle works:

    agent → (tool calls?) → tools → agent → (loop) → END

The LLM decides when to search and what to search for using Tavily as a bound tool.
"""

import logging

import flyte
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from tools.search import create_search_tool

log = logging.getLogger(__name__)


def build_research_graph(openai_api_key: str, tavily_api_key: str, max_searches: int = 3, model: str = "gpt-4.1-nano"):
    """Build a research agent that uses Tavily search as a tool."""

    # ------------------------------------------------------------------
    # 1. Set up tools and LLM
    # ------------------------------------------------------------------
    # Create the Tavily web search tool and bind it to the LLM.
    # bind_tools() tells the model about available tools so it can
    # generate tool_calls in its response when it wants to search.
    web_search = create_search_tool(tavily_api_key)
    tools = [web_search]
    llm = ChatOpenAI(model=model, api_key=openai_api_key).bind_tools(tools)

    # The system prompt instructs the agent on how to behave —
    # how many searches to make, when to stop, and what to output.
    system_prompt = (
        f"You are a research agent. Your job is to thoroughly research a topic by searching the web. "
        f"Use the web_search tool up to {max_searches} times to gather information from different angles. "
        f"After gathering enough information, write a clear research summary with key findings and sources."
    )

    # ------------------------------------------------------------------
    # 2. Define the agent node
    # ------------------------------------------------------------------
    # This is the "Reason" step of ReAct. The LLM sees the full message
    # history and decides to either call a tool or return a final answer.
    @flyte.trace
    async def agent(state: MessagesState) -> MessagesState:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                log.info(f"[Agent] Tool call: {tc['name']}({tc['args']})")
        elif response.content:
            log.info(f"[Agent] Response: {response.content[:200]}")

        return {"messages": [response]}

    # ------------------------------------------------------------------
    # 3. Define the routing function
    # ------------------------------------------------------------------
    # After the agent responds, check if it wants to call tools.
    # If yes → route to the "tools" node. If no → we're done.
    @flyte.trace
    async def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            log.info(f"[Agent] Routing → tools ({len(last.tool_calls)} call(s))")
            return "tools"
        log.info("[Agent] Routing → done (final answer)")
        return "__end__"

    # ------------------------------------------------------------------
    # 4. Build the graph
    # ------------------------------------------------------------------
    # Two nodes: "agent" (LLM reasoning) and "tools" (tool execution).
    # The conditional edge after "agent" decides whether to loop or stop.
    #
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
    graph.add_edge("tools", "agent")  # After tools run, always go back to agent

    return graph.compile()