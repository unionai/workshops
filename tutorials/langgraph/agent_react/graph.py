"""
LangGraph ReAct agent with math and string tools.

The LLM reasons about what tool to call, observes the result, and repeats
until it can answer the question. Classic Reason → Act → Observe loop.

Graph: agent →(tool calls?)→ tools → agent →(loop)→ END
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from tools.math import math_tools
from tools.string import string_tools

log = logging.getLogger(__name__)


def build_react_graph(openai_api_key: str, max_steps: int = 10):
    """Build a ReAct agent with math and string tools."""
    tools = math_tools + string_tools
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=openai_api_key).bind_tools(tools)

    system_prompt = (
        "You are a ReAct agent. You solve problems step-by-step using tools.\n\n"
        "Available tools:\n"
        "- add(a, b): Add two numbers\n"
        "- multiply(a, b): Multiply two numbers\n"
        "- power(a, b): Raise a to the power of b\n"
        "- word_count(text): Count words in a string\n"
        "- letter_count(text): Count letters in a string\n"
        "- reverse_string(text): Reverse a string\n\n"
        "For each step:\n"
        "1. THINK: Reason about what you need to do next\n"
        "2. ACT: Call a tool to get information\n"
        "3. OBSERVE: Look at the result and decide next step\n\n"
        "Break complex problems into small steps. Use one tool at a time. "
        "When you have the final answer, respond directly without calling tools."
    )

    step_count = 0

    async def agent(state: MessagesState):
        nonlocal step_count
        step_count += 1
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                log.info(f"[Step {step_count}] Tool call: {tc['name']}({tc['args']})")
        elif response.content:
            log.info(f"[Step {step_count}] Final answer: {response.content[:200]}")

        return {"messages": [response]}

    async def should_continue(state: MessagesState) -> str:
        nonlocal step_count
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            if step_count >= max_steps:
                log.info(f"[Step {step_count}] Max steps reached, stopping")
                return "__end__"
            log.info(f"[Step {step_count}] Routing → tools")
            return "tools"
        log.info(f"[Step {step_count}] Routing → done")
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
