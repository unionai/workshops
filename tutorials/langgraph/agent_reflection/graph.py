"""
LangGraph reflection agent — generate, critique, and refine in a loop.

Graph: generate → critique →(score < threshold?)→ generate →(loop)→ END
"""

import json
import logging
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)


class ReflectionState(TypedDict):
    request: str
    response: str
    critique: str
    score: int
    iteration: int
    max_iterations: int
    quality_threshold: int
    history: Annotated[list[str], add]


def build_reflection_graph(openai_api_key: str, quality_threshold: int = 8, max_iterations: int = 3):
    """Build a reflection agent that generates, critiques, and refines."""
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=openai_api_key)

    def generate(state: ReflectionState) -> dict:
        is_refinement = state.get("iteration", 0) > 0
        if is_refinement:
            prompt = (
                f"Original request: {state['request']}\n\n"
                f"Your previous response:\n{state['response']}\n\n"
                f"Critique of your response:\n{state['critique']}\n\n"
                f"Rewrite your response addressing all the issues raised in the critique. "
                f"Keep what was good, fix what was bad."
            )
            log.info(f"[Iteration {state['iteration'] + 1}] Refining based on critique...")
        else:
            prompt = (
                f"Complete this task thoroughly and thoughtfully:\n\n{state['request']}"
            )
            log.info("[Iteration 1] Generating initial response...")

        response = llm.invoke(prompt)
        iteration = state.get("iteration", 0) + 1
        log.info(f"[Iteration {iteration}] Response: {response.content[:150]}")

        return {
            "response": response.content,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
            "history": [f"[Iteration {iteration}] Response: {response.content[:200]}"],
        }

    def critique(state: ReflectionState) -> dict:
        log.info(f"[Iteration {state['iteration']}] Critiquing response...")
        response = llm.invoke(
            f"You are a critical reviewer. Evaluate this response to the given task.\n\n"
            f"Task: {state['request']}\n\n"
            f"Response:\n{state['response']}\n\n"
            f"Provide your critique in this exact JSON format:\n"
            f'{{"score": <1-10>, "issues": ["issue1", "issue2"], "summary": "overall assessment"}}\n\n'
            f"Be strict but fair. Score 8+ means the response is good enough."
        )

        try:
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            score = int(parsed.get("score", 5))
            summary = parsed.get("summary", "No summary")
            issues = parsed.get("issues", [])
        except (json.JSONDecodeError, ValueError):
            score = 5
            summary = response.content[:200]
            issues = ["Could not parse structured critique"]

        critique_text = f"Score: {score}/10. {summary}. Issues: {', '.join(issues)}"
        log.info(f"[Iteration {state['iteration']}] Critique: {critique_text[:150]}")

        return {
            "critique": critique_text,
            "score": score,
            "history": [f"[Iteration {state['iteration']}] Critique: score={score}, {summary[:100]}"],
        }

    def should_continue(state: ReflectionState) -> str:
        if state["score"] >= state["quality_threshold"]:
            log.info(f"Score {state['score']} >= threshold {state['quality_threshold']} → done")
            return "__end__"
        if state["iteration"] >= state["max_iterations"]:
            log.info(f"Max iterations reached → done")
            return "__end__"
        log.info(f"Score {state['score']} < threshold {state['quality_threshold']} → refining")
        return "generate"

    graph = StateGraph(ReflectionState)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", should_continue, {
        "generate": "generate",
        "__end__": "__end__",
    })

    return graph.compile()
