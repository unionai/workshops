"""The engineer's graph. Yours to shape; Flyte supplies the durable nodes.

    START -> think -> (tools -> think)* -> decision -> END

`think` is `flyteplugins.agents.langgraph.ai_node`: it records every model turn with
`flyte.trace`, so a retry replays it instead of paying for it again. `tools` is our own
node, a variant of the plugin's `tool_node` that launches every tool call in a turn
concurrently, so when the model asks for three evals, three T4 containers start at once.
`decision` is also ours: a structured-output call that turns the transcript into a typed
`Decision`. It uses the same `durable_step` primitive, so it replays too.

`run_agent(agent=graph)` drives the compiled graph inside a Flyte task and renders the
timeline into the task report. With the Grafana plugin initialized, it also hands the
graph a callback handler, which is how every generation and tool call reaches Grafana
without this file knowing Grafana exists.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from dataclasses import dataclass
from typing import Literal

from flyteplugins.agents.core import ReportTimeline, abbrev, coerce_tool_args, durable_step, fingerprint
from flyteplugins.agents.langgraph import ai_node, run_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, messages_to_dict
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from llm import DEFAULT_MODEL, chat_model
from tools import TOOLS

MAX_TOOL_ROUNDS = 10


@dataclass(frozen=True)
class Request:
    """What the product team asked for. The constraints are what the agent is graded on."""

    min_accuracy: float = 0.95
    max_latency_ms: float = 150.0
    candidates_to_screen: int = 5  # how many base models to baseline (all of them, by default)
    max_fine_tunes: int = 3  # how many distinct candidates may be fine-tuned
    budget: int = 12  # run_eval + fine_tune calls in total (the other tools are free)
    dataset: str = "v1"  # which tickets the bar is measured on
    situation: str = ""  # anything the team wants the engineer to know first (steps 5 and 6 use this)
    fresh: bool = True  # True: build a router from the candidates. False: something is in production; check it first

    def text(self) -> str:
        opening = (
            "Request from the support team: we need a ticket router in production. Treat this as a fresh build: "
            "ignore whatever is currently deployed and pick the best candidate for the request below.\n"
            if self.fresh
            else "Request from the support team: keep the ticket router in production meeting the bar below.\n"
        )
        return (
            (self.situation + "\n\n" if self.situation else "")
            + opening
            + f'- Measured on dataset {self.dataset}. Pass dataset="{self.dataset}" to every eval, fine-tune and promote.\n'
            f"- Open weights, so we can self-host it.\n"
            f"- At least {self.min_accuracy:.0%} accuracy on our held-out tickets.\n"
            f"- Under {self.max_latency_ms:.0f} ms per ticket (p50, batch size 1, on a T4).\n"
            f"- Baseline up to {self.candidates_to_screen} candidate model(s). Fine-tune up to {self.max_fine_tunes} distinct candidates; "
            f"re-training the same candidate with more epochs or a bigger lora_r does not count as a new candidate.\n"
            f"- Spend at most {self.budget} eval or fine-tune runs in total. A run is one run_eval, run_eval_cpu or fine_tune call. "
            f"Use the budget: stopping early with runs left and no model promoted is a worse outcome than a heavier fine-tune.\n"
            "Independent runs can be launched together: ask for several tools in one turn and they run in parallel.\n"
            "If more than one model meets the bar, promote the smallest one: it is cheaper to serve.\n"
            "Measure, fine-tune if that is what it takes, promote exactly one model, then test the live deployment."
        )


SYSTEM_PROMPT = """\
You are the ML engineer on duty for the model factory. Your tools: see what is in
production, list candidate models, evaluate a model on held-out tickets, LoRA fine-tune a
model on our training tickets, promote one model to production, test the live deployment,
and roll back.

Work like a good engineer:
- Measure before you decide. Do not assume a model's accuracy or latency; run the eval.
- Small models are fast and weak; bigger ones are slower and stronger. Fine-tuning a small
  model on our own data usually beats both, but it costs a run and a few minutes.
- Choose the training you ask for. One epoch is a sensible start for a chat model; an
  encoder starts from an untrained head and wants two or three epochs, which still take
  seconds. If a fine-tune falls short of the bar, that is data, not a dead end: train
  longer or wider and measure again, as long as the budget allows.
- Independent work runs in parallel: request all the baseline evals in one turn, and all
  the fine-tunes in one turn. Each tool call is its own GPU job.
- Evaluate on the full set (n=120 or more) for any number you will put in the decision.
- Fine-tuned models are referenced by what fine_tune returns (an artifact reference or a path); pass it verbatim to run_eval and promote.
- The bar is the bar. 94.2% against a 95% bar is below it; do not round, and do not
  promote a model that misses it. If the budget runs out first, promote nothing and say
  the request could not be met within budget.
- When the request says to keep the router in production meeting the bar, measure the
  exact version production_status names (never an unversioned "latest": cached evals are
  keyed on the reference) before training anything. If it meets the bar, keep it: promote
  nothing, report its numbers. When the request says it is a fresh build, do not bother.
- Think about the shape of the problem before spending the last runs: a heavier fine-tune
  of the best candidate, or a different kind of model (an encoder is purpose-built for
  classification), can be a better bet than another baseline.
- Stay within the run budget. Promote exactly one model, with the numbers you actually measured.
- After promote, run test_deployment against the live app. If it fails, roll back to the
  previous production version and say so.
- If a tool returns an error, say so. Never report a promotion, a deployment test or a result
  that did not happen.

When the live app has passed its deployment test (or you have rolled back), say so in one
sentence and stop.
"""

DECISION_PROMPT = """\
Write up your decision now. Say what you did (promoted a model, kept the one in production,
or gave up), name the model, the accuracy and p50 latency you measured for it, whether it
meets every constraint, what the deployment test of the live app said, whether you rolled
back, and the evidence from the tool results that led you there. Count the eval and
fine_tune runs you spent.
"""


class Decision(BaseModel):
    """The structured outcome of the engineer's work."""

    action: Literal["promoted", "kept_production", "gave_up"] = Field(
        description="promoted: a new model was promoted. kept_production: the model already in production meets the bar, nothing was promoted. gave_up: nothing met the bar within budget."
    )
    promoted_model: str = Field(
        description="The model promoted (action=promoted), or the production model that was kept (kept_production), or empty."
    )
    base_model: str = Field(description="The candidate id it derives from (itself if not fine-tuned).")
    fine_tuned: bool
    accuracy: float = Field(ge=0, le=1, description="Measured accuracy of the promoted model on the full eval set.")
    latency_p50_ms: float = Field(description="Measured p50 latency per ticket of the promoted model.")
    meets_constraints: bool
    promoted: bool = Field(description="True only if the promote tool succeeded.")
    deployment_test_passed: bool | None = Field(
        default=None,
        description="What test_deployment said about the live app; None if it was not run or not applicable.",
    )
    rolled_back: bool = False
    runs_spent: int = Field(description="Number of run_eval and fine_tune calls made.")
    evidence: list[str] = Field(description="Two to five concrete numbers from the tool results.")
    rationale: str = Field(description="One paragraph the support team would understand.")


class FactoryState(MessagesState):
    decision: dict


def _action_label(tool_name: str, args: dict) -> str:
    """`run_eval · qwen2.5-0.5b`, `fine_tune · modernbert-base · 3ep`, `rollback · v1789611341`."""
    parts = [tool_name]
    model = args.get("model") or args.get("version")
    if model:
        model = str(model)
        if model.startswith("artifact:"):
            model = model.split("@")[0].removeprefix("artifact:ticket-router-")
        parts.append(model)
    if "epochs" in args:
        parts.append(f"{float(args['epochs']):g}ep")
    if args.get("dataset") and args["dataset"] != "v1":
        parts.append(str(args["dataset"]))
    return " · ".join(parts)[:60]


def parallel_tool_node(tools, *, name: str = "tools"):
    """Run every tool call in the last message concurrently; each is a durable child action.

    Same contract as the plugin's `tool_node` (a `ToolMessage` per call, in order, errors
    surfaced to the model as text), with `asyncio.gather` instead of a loop. On a cluster
    that turns "evaluate these three" into three containers running at the same time.
    """
    registry = {getattr(t, "name", getattr(t, "__name__", "")): t for t in tools}
    timeline = ReportTimeline()

    async def _one(call: dict) -> str:
        selected = registry.get(call["name"])
        if selected is None:
            return f"Error: unknown tool '{call['name']}'"
        args = call.get("args") or {}
        try:
            task = getattr(selected, "flyte_task", None)
            if task is not None:
                # Name the action after what it is doing, so the run graph reads
                # "run_eval · qwen2.5-0.5b" rather than five identical run_eval rows.
                # dataclasses.replace keeps resources, cache and retries; override() would not.
                named = dataclasses.replace(task, short_name=_action_label(call["name"], args))
                return str(await named.aio(**coerce_tool_args(task, args)))
            return str(await selected.ainvoke(args))
        except Exception as exc:  # surface tool errors back to the model, and keep a record
            return f"Error: {call['name']} failed: {exc}"

    async def _tools(state: dict) -> dict:
        from langchain_core.messages import ToolMessage

        calls = getattr(state["messages"][-1], "tool_calls", None) or []
        for call in calls:
            timeline.row(icon="🛠️", label=call["name"], meta="tool", detail=abbrev(str(call.get("args", {})), 160))
        outputs = await asyncio.gather(*(_one(c) for c in calls))
        results = []
        for call, output in zip(calls, outputs, strict=True):
            timeline.row(icon="🔧", label=call["name"], meta="tool result", detail=abbrev(output, 160))
            # A stable id, derived from the tool call. LangGraph assigns a random UUID to any
            # message without one, and `ai_node` fingerprints the whole transcript (ids
            # included) to find a turn's durable record. Random ids would mean no replay.
            results.append(
                ToolMessage(
                    content=output, tool_call_id=call.get("id", ""), name=call["name"], id=f"tool:{call.get('id', '')}"
                )
            )
        return {"messages": results}

    _tools.__name__ = name
    return _tools


def build_graph(model, tools=TOOLS, max_tool_rounds: int = MAX_TOOL_ROUNDS):
    """Compile the graph around a LangChain chat model."""
    think = ai_node(model, tools, name="think")
    execute = parallel_tool_node(tools, name="tools")
    structured = model.with_structured_output(Decision)

    async def decision(state: FactoryState) -> dict:
        messages = [*state["messages"], HumanMessage(content=DECISION_PROMPT)]

        async def _call() -> dict:
            return (await structured.ainvoke(messages)).model_dump()

        key = fingerprint({"node": "decision", "messages": messages_to_dict(messages)})
        data = await durable_step(key, _call, name="decision:model", dumps=json.dumps, loads=json.loads)
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        # The args of every call, keyed by call id, so the report can show what was asked.
        call_args = {
            tc["id"]: tc.get("args") or {}
            for m in state["messages"]
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        }
        tool_log = [
            {
                "name": m.name,
                "args": call_args.get(m.tool_call_id, {}),
                "output": str(m.content)[:600],
                "ok": not str(m.content).startswith("Error:"),
            }
            for m in tool_msgs
        ]
        thoughts = [
            str(m.content) for m in state["messages"] if isinstance(m, AIMessage) and m.content and not m.tool_calls
        ]
        turns = {
            "model_turns": sum(1 for m in state["messages"] if isinstance(m, AIMessage)) + 1,
            "tool_calls": len(tool_msgs),
            "tools_used": [m.name for m in tool_msgs],
            "tools_failed": [m.name for m in tool_msgs if str(m.content).startswith("Error:")],
            "tool_log": tool_log,
            "thoughts": thoughts[-3:],
        }
        final = AIMessage(content=json.dumps({"decision": data, "turns": turns}))
        return {"decision": data, "messages": [final]}

    def route(state: FactoryState) -> str:
        last = state["messages"][-1]
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        if getattr(last, "tool_calls", None) and rounds <= max_tool_rounds:
            return "tools"
        return "decision"

    builder = StateGraph(FactoryState)
    builder.add_node("think", think)
    builder.add_node("tools", execute)
    builder.add_node("decision", decision)
    builder.add_edge(START, "think")
    builder.add_conditional_edges("think", route, {"tools": "tools", "decision": "decision"})
    builder.add_edge("tools", "think")
    builder.add_edge("decision", END)
    return builder.compile()


async def engineer(request: Request, model_spec: str | None = None, model=None) -> dict:
    """Run the graph on one request. Call this from inside a Flyte task.

    Returns the decision plus what the run cost: model turns, tool calls, tokens, seconds.
    The token counts come from LangChain's usage callback, the same numbers Grafana sees.
    """
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    model = model or chat_model(model_spec)
    graph = build_graph(model)
    usage = UsageMetadataCallbackHandler()
    # Stable ids for the same reason as in parallel_tool_node: the transcript must hash the
    # same way on every attempt for the recorded model turns to replay.
    state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT, id="system"),
            HumanMessage(content=request.text(), id="request"),
        ]
    }
    started = time.perf_counter()
    text = await run_agent(state, agent=graph, name="model-factory", config={"callbacks": [usage]})
    seconds = time.perf_counter() - started

    final = json.loads(text)
    tokens_in = sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values())
    tokens_out = sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values())
    return {
        "request": request.__dict__,
        "model": model_spec or DEFAULT_MODEL,
        "decision": final["decision"],
        "stats": {
            **{k: v for k, v in final["turns"].items() if k not in ("tool_log", "thoughts")},
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "seconds": round(seconds, 1),
        },
        "tool_log": final["turns"]["tool_log"],
        "thoughts": final["turns"]["thoughts"],
    }
