"""The support agent: the thing in production that the factory builds a router for.

A deliberately small LangGraph agent. A ticket comes in, the agent routes it to a queue,
then drafts a two-line reply for the queue's team to send. It is a stand-in for whatever
customer-facing agent you actually run; the point is that its first step, routing, is a
classification problem that today costs an API call per ticket and is the first thing to
move to a self-hosted open-source model.

    route ──► draft ──► END

Two routers, one flag:

    --router llm   the API model classifies the ticket (today's production)
    --router oss   the ticket-router app the factory deployed (tomorrow's)

Run it over a batch of held-out tickets and the report shows routing accuracy, p50
latency, tokens and cost per 1,000 tickets, and the distribution of queues, so "before" and
"after" sit next to each other. Run it over dataset v2 and the same report shows drift: the
GDPR tickets pile up in `other`, at low confidence.

    flyte run --local support_agent.py handle_tickets --n 10 --router llm
    flyte run support_agent.py handle_tickets --router oss
    flyte run support_agent.py handle_tickets --router oss --dataset v2      # drift, visible

Every ticket is a durable step, so a batch that dies halfway replays what it already did.
With Grafana configured, each batch is a conversation in Agent Observability.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TypedDict

import flyte
import flyte.report
from flyteplugins.agents.core import apply_instrumentation, durable_step, fingerprint
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import GRAFANA_LINKS, observed_env
from llm import DEFAULT_MODEL, chat_model, cost_usd, short_name
from report import support_html
from tickets import DATASET_VERSIONS, load_split, parse_label

QUEUE_OWNERS = {
    "billing": "Billing team",
    "refund": "Refunds desk",
    "shipping": "Logistics",
    "account_access": "Account security",
    "bug_report": "Engineering triage",
    "feature_request": "Product",
    "cancellation": "Retention",
    "data_request": "Privacy office",
    "other": "General support",
}


class TicketState(TypedDict, total=False):
    ticket_id: str
    text: str
    expected: str
    queue: str
    confidence: float | None
    route_ms: float
    reply: str
    router: str
    labels: list[str]


class Route(BaseModel):
    queue: str = Field(description="Exactly one of the allowed queue names.")


def build_support_graph(model, router: str, endpoint: str | None):
    """The two-node graph. Routing is either a model call or an HTTP call to the app."""
    structured = model.with_structured_output(Route)

    async def route(state: TicketState) -> dict:
        labels = state["labels"]
        t0 = time.perf_counter()
        if router == "oss":
            import httpx

            async def _call() -> dict:
                async with httpx.AsyncClient(timeout=120) as client:
                    r = await client.post(endpoint + "/classify", json={"text": state["text"]})
                    r.raise_for_status()
                    return r.json()

            key = fingerprint({"node": "route", "router": "oss", "endpoint": endpoint, "ticket": state["ticket_id"]})
            body = await durable_step(key, _call, name="route:app", dumps=json.dumps, loads=json.loads)
            queue = body["label"] if body["label"] in labels else "other"
            confidence = body.get("confidence")
            latency = body.get("latency_ms", (time.perf_counter() - t0) * 1000)
        else:
            msgs = [SystemMessage(content=system_prompt_for(labels)), HumanMessage(content=state["text"])]

            async def _call() -> str:
                return (await structured.ainvoke(msgs)).queue

            key = fingerprint(
                {
                    "node": "route",
                    "router": "llm",
                    "model": short_name(None),
                    "ticket": state["ticket_id"],
                    "labels": labels,
                }
            )
            raw = await durable_step(key, _call, name="route:model")
            queue = parse_label(raw, labels)
            confidence = None
            latency = (time.perf_counter() - t0) * 1000
        return {"queue": queue, "confidence": confidence, "route_ms": latency}

    async def draft(state: TicketState) -> dict:
        owner = QUEUE_OWNERS.get(state["queue"], "General support")
        msgs = [
            SystemMessage(
                content=f"You draft replies for the {owner}. Two sentences, plain, no promises you cannot keep, no sign-off."
            ),
            HumanMessage(content=f"Ticket routed to {state['queue']}:\n\n{state['text']}"),
        ]

        async def _call() -> str:
            return (await model.ainvoke(msgs)).content

        key = fingerprint(
            {"node": "draft", "model": short_name(None), "ticket": state["ticket_id"], "queue": state["queue"]}
        )
        reply = await durable_step(key, _call, name="draft:model")
        return {"reply": reply if isinstance(reply, str) else str(reply)}

    builder = StateGraph(TicketState)
    builder.add_node("route", route)
    builder.add_node("draft", draft)
    builder.add_edge(START, "route")
    builder.add_edge("route", "draft")
    builder.add_edge("draft", END)
    return builder.compile()


def system_prompt_for(labels: list[str]) -> str:
    return (
        "You are a support ticket router. Classify the ticket into exactly one queue from this list: "
        + ", ".join(labels)
        + ". Reply with the queue name only."
    )


async def _endpoint() -> str:
    """The app's endpoint, once it answers: a promotion may have just redeployed it."""
    from tools import _wait_for_app

    await flyte.init_in_cluster.aio()
    return await _wait_for_app()


@observed_env.task(report=True, retries=2, links=GRAFANA_LINKS)
async def handle_tickets(
    n: int = 30,
    router: str = "llm",
    dataset: str = "v1",
    labels_from: str = "v1",
    model: str | None = None,
    concurrency: int = 4,
) -> dict:
    """Run the support agent over n held-out tickets and report how the routing went.

    `dataset` is where the tickets come from; `labels_from` is the queue list the router
    knows about (production's, which is why v2 tickets on a v1 router show drift).
    """
    if router not in ("llm", "oss"):
        raise ValueError("router must be 'llm' or 'oss'")
    if dataset not in DATASET_VERSIONS or labels_from not in DATASET_VERSIONS:
        raise ValueError(f"datasets: {', '.join(DATASET_VERSIONS)}")
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    labels = DATASET_VERSIONS[labels_from]
    chat = chat_model(model)
    endpoint = None
    if router == "oss":
        ctx = flyte.ctx()
        if ctx is None or ctx.mode == "local":
            raise ValueError(
                "--router oss needs the deployed ticket-router app; run this on the cluster (or promote something first)"
            )
        endpoint = await _endpoint()
    graph = build_support_graph(chat, router, endpoint)
    usage = UsageMetadataCallbackHandler()
    config = apply_instrumentation("langgraph", {"callbacks": [usage]}) or {"callbacks": [usage]}

    tickets = load_split("test", dataset)[-max(1, min(n, 126)) :]
    sem = asyncio.Semaphore(concurrency)

    async def one(t) -> dict:
        async with sem:
            state: TicketState = {
                "ticket_id": t.id,
                "text": t.text,
                "expected": t.label,
                "router": router,
                "labels": labels,
            }
            out = await graph.ainvoke(state, config=config)
            return {
                "ticket_id": t.id,
                "text": t.text,
                "expected": t.label,
                "queue": out["queue"],
                "confidence": out.get("confidence"),
                "route_ms": round(out["route_ms"], 1),
                "reply": out["reply"],
            }

    started = time.perf_counter()
    rows = await asyncio.gather(*(one(t) for t in tickets))
    seconds = time.perf_counter() - started

    correct = sum(r["queue"] == r["expected"] for r in rows)
    tokens_in = sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values())
    tokens_out = sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values())
    spent = cost_usd(model, tokens_in, tokens_out)
    lat = sorted(r["route_ms"] for r in rows)[len(rows) // 2]
    dist = {q: sum(r["queue"] == q for r in rows) for q in labels}
    unknown_expected = sum(r["expected"] not in labels for r in rows)
    stats = {
        "router": router,
        "model": model or DEFAULT_MODEL,
        "dataset": dataset,
        "labels_from": labels_from,
        "n": len(rows),
        "routing_accuracy": round(correct / len(rows), 4),
        "route_p50_ms": round(lat, 1),
        "tokens": tokens_in + tokens_out,
        "cost_usd": None if spent is None else round(spent, 4),
        "cost_per_1000_tickets_usd": None if spent is None else round(spent / len(rows) * 1000, 2),
        "seconds": round(seconds, 1),
        "queue_distribution": dist,
        "tickets_with_unknown_queue": unknown_expected,
    }
    await flyte.report.replace.aio(support_html(rows, stats))
    await flyte.report.flush.aio()
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(handle_tickets, n=30, router="llm")
    print(run.url)
