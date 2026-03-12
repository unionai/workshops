# Building a Research Agent Pipeline with LangGraph, Tavily and Flyte

Most AI agent tutorials show you one framework doing everything. But production agents need two things that don't naturally live in the same tool: **dynamic decision-making** (should I research more? is this good enough?) and **scalable compute** (run 100 researchers in parallel, each on its own container with retries).

LangGraph handles the first. Flyte handles the second. This tutorial shows how to integrate them — not as separate layers, but as co-orchestrators of the same pipeline.

We'll build a research agent that plans sub-topics, fans out parallel researchers using LangGraph's `Send` API dispatching to Flyte tasks, synthesizes results, evaluates quality, and loops back to research gaps. LangGraph controls the pipeline logic. Flyte provides the compute. The same pattern scales from 3 topics on your laptop to a full deep research system with 100+ parallel agents on a cluster.

## Why Two Orchestrators?

This is the question worth answering first. If LangGraph can build graphs and Flyte can run workflows, why use both?

Because they're good at different things:

**LangGraph** excels at agent logic — ReAct loops, conditional routing, quality gates, dynamic fan-out. It's built for LLMs making decisions in cycles. The graph is the control plane.

**Flyte** excels at production compute — each task gets its own container, resources, retries, caching, secrets, and real-time observability. It's built for running workloads at scale. The task is the compute plane.

In this pipeline, LangGraph decides *what* to research and *whether the research is good enough*. Flyte decides *how* to run each researcher — on what hardware, with what resources, with what retry policy. That's a natural boundary, and it's the integration point this tutorial demonstrates.

## The Architecture

Two graphs, one pipeline:

**The pipeline graph** (LangGraph) controls the research flow:

```
START → plan ──Send──→ research → synthesize → quality_check
                           ▲                       │
                           │                 gaps? │ no gaps
                  identify_gaps ◄──────────────────┤
                  (Send fan-out)                    ▼
                                                finalize → END
```

**The research subgraph** (LangGraph, inside each Flyte task) runs a ReAct agent:

```
agent → (tool calls?) → tools → agent → ... → END
```

The key integration point is the `research` node. LangGraph's `Send` API fans out one message per topic. Each message triggers a call to a Flyte task — `research_topic` — which runs the ReAct agent on its own compute. LangGraph controls the routing. Flyte provides the containers.

## The ReAct Research Agent

Each researcher is a ReAct (Reason + Act) agent — the pattern introduced by [Yao et al. (2022)](https://arxiv.org/abs/2210.03629). The LLM reasons about what it knows, acts by calling tools, observes the results, and loops until it has enough information.

The agent uses [Tavily](https://tavily.com/) for web search. Unlike standard search APIs that return links and snippets, Tavily returns extracted, relevant content — clean text an LLM can reason over directly. Less token waste, better research quality.

```python
def build_research_subgraph(openai_api_key, tavily_api_key, max_searches=3, model="gpt-4.1-nano"):
    web_search = create_search_tool(tavily_api_key)
    tools = [web_search]
    llm = ChatOpenAI(model=model, api_key=openai_api_key).bind_tools(tools)

    @flyte.trace
    async def agent(state: MessagesState) -> MessagesState:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    @flyte.trace
    async def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": "__end__"})
    graph.add_edge("tools", "agent")
    return graph.compile()
```

This is a hand-built ReAct loop, not LangGraph's prebuilt `create_react_agent`. The conditional edge is the key — it lets the LLM control the flow: search again, or write the final summary.

Notice the `@flyte.trace` decorators on the agent and routing nodes. These give you observability in Flyte's UI — you can see each reasoning step and tool call — without changing the graph structure.

## The Pipeline Graph: Where LangGraph Meets Flyte

The research pipeline is where the two frameworks actually integrate. The pipeline graph accepts a Flyte task as a parameter — this is how LangGraph dispatches to Flyte compute:

```python
def build_pipeline_graph(openai_api_key, tavily_api_key, research_task, model="gpt-4.1-nano"):
    llm = ChatOpenAI(model=model, api_key=openai_api_key)

    class PipelineState(TypedDict, total=False):
        query: str
        topics: list[str]
        research_results: Annotated[list[dict], operator.add]
        synthesis: str
        score: int
        gaps: list[str]
        final_report: str
        # ... plus iteration tracking fields
```

### Plan: Split the Question

The plan node breaks a research question into focused sub-topics:

```python
@flyte.trace
async def plan(state: PipelineState) -> dict:
    response = llm.invoke(
        f"Break this research question into exactly {num_topics} focused sub-topics. "
        f"Return ONLY a JSON array of strings, nothing else.\n\n"
        f"Question: {state['query']}"
    )
    topics = json.loads(response.content)
    return {"topics": topics, "iteration": 1}
```

### Fan-Out: LangGraph Send → Flyte Tasks

This is the integration point. LangGraph's `Send` API creates one message per topic. Each message triggers the `research` node, which calls the Flyte task:

```python
def route_to_research(state: PipelineState) -> list[Send]:
    """Create a Send for each topic — each dispatches to a Flyte task."""
    topics = state.get("gaps") or state["topics"]
    return [
        Send("research", {"topic": t, "max_searches": state.get("max_searches", 2)})
        for t in topics
    ]

async def research(state: dict) -> dict:
    """Run research on a single topic via a Flyte task."""
    topic = state["topic"]
    result_json = await research_task(topic, state.get("max_searches", 2))
    result = json.loads(result_json)
    return {"research_results": [result]}
```

`research_task` is a Flyte task passed in from the workflow. Locally, it runs as an async function. On a cluster, each call becomes a separate container with its own resources, retries, and observability. LangGraph doesn't know or care — it just awaits the result.

### Synthesize + Quality Gate: The Loop

After all researchers report back, the synthesize node combines results. Then the quality check evaluates the report and decides whether to loop:

```python
@flyte.trace
async def quality_check(state: PipelineState) -> dict:
    response = llm.invoke(
        f"Evaluate this research report for the question: {state['query']}\n\n"
        f"Report:\n{state['synthesis']}\n\n"
        f"Rate quality 1-10 and identify gaps. Return JSON: {{\"score\": <int>, \"gaps\": [...]}}"
    )
    evaluation = json.loads(response.content)
    # Don't loop forever
    if state.get("iteration", 1) >= state.get("max_iterations", 2):
        return {"score": evaluation["score"], "gaps": [], "iteration": state["iteration"] + 1}
    return {"score": evaluation["score"], "gaps": evaluation.get("gaps", []), "iteration": state["iteration"] + 1}

def after_quality_check(state: PipelineState) -> str:
    if state.get("gaps"):
        return "research_more"
    return "finalize"
```

If gaps are found, the graph routes to `identify_gaps` → `route_to_research` (Send fan-out on the gaps) → back through the research → synthesize → quality check loop. Each gap becomes a new Flyte task. The pipeline iterates until the quality is good enough or max iterations are reached.

### Wiring It Together

```python
graph = StateGraph(PipelineState)
graph.add_node("plan", plan)
graph.add_node("research", research)
graph.add_node("synthesize", synthesize)
graph.add_node("quality_check", quality_check)
graph.add_node("identify_gaps", identify_gaps)
graph.add_node("finalize", finalize)

graph.add_edge(START, "plan")
graph.add_conditional_edges("plan", route_to_research, ["research"])
graph.add_edge("research", "synthesize")
graph.add_edge("synthesize", "quality_check")
graph.add_conditional_edges("quality_check", after_quality_check, {
    "research_more": "identify_gaps",
    "finalize": "finalize",
})
graph.add_conditional_edges("identify_gaps", route_to_research, ["research"])
graph.add_edge("finalize", END)
```

## The Flyte Side: Tasks and Reports

The workflow file defines the Flyte tasks that the LangGraph pipeline dispatches to:

```python
@env.task(report=True)
async def research_topic(topic: str, max_searches: int = 2) -> str:
    """Run the ReAct research agent on a single sub-topic."""
    graph = build_research_subgraph(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        max_searches=max_searches,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content=f"Research this topic: {topic}")]})
    report = result["messages"][-1].content

    await flyte.report.replace.aio(f"<h2>{topic}</h2>{md_to_html(report)}")
    await flyte.report.flush.aio()
    return json.dumps({"topic": topic, "report": report})
```

Each `research_topic` task:
- Runs a full ReAct agent with web search
- Generates a live HTML report you can watch in the Flyte UI
- On a cluster, gets its own container, resources, and retry policy

The orchestrator builds the pipeline graph, passing the Flyte task as the compute backend:

```python
@env.task(report=True)
async def research_pipeline(query: str, num_topics: int = 3, max_searches: int = 2, max_iterations: int = 2) -> str:
    pipeline = build_pipeline_graph(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        research_task=research_topic,  # LangGraph dispatches to this Flyte task
        model=MODEL,
    )
    result = await pipeline.ainvoke({...})
```

This is the full integration: LangGraph owns the pipeline logic (plan, route, quality gate, loop). Flyte owns the compute (containers, resources, reports). The `research_task` parameter is the bridge between them.

## Run It

### Setup

```bash
git clone https://github.com/unionai/workshops
cd workshops/tutorials/langgraph/agent_research_pipeline

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Add your API keys to `.env`:

```
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
```

### Run locally

```bash
flyte run --local --tui workflow.py research_pipeline \
  --query "Compare quantum computing approaches: superconducting vs trapped ion"
```

You'll see the pipeline plan sub-topics, fan out researchers, synthesize, check quality, and potentially loop back to fill gaps — all in the live terminal UI.

### Deploy to a cluster

```bash
# One-time setup
flyte create config \
    --endpoint your-cluster.hosted.unionai.cloud \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project your-project

flyte create secret OPENAI_API_KEY
flyte create secret TAVILY_API_KEY

# Run
flyte run workflow.py research_pipeline \
  --query "Compare quantum computing approaches" \
  --num_topics 5 --max_searches 3 --max_iterations 3
```

On the cluster, each `research_topic` call becomes a separate container. The Flyte UI shows the full task tree — you can click into any researcher to see its report, traces, and logs in real time.

## Seeing It Run: The Flyte UI

This is where the integration really clicks. When you run the pipeline on a cluster, Flyte's UI shows the full execution as it happens.

### Task Tree and Parallel Execution

The orchestrator task (`research_pipeline`) fans out to multiple `research_topic` tasks in parallel. Each researcher spins up in its own container, runs its ReAct agent with Tavily searches, and reports back. You can see them all running simultaneously:

<!-- TODO: GIF of task tree showing research_pipeline → parallel research_topic tasks running -->

### Traces: Inside the Agent Loop

Click into any research task and you can see the `@flyte.trace` spans — each reasoning step, each tool call, each search query. This is what LangGraph's ReAct loop looks like from the Flyte observability side:

<!-- TODO: GIF of trace view showing agent → tool_call → web_search → agent cycle -->

### Live Reports

Each research task generates a live HTML report as it runs. The orchestrator combines them into a final report with tabs — one per sub-topic, plus the synthesized result with quality score:

<!-- TODO: GIF of report view showing tabs for each sub-topic and the final synthesis -->

### The Quality Gate Loop

When the quality check finds gaps, you can see the pipeline loop back — new research tasks spin up for the gap topics, run their agents, and feed back into synthesis. The iteration count in the final report shows how many rounds it took:

<!-- TODO: GIF of the quality gate triggering a second round of research tasks -->

The Flyte UI makes the invisible visible: you can watch LangGraph's routing decisions play out as real containers spinning up and completing in real time.

## What Each Framework Contributes

| Capability | LangGraph | Flyte |
|---|---|---|
| ReAct agent loop | Agent ↔ tools cycle with conditional routing | — |
| Pipeline routing | Plan → fan-out → synthesize → quality gate → loop | — |
| Dynamic fan-out | `Send` API creates one message per topic | Each `Send` dispatches to a Flyte task |
| Quality gates | LLM evaluates and decides to loop or finish | — |
| Parallel compute | — | Each researcher in its own container |
| Observability | `@flyte.trace` on graph nodes | Task-level logs, reports, traces |
| Retries & caching | — | Per-task retry policies and caching |
| Secrets | — | Injected securely per environment |
| Live reports | — | HTML reports with tabs per sub-topic |

Neither framework is doing the other's job. LangGraph isn't managing containers. Flyte isn't deciding whether to loop. The `research_task` parameter is the clean boundary between them.

## From Research Agent to Deep Research

This tutorial builds a research agent with quality gates and iterative deepening. It's not a full deep research system — products like ChatGPT Deep Research and Perplexity spend minutes doing multi-hop research across dozens of sources, building up structured understanding over many rounds. But the architecture here is the foundation for that.

The pattern scales directly toward deep research:

- **More researchers** — fan out to 50+ sub-topics, each on dedicated Flyte compute
- **Multi-hop reasoning** — chain research agents where one agent's output feeds another's input, adding new `Send` edges to the graph
- **Better tools** — add academic paper search, code execution, database queries alongside Tavily web search
- **Deeper iteration** — increase `max_iterations`, make the quality gate more rigorous, let the LLM identify more specific gaps
- **Structured knowledge** — accumulate findings across iterations instead of re-synthesizing from scratch

The architecture stays the same: LangGraph controls the routing and decisions, Flyte provides the compute. You just add more nodes, more tools, and more tasks. Going from 3 topics to 100 is a config change, not an architecture change — that's the point of having Flyte handle the scale.

We also have a [simpler version](../agent_research/) that uses LangGraph only for the ReAct agent loop, with Flyte handling all the orchestration. That's the right starting point if your pipeline is linear (plan → research → synthesize). This tutorial adds the LangGraph pipeline graph when you need conditional routing and quality gates — decisions an LLM makes mid-pipeline.

## Project Structure

```
agent_research_pipeline/
├── config.py           # Flyte environment, secrets, resources
├── graph.py            # LangGraph graphs — pipeline + ReAct subgraph
├── workflow.py         # Flyte tasks — research_topic + research_pipeline orchestrator
├── requirements.txt
└── tools/
    └── search.py       # Tavily web search tool
```

## Wrapping Up

The integration pattern here is straightforward: **LangGraph builds the graph, Flyte runs the tasks**. The pipeline graph accepts a Flyte task as a parameter. LangGraph's `Send` API fans out work. Each `Send` dispatches to a Flyte task with its own compute. The quality gate decides whether to loop. Flyte handles the rest — containers, retries, caching, reports.

This isn't about forcing two frameworks together. It's about using each for what it's best at:

- **LangGraph** for decisions: routing, quality gates, conditional loops, agent reasoning
- **Tavily** for search: clean, extracted web content purpose-built for AI agents
- **Flyte** for compute: parallel containers, retries, caching, secrets, observability

Start with 3 researchers on your laptop. Deploy to a cluster when you're ready. Scale to 100+ parallel agents, add multi-hop reasoning, and build toward a full deep research system — the architecture holds because each framework is doing what it does best.

The full code is at [github.com/unionai/workshops](https://github.com/unionai/workshops) under `tutorials/langgraph/agent_research_pipeline/`.
