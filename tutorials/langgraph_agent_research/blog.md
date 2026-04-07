# LangGraph + Flyte: Turning `Send` into Parallel Containers

Most LangGraph tutorials run everything in a single process. That works for demos, but the moment you need real parallelism — separate compute, isolated resources, container-level observability — you hit a wall. LangGraph controls *logic*. It doesn't manage infrastructure.

That's where Flyte comes in. In this post, we'll walk through a research agent pipeline where LangGraph orchestrates the plan and Flyte provides the muscle. The key insight: **LangGraph's `Send` API maps directly to Flyte tasks, so each fan-out becomes a separate container running on a cluster.**

## The Architecture

The pipeline follows a plan-research-synthesize loop:

```
research_pipeline (LangGraph graph, running inside a Flyte task)
  ├── plan → split query into sub-topics
  ├── research (Send fan-out → Flyte tasks)
  │     ├── research_topic("topic A")  ┐
  │     ├── research_topic("topic B")  ├── parallel Flyte tasks
  │     └── research_topic("topic C")  ┘
  ├── synthesize → combine into report
  ├── quality_check → score + identify gaps
  │     ├── gaps found → identify_gaps → Send fan-out → research again
  │     └── good enough → finalize
  └── finalize → final report
```

LangGraph handles the control flow: planning, routing, quality gates, and deciding when to loop. Flyte handles the compute: each `research_topic` call runs as a separate task with its own container, resources, and observability. When you run locally, those tasks execute as async calls. When you deploy to a Flyte cluster, they spin up as independent containers — same code, no changes.

## Setting Up the Environment

First, the Flyte configuration. A single `TaskEnvironment` defines everything the tasks need: container image, dependencies, secrets, and resources.

```python
# config.py
import os
from dotenv import load_dotenv
import flyte

load_dotenv()

base_env = flyte.TaskEnvironment(
    name="research-pipeline-env",
    image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
    secrets=[
        flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
```

`.with_requirements()` points at the project's `requirements.txt`, so the container image gets the same dependencies you install locally. Each task gets 2 CPUs and 2GB of memory. Secrets are injected as environment variables at runtime — they never touch your code bundle.

## The Research Agent (ReAct Subgraph)

Each sub-topic gets its own ReAct agent. This is a standard LangGraph agent loop: the LLM decides whether to call a tool, the tool executes, the result feeds back to the LLM, repeat until done.

```python
# graph.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from tools.search import create_search_tool

log = logging.getLogger(__name__)

def build_research_subgraph(
    openai_api_key: str,
    tavily_api_key: str,
    max_searches: int = 3,
    model: str = "gpt-4.1-nano",
):
    """Build a ReAct research agent that uses Tavily search."""
    web_search = create_search_tool(tavily_api_key)
    tools = [web_search]
    llm = ChatOpenAI(model=model, api_key=openai_api_key).bind_tools(tools)

    system_prompt = f"""\
You are a research agent. Your job is to thoroughly research a topic by searching the web. \
Use the web_search tool up to {max_searches} times to gather information from different angles. \
After gathering enough information, write a clear research summary with key findings and sources."""

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
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "__end__": "__end__",
    })
    graph.add_edge("tools", "agent")

    return graph.compile()
```

The `@flyte.trace` decorator gives you span-level tracing in the Flyte UI. Every LLM call and tool call shows up as a trace — useful for debugging when you have multiple agents running in parallel.

The search tool itself is straightforward — a Tavily client wrapped as a LangChain tool:

```python
# tools/search.py
import logging
from langchain_core.tools import tool
from tavily import TavilyClient
import flyte

log = logging.getLogger(__name__)

def create_search_tool(tavily_api_key: str):
    """Create a web_search tool bound to a Tavily API key."""
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

    return web_search
```

## The Pipeline Graph: Where `Send` Meets Flyte

This is where it gets interesting. The pipeline graph is a LangGraph `StateGraph` that uses `Send` to fan out work. But instead of sending to in-process functions, each `Send` dispatches to a **Flyte task** — which on a cluster means a separate container.

```python
# graph.py
from langgraph.types import Send

def build_pipeline_graph(
    openai_api_key: str,
    tavily_api_key: str,
    research_task,       # <-- This is a Flyte task, passed in as a parameter
    model: str = "gpt-4.1-nano",
):
    llm = ChatOpenAI(model=model, api_key=openai_api_key)

    class PipelineState(TypedDict, total=False):
        query: str
        num_topics: int
        max_searches: int
        iteration: int
        max_iterations: int
        topics: list[str]
        research_results: Annotated[list[dict], operator.add]  # append-only
        synthesis: str
        score: int
        gaps: list[str]
        final_report: str
```

The `research_results` field uses `Annotated[list[dict], operator.add]` — LangGraph's reducer pattern. When multiple `Send` branches return results, they get concatenated automatically.

### Planning

The plan node asks the LLM to decompose the query into sub-topics:

```python
    @flyte.trace
    async def plan(state: PipelineState) -> dict:
        """Split the query into focused sub-topics."""
        query = state["query"]
        num_topics = state.get("num_topics", 3)

        response = llm.invoke(f"""\
Break this research question into exactly {num_topics} focused sub-topics. \
Return ONLY a JSON array of strings, nothing else.

Question: {query}""")
        try:
            topics = json.loads(response.content)
        except json.JSONDecodeError:
            topics = [query]

        topics = topics[:num_topics]
        log.info(f"[Plan] {len(topics)} sub-topics: {topics}")
        return {"topics": topics, "iteration": 1}
```

### The Fan-Out: `Send` to Flyte Tasks

Here's the critical part. `route_to_research` returns a list of `Send` objects — one per topic. Each `Send` targets the `"research"` node with a different topic:

```python
    def route_to_research(state: PipelineState) -> list[Send]:
        topics = state.get("gaps") or state["topics"]
        max_searches = state.get("max_searches", 2)
        return [
            Send("research", {"topic": t, "max_searches": max_searches})
            for t in topics
        ]
```

And the research node calls the Flyte task:

```python
    async def research(state: dict) -> dict:
        topic = state["topic"]
        max_searches = state.get("max_searches", 2)
        log.info(f"[Research] Dispatching to Flyte task: {topic}")

        result_json = await research_task(topic, max_searches)
        result = json.loads(result_json)
        log.info(f"[Research] Flyte task complete: {topic}")

        return {"research_results": [result]}
```

**This is the key integration.** `research_task` is a Flyte task passed in as a parameter. LangGraph doesn't know or care that it's a Flyte task — it just `await`s the result. But on a Flyte cluster, each of these calls spins up a separate container with its own CPU, memory, and execution context. Three topics means three containers running in parallel, each with its own ReAct agent doing web searches independently.

The `Send` API was designed for in-process parallelism in LangGraph. By pointing it at Flyte tasks, you get *infrastructure-level* parallelism for free. Same code, different runtime.

### Quality Gate and Iterative Deepening

After synthesis, the quality check node evaluates the report and identifies gaps:

```python
    @flyte.trace
    async def quality_check(state: PipelineState) -> dict:
        """Evaluate the synthesis and identify any gaps."""
        query = state["query"]
        synthesis = state["synthesis"]
        iteration = state.get("iteration", 1)
        max_iterations = state.get("max_iterations", 2)

        response = llm.invoke(f"""\
Evaluate this research report for the question: {query}

Report:
{synthesis}

Rate the report quality from 1-10 and identify any gaps or missing perspectives. \
Return JSON: {{"score": <int>, "gaps": [<string>, ...]}}
If the report is comprehensive (score >= 8) or there are no significant gaps, \
return an empty gaps list.""")

        try:
            evaluation = json.loads(response.content)
            score = evaluation.get("score", 8)
            gaps = evaluation.get("gaps", [])
        except json.JSONDecodeError:
            score = 8
            gaps = []

        # Don't loop forever
        if iteration >= max_iterations:
            gaps = []
            log.info(f"[Quality] Max iterations reached ({max_iterations}), finishing")

        log.info(f"[Quality] Score: {score}/10, Gaps: {len(gaps)} (iteration {iteration})")
        return {"score": score, "gaps": gaps, "iteration": iteration + 1}
```

If gaps are found and we haven't hit the iteration limit, the graph routes back through `identify_gaps` — which triggers another `route_to_research` fan-out. More `Send` calls, more Flyte tasks, more containers. The graph keeps looping until the quality score is good enough or the iteration budget runs out.

### Wiring the Graph

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

    return graph.compile()
```

## The Flyte Tasks

The workflow file defines two Flyte tasks. The first is the research worker — a task that runs the ReAct subgraph on a single topic:

```python
# workflow.py
import markdown
import flyte.report

env = base_env
MODEL = "gpt-4.1-nano"

def md_to_html(text: str) -> str:
    """Convert markdown to HTML for Flyte reports."""
    return markdown.markdown(text, extensions=["tables", "fenced_code"])

@env.task(report=True)
async def research_topic(topic: str, max_searches: int = 2) -> str:
    """Run the ReAct research agent on a single sub-topic."""
    log.info(f"[Research Task] Starting: {topic}")

    await flyte.report.replace.aio(f"<h2>Researching: {topic}</h2><p>Running searches...</p>")
    await flyte.report.flush.aio()

    graph = build_research_subgraph(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        max_searches=max_searches,
        model=MODEL,
    )
    result = await graph.ainvoke({
        "messages": [HumanMessage(content=f"Research this topic: {topic}")]
    })
    report = result["messages"][-1].content
    log.info(f"[Research Task] Done: {topic}")

    await flyte.report.replace.aio(f"<h2>{topic}</h2>{md_to_html(report)}")
    await flyte.report.flush.aio()

    return json.dumps({"topic": topic, "report": report})
```

The second is the orchestrator — a task that builds and runs the pipeline graph, passing `research_topic` as the compute backend:

```python
@env.task(report=True)
async def research_pipeline(
    query: str,
    num_topics: int = 3,
    max_searches: int = 2,
    max_iterations: int = 2,
) -> str:
    log.info(f"Starting research pipeline: {query}")

    # Build the pipeline graph, passing the Flyte task as the compute backend
    pipeline = build_pipeline_graph(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        research_task=research_topic,
        model=MODEL,
    )

    # Visualize the graphs in report tabs
    graph_tab = flyte.report.get_tab("Agent Graphs")

    png_bytes = pipeline.get_graph().draw_mermaid_png()
    img_b64 = base64.b64encode(png_bytes).decode()
    graph_tab.log(f"""\
<h2>Research Pipeline</h2>\
<img src="data:image/png;base64,{img_b64}" alt="Research pipeline" />""")

    subgraph = build_research_subgraph(OPENAI_API_KEY, TAVILY_API_KEY, max_searches, model=MODEL)
    sub_png = subgraph.get_graph().draw_mermaid_png()
    sub_b64 = base64.b64encode(sub_png).decode()
    graph_tab.log(f"""\
<h2>Research Agent (ReAct)</h2>\
<img src="data:image/png;base64,{sub_b64}" alt="ReAct research agent" />""")
    await flyte.report.flush.aio()

    # Run the pipeline
    result = await pipeline.ainvoke({
        "query": query,
        "num_topics": num_topics,
        "max_searches": max_searches,
        "max_iterations": max_iterations,
        "iteration": 0,
        "topics": [],
        "research_results": [],
        "synthesis": "",
        "score": 0,
        "gaps": [],
        "final_report": "",
    })

    # Build the report
    final_report = result["final_report"]
    sub_reports = result["research_results"]
    score = result.get("score", "N/A")
    iteration = result.get("iteration", 1) - 1

    await flyte.report.replace.aio(f"""\
<h2>Research Report</h2>\
<p><b>Query:</b> {query}</p>\
<p><b>Quality:</b> {score}/10 after {iteration} iteration(s)</p>\
<hr/>{md_to_html(final_report)}""")
    await flyte.report.flush.aio()

    log.info(f"Research pipeline complete. Score: {score}/10, Iterations: {iteration}")
    return json.dumps({
        "query": query,
        "report": final_report,
        "sub_reports": sub_reports,
        "score": score,
        "iterations": iteration,
    })
```

Both tasks use `report=True`, which gives them a live HTML report in the Flyte UI. The `research_topic` task updates its report as it progresses — first showing a "Running searches..." status, then replacing it with the final rendered report. The orchestrator renders the LangGraph graph diagrams as images in a separate report tab, and builds a final report with the quality score and synthesized output.

## Why This Combination Works

LangGraph and Flyte are solving different problems, and they compose naturally:

**LangGraph gives you agent logic.** Conditional routing, state management, tool calling, quality gates, iterative loops. It's a graph framework for building agents that reason and adapt.

**Flyte gives you production compute.** Container isolation, resource allocation, secrets management, caching, retries, observability, and a UI to watch it all happen. It's an orchestrator built for running workloads at scale.

The `Send` → Flyte task pattern is the bridge between them. In LangGraph, `Send` is a way to fan out work to parallel branches. By making each branch call a Flyte task, you get:

- **True parallelism**: Each researcher runs in its own container with dedicated CPU and memory. No GIL, no shared process, no resource contention.
- **Independent scaling**: Need more researchers? Add more topics. Flyte spins up more containers automatically.
- **Per-task observability**: Each researcher gets its own logs, traces, execution timeline, and live report in the Flyte UI. When one agent hangs or fails, you see exactly which one.
- **Retries and fault tolerance**: If a researcher container crashes, Flyte can retry it without rerunning the entire pipeline.
- **Resource isolation**: One researcher doing heavy computation doesn't slow down the others.

And the best part: **the code doesn't change between local and remote**. `flyte run --local` runs everything in-process. `flyte run` on a cluster fans out to containers. Same `Send`, same graph, same tasks.

## Running It

```bash
# Local with the TUI
flyte run --local --tui workflow.py research_pipeline \
  --query "Compare quantum computing approaches: superconducting vs trapped ion"

# Remote (on a Flyte cluster) — each researcher gets its own container
flyte run workflow.py research_pipeline \
  --query "Compare quantum computing approaches" \
  --num-topics 5 --max-searches 3 --max-iterations 3
```

The `--num-topics` flag controls how many parallel researchers to spin up. On a cluster, setting `--num-topics 5` means five containers running simultaneously, each with its own ReAct agent doing independent web searches. The quality gate can then identify gaps and spin up *more* containers for follow-up research — all driven by LangGraph's graph logic.

## Bonus: Adding a Gradio UI

The pipeline works great from the CLI, but sometimes you want a UI. With Flyte's `AppEnvironment`, you can wrap the pipeline in a Gradio app and deploy it to a cluster — same pattern as everything else.

The app has three modes:

1. **Local app + local task**: `RUN_MODE=local python app.py` — everything runs in-process
2. **Local app + remote task**: `python app.py` — Gradio runs locally, but kicks off the pipeline on a Flyte cluster
3. **Full remote**: `flyte deploy app.py serving_env` — the entire app runs on the cluster

```python
# app.py
import json
import os
from dotenv import load_dotenv
import flyte
import flyte.app
from workflow import research_pipeline

load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "remote")

serving_env = flyte.app.AppEnvironment(
    name="research-pipeline-ui",
    image=flyte.Image.from_debian_base(python_version=(3, 11)).with_pip_packages(
        "flyte>=2.1.2", "gradio", "langgraph>=1.0.7", "langchain-openai",
        "tavily-python", "markdown", "python-dotenv", "unionai-reuse",
    ),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[
        flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    requires_auth=False,
    port=7860,
)
```

The `AppEnvironment` is similar to `TaskEnvironment` — it defines the image, resources, and secrets — but it's designed for long-running apps instead of batch tasks.

The core of the app is a function that kicks off the pipeline as a Flyte task and streams results back to the UI:

```python
def run_query(query, num_topics, max_searches, max_iterations):
    """Kick off the research pipeline as a Flyte task, stream URL then result."""
    result = flyte.with_runcontext(mode=RUN_MODE).run(
        research_pipeline,
        query=query,
        num_topics=int(num_topics),
        max_searches=int(max_searches),
        max_iterations=int(max_iterations),
    )

    # Show the run link immediately
    run_url = getattr(result, "url", None)
    link_html = ""
    if run_url:
        url_str = str(run_url)
        if url_str.startswith("http"):
            link_html = f'<a href="{url_str}" target="_blank">View run on Flyte</a>'
            yield "", link_html
        else:
            link_html = f'<code style="font-size:0.85em;color:#666;">Local run: {url_str}</code>'
            yield "", link_html
    else:
        yield "", "Running..."

    # Wait for completion, then show the report
    result.wait()
    output = json.loads(result.outputs()[0])
    report = output["report"]
    score = output.get("score", "N/A")
    iterations = output.get("iterations", "N/A")

    header = f"**Quality:** {score}/10 | **Iterations:** {iterations}\n\n---\n\n"
    yield header + report, link_html
```

`flyte.with_runcontext(mode=RUN_MODE).run()` is the bridge between the app and the pipeline. In `remote` mode, it submits the pipeline as a Flyte execution and returns a handle with a `.url` — so the UI can immediately show a clickable link to watch the run on the platform. In `local` mode, it runs in-process.

The Gradio interface itself is standard — sliders for the parameters, a button to kick it off, and a markdown output for the report:

```python
def create_demo():
    import gradio as gr

    with gr.Blocks(title="Research Agent") as demo:
        gr.Markdown("# Research Agent\nAsk a question — the agent searches the web via Tavily and synthesizes a report.")

        with gr.Row():
            query = gr.Textbox(label="Research Question", placeholder="Compare quantum computing approaches: superconducting vs trapped ion", scale=3)
            submit = gr.Button("Research", variant="primary", scale=1)

        with gr.Row():
            num_topics = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Sub-topics")
            max_searches = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Max searches per topic")
            max_iterations = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Max quality iterations")

        run_link = gr.HTML()
        report = gr.Markdown(label="Report")

        inputs = [query, num_topics, max_searches, max_iterations]
        submit.click(fn=run_query, inputs=inputs, outputs=[report, run_link])
        query.submit(fn=run_query, inputs=inputs, outputs=[report, run_link])

    return demo
```

To deploy the app to a cluster, the `@serving_env.server` decorator tells Flyte how to launch it:

```python
@serving_env.server
def app_server():
    """Launch the Gradio app (called by Flyte on remote deployment)."""
    create_demo().launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    if RUN_MODE == "remote":
        flyte.init_from_config()
    create_demo().launch()
```

Running it:

```bash
# Fully local (no cluster needed)
RUN_MODE=local python app.py

# Local app, remote pipeline execution
python app.py

# Deploy the whole app to a Flyte cluster
flyte deploy app.py serving_env
```

The nice thing about this pattern is that the Gradio app doesn't need to know anything about LangGraph or the pipeline internals. It just calls `research_pipeline` as a Flyte task and renders whatever comes back. You could swap the pipeline for a completely different implementation and the app wouldn't change.

## Takeaway

LangGraph is great at agent orchestration. Flyte is great at running compute at scale. Instead of picking one, use both: let LangGraph control the *what* and Flyte control the *where*. The `Send` API is the natural seam between them — what starts as a fan-out in a graph becomes parallel containers on a cluster, with no code changes required.
