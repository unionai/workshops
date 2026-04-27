# Full Tutorial Script (~5 min)

## Build a Research Agent Pipeline with Claude + Flyte

---

### INTRO (0:00–0:20)

**Show:** Title card or repo README

**Say:**
"In this tutorial we're building a research agent pipeline — Claude does the thinking, Flyte orchestrates it. The pipeline breaks a research question into sub-topics, fans out parallel agents that search the web, synthesizes everything into a report, and loops if the quality isn't good enough. Let's set it up."

---

### SETUP — ENVIRONMENT (0:20–1:00)

**Show:** Terminal, cd into the project directory

**Say:**
"First, clone the repo and cd into the claude agent research directory."

```bash
cd tutorials/claude_agent_research
```

**Say:**
"Create a virtual environment and install dependencies."

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Show:** Scroll through `requirements.txt` briefly

**Say:**
"We need the Anthropic SDK, Tavily for web search, and Flyte 2.0 with the Flyte UI."

---

### SETUP — DEVBOX + SECRETS (1:00–1:45)

**Show:** Terminal

**Say:**
"Now we spin up a Flyte devbox. You'll need Docker installed — I won't walk through that, but make sure it's running."

```bash
flyte start devbox
```

**Show:** Devbox starting up, output showing the endpoint and registry URLs

**Say:**
"Devbox gives you a local Flyte cluster that behaves like a remote deployment — same UI, same secret management, same task isolation. Once it's up, set your config to point at it."

```bash
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

**Say:**
"This points your CLI at the devbox cluster — sets the endpoint, default project and domain, and uses the local image builder. Now any `flyte run` hits the devbox instead of running locally. Since it acts like a real cluster, we need to register secrets — not just a .env file."

```bash
flyte create secret ANTHROPIC_API_KEY --project flytesnacks --domain development
flyte create secret TAVILY_API_KEY --project flytesnacks --domain development
```

**Show:** Secret creation prompts (it will ask you to paste the key values)

**Say:**
"Flyte prompts you for the values. These get stored securely in the cluster and injected into your tasks at runtime — same way it works in production."

---

### CODE WALKTHROUGH — MODELS (1:45–2:15)

**Show:** `models.py` — full file

**Say:**
"Let's look at the code. We start with Pydantic models — TopicReport holds a topic and its research, QualityResult has a score and gaps, and PipelineResult wraps everything up at the end. Flyte natively serializes these between tasks, so there's no manual JSON wrangling — just typed data flowing through the pipeline."

---

### CODE WALKTHROUGH — AGENT (2:15–3:15)

**Show:** `agent.py` — scroll through at a steady pace

**Say:**
"agent.py is where the AI logic lives — no Flyte here, just pure Claude."

**Show:** Highlight `SEARCH_TOOL` dict

**Say:**
"We define a web search tool in Anthropic's tool-use format. This tells Claude it can search the web."

**Show:** Highlight `run_research_agent` function, especially the for loop

**Say:**
"The core is this ReAct loop. We send a prompt to Claude, Claude decides to call the search tool, we execute it with Tavily, send the results back, and repeat. When Claude has enough info, it stops and writes a summary. The key trick — once the search limit is hit, we remove the tools entirely, which forces Claude to summarize instead of searching forever."

**Show:** Briefly scroll past `plan_topics`, `synthesize_reports`, `evaluate_quality`

**Say:**
"Below that we have helper functions — plan_topics breaks the query into sub-topics, synthesize_reports combines them, and evaluate_quality scores the report and finds gaps. These all use simple Claude calls, no tools needed."

---

### CODE WALKTHROUGH — WORKFLOW (3:15–4:00)

**Show:** `workflow.py` — scroll through

**Say:**
"workflow.py is the Flyte layer. Each step is a Flyte task with a report decorator so you can see progress in the UI."

**Show:** Highlight `research_topic` task, point out return type

**Say:**
"research_topic calls the agent and returns a TopicReport. Notice the types — Flyte handles the Pydantic serialization."

**Show:** Highlight `research_pipeline` function

**Say:**
"The pipeline orchestrator ties it all together. Plan sub-topics, fan out research to parallel tasks with asyncio.gather, synthesize, quality check — and if there are gaps, loop back and research those too. Each iteration builds on the last."

**Show:** Highlight the quality loop (`if not gaps or score >= 8`)

**Say:**
"It stops when the score hits 8 out of 10 or we run out of iterations."

---

### DEMO — RUN IT (4:00–4:45)

**Show:** Terminal, then switch to browser

**Say:**
"Let's run it on the devbox."

```bash
flyte run workflow.py research_pipeline \
  --query "What are the latest advances in AI for healthcare?" \
  --num_topics 2 --max_searches 2
```

**Show:** Switch to browser — Flyte UI at localhost:30080, tasks appearing

**Say:**
"Just `flyte run` — no flags needed. It builds the image, pushes to the devbox registry, and runs on the cluster. Open the Flyte UI in your browser and you can see each task as it runs. You can see the sub-topics being planned, then the research agents fanning out in parallel — each one searching the web and building a report."

**Show:** Click into a research_topic task to show its report

**Say:**
"Click into any task to see its live report. Here's Claude searching and building up findings in real time."

**Show:** Wait for synthesis and quality check to complete

**Say:**
"After research, it synthesizes everything and runs a quality check. If the score is high enough, we get the final report."

**Show:** Final report output

**Say:**
"And there it is — a structured research report with findings from multiple angles, all orchestrated by Flyte."

---

### OUTRO (4:45–5:00)

**Show:** Architecture diagram from README or the project structure

**Say:**
"That's a full research agent pipeline — Claude for reasoning and search, Pydantic for typed data between steps, and Flyte for parallel orchestration and observability. The agent logic is completely separate from the pipeline, so you can swap models, add tools, or change the orchestrator without touching the other. Code's in the repo — link in the description."
