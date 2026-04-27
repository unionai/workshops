# Quick Start Script (~90 sec)

## Claude Research Agent in 90 Seconds

---

### HOOK (0:00–0:08)

**Show:** Final research report in the Flyte UI in the browser (pre-recorded)

**Say:**
"A research agent that searches the web, writes a report, and loops until it's good enough — Claude plus Flyte, no framework needed. Here's how."

---

### SETUP (0:08–0:30)

**Show:** Terminal, fast-paced

**Say:**
"Clone the repo. Install deps with uv. Start a Flyte devbox — you just need Docker running — set the config, and register your API keys as secrets."

```bash
cd tutorials/claude_agent_research
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
flyte start devbox
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
flyte create secret ANTHROPIC_API_KEY --project flytesnacks --domain development
flyte create secret TAVILY_API_KEY --project flytesnacks --domain development
```

---

### THE IDEA (0:30–0:45)

**Show:** Architecture diagram from README (or animate it simply)

**Say:**
"The pipeline plans sub-topics, fans out parallel research agents — each one is a Claude ReAct loop with web search — then synthesizes results and runs a quality check. If there are gaps, it loops back and fills them."

---

### KEY CODE (0:45–1:05)

**Show:** `agent.py` — zoom into the ReAct loop (lines ~104–138), highlight briefly

**Say:**
"Each agent is just a loop — send a prompt, Claude calls the search tool, we execute it, send results back, repeat until Claude writes the summary. That's it. No agent framework."

**Show:** Quick cut to `workflow.py` — the `research_pipeline` function

**Say:**
"Flyte wraps each step as a task — parallel fan-out, typed Pydantic models between steps, live reports in the UI."

---

### RUN IT (1:05–1:25)

**Show:** Terminal, run the command

```bash
flyte run workflow.py research_pipeline \
  --query "Latest advances in AI for healthcare" \
  --num_topics 2 --max_searches 2
```

**Show:** Switch to browser, speed up the Flyte UI — tasks fanning out, reports appearing, quality score badge

**Say:**
"Sub-topics planned, agents searching in parallel, synthesis, quality check — done. Full report."

---

### CLOSE (1:25–1:30)

**Show:** Repo link on screen

**Say:**
"Link in the description. Go build something."
