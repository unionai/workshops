# Multi-Agent Workflows

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/unionai/workshops?quickstart=1)


Orchestrating multiple AI agents at scale using six distinct workflow patterns. Each workflow demonstrates a different approach to coordinating agents for complex tasks.

Since this example is a structured Python project we reccomend running it locally or in [GitHub code spaces](https://github.com/codespaces/new/unionai/workshops) if you want a quick oneline environment.


**Agent Pattern Comparison:**

| Pattern | Coordination | Best For | Run in Colab |
|---------|--------------|----------|--------------|
| **Planner** | Static plan → parallel waves | Known decomposition, maximize speed | <a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/multi-agent-workflows/tutorial_planner_agent.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| **ReAct** | Adaptive single agent | Exploration, unknown steps | <a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/multi-agent-workflows/tutorial_react_agent.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| **Reflection** | Self-improvement | High quality single outputs | <a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/multi-agent-workflows/tutorial_reflection_agent.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| **Sequential** | Fixed pipeline | Predictable workflows | <a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/multi-agent-workflows/tutorial_sequential_agent.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| **Debate** | Peer collaboration | Accuracy through consensus | <a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/multi-agent-workflows/tutorial_debate_agent.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| **Manager-Worker** | Hierarchical supervision | Quality control, complex projects | <a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/multi-agent-workflows/tutorial_manager_agent.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|


## Setup

### Local Setup

```bash
# Clone the repository
git clone https://github.com/unionai/workshops
cd workshops/tutorials/multi-agent-workflows

# Create virtual environment
uv venv .venv --python 3.11

# Activate the venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt

# Set your OpenAI API key (or add it to a .env file)
export OPENAI_API_KEY="your-key-here"
```

### Remote Setup (Flyte/Union)

To run workflows on a remote Flyte cluster instead of locally:

```bash
# Connect to the Flyte cluster
flyte create config \
    --endpoint tryv2.hosted.unionai.cloud \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks

# Store your API key as a Flyte secret
flyte create secret OPENAI_API_KEY
```

Run workflows remotely by omitting the `--local` flag:

```bash
# Remote (builds and pushes a container image on first run)
python -m workflows.planner --request "Calculate 15 times 7"

# Local
python -m workflows.planner --local --request "Calculate 15 times 7"
```

## Workflow Patterns

### 1. Planner
**Dynamic task planning with intelligent agent routing.**

The planner analyzes a request and creates a DAG (directed acyclic graph) of tasks with dependencies. Tasks without dependencies run in parallel, and results flow between agents via context injection.

```bash
python -m workflows.planner --local --request "Calculate 15 times 7 and then add 20"
```

Tutorial: `tutorials/tutorial_planner_agent.ipynb`

### 2. ReAct (Reasoning + Action)
**Adaptive agent execution with iterative reasoning and reflection.**

Each iteration follows a Reason → Act → Observe → Reflect loop. The agent reasons about what to do, chooses an agent, executes it, observes the result, and reflects on progress. Continues until the goal is achieved or max steps are reached.

```bash
python -m workflows.react --local --request "Find the GDP of France and Germany and compare them"
```

Tutorial: `tutorials/tutorial_react_agent.ipynb`

### 3. Reflection
**Iterative self-improvement through critique and refinement.**

Selects an appropriate agent, generates an initial response, then uses an LLM critic to score quality (1-10) and identify issues. The response is refined iteratively until a quality threshold is met.

```bash
python -m workflows.reflection --local --request "Calculate 5 factorial and explain the steps"
```

Tutorial: `tutorials/tutorial_reflection_agent.ipynb`

### 4. Debate
**Multiple agents debate to reach consensus.**

All agents solve the same task independently, then engage in multi-round debate where they critique each other's responses and refine their own. A final synthesis step combines the best parts (via judge or vote).

```bash
python -m workflows.debate --local --request "Calculate 5 factorial" --agents math,math,code --rounds 2
```

Tutorial: `tutorials/tutorial_debate_agent.ipynb`

### 5. Sequential
**Predefined agent pipelines with no LLM planning overhead.**

Executes a static sequence of agents with placeholder-based result flow (`{input}`, `{previous}`, `{previous_0}`, etc.). Simpler, faster, and cheaper than dynamic workflows.

Built-in pipelines: `write-review-edit`, `search-summarize`, `calculate-explain`, `weather-analyze`, `multi-calc`

```bash
python -m workflows.sequential --local --pipeline write-review-edit --input "Write a poem about AI"
```

Tutorial: `tutorials/tutorial_sequential_agent.ipynb`

### 6. Manager
**Hierarchical coordination with active supervision and quality gates.**

A manager creates a delegation plan, assigns tasks to worker agents, reviews each output with quality scoring, and requests revisions if below threshold. Final synthesis combines all approved outputs.

```bash
python -m workflows.manager --local --request "Build a REST API for user management"
```

Tutorial: `tutorials/tutorial_manager_agent.ipynb`

## Available Agents

| Agent | Description |
|-------|-------------|
| `math` | Arithmetic and multi-step calculations |
| `string` | String manipulation and analysis |
| `code` | Code generation and analysis |
| `planner` | Task planning and agent routing |
| `web_search` | Web search with summarization |
| `web_search_reflexion` | Web search with self-critique |
| `weather` | Weather information retrieval |
| `editor` | Text editing |
| `writer` | Content writing |

## Project Structure

```
tutorials/multi-agent-workflows/
├── README.md
├── config.py                          # Configuration settings
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables (API keys)
├── agents/                            # Agent implementations
│   ├── __init__.py
│   ├── code_agent.py
│   ├── editor_agent.py
│   ├── math_agent.py
│   ├── planner_agent.py
│   ├── string_agent.py
│   ├── weather_agent.py
│   ├── web_search_agent.py
│   ├── web_search_reflexion_agent.py
│   └── writer_agent.py
├── tools/                             # Tool definitions for agents
│   ├── __init__.py
│   ├── code_tools.py
│   ├── math_tools.py
│   ├── string_tools.py
│   ├── weather_tools.py
│   └── web_search_tools.py
├── workflows/                         # Workflow orchestration patterns
│   ├── __init__.py
│   ├── planner.py
│   ├── react.py
│   ├── reflection.py
│   ├── debate.py
│   ├── sequential.py
│   └── manager.py
├── utils/                             # Utility functions
│   ├── __init__.py
│   ├── decorators.py
│   ├── file_viewer.py
│   ├── logger.py
│   ├── plan_executor.py
│   └── summarizer.py
├── tutorials/                         # Jupyter notebook tutorials
│   ├── tutorial_planner_agent.ipynb
│   ├── tutorial_react_agent.ipynb
│   ├── tutorial_reflection_agent.ipynb
│   ├── tutorial_debate_agent.ipynb
│   ├── tutorial_sequential_agent.ipynb
│   └── tutorial_manager_agent.ipynb
└── traces/                            # Execution trace logs
    ├── debate_trace_log.jsonl
    ├── plan_executor_trace_log.jsonl
    ├── planner_trace_log.jsonl
    ├── react_trace_log.jsonl
    └── reflection_trace_log.jsonl
```