
# Flyte & Union.ai Tutorials

Tutorials and examples for building AI agents, ML pipelines, and data workflows with [Flyte 2](https://flyte.org/).

---

## Featured

| Example | Description |
|---------|-------------|
| [LangGraph ReAct Agent](tutorials/starter-examples/langgraph-react-agent/) | Build a ReAct agent with LangGraph + OpenAI on Flyte |
| [Stable Diffusion](tutorials/starter-examples/stable-diffusion/) | Generate images from text prompts with SDXL Turbo on GPU |
| [DuckDB ETL](tutorials/starter-examples/duckdb-etl/) | Extract and transform data with DuckDB SQL |

---

## Get Started

| Tutorial | What you'll learn |
|----------|-------------------|
| [Flyte Basics](tutorials/starter-examples/flyte-basics/) | Flyte 2 fundamentals — tasks, pipelines, error handling, `TaskEnvironment`, `ReusePolicy`, `map()` |
| [LangGraph ReAct Agent](tutorials/starter-examples/langgraph-react-agent/) | Build a ReAct agent with tools in a single file |
| [Stable Diffusion](tutorials/starter-examples/stable-diffusion/) | GPU inference with Flyte reports |
| [Image Classifier](tutorials/starter-examples/image-classifier/) | Fine-tune ResNet18 on HuggingFace dataset with PyTorch |
| [DuckDB ETL](tutorials/starter-examples/duckdb-etl/) | Data pipeline with DuckDB SQL and Flyte reports |
| [Snowflake ETL](tutorials/starter-examples/snowflake-etl/) | ETL pipeline with the Snowflake connector |

---

## Agents

| Tutorial | Description |
|----------|-------------|
| [LangGraph ReAct Agent](tutorials/starter-examples/langgraph-react-agent/) | Single-file ReAct agent with LangGraph |
| [Planner Multi-Agent System](tutorials/multi-agent-workflows/tutorial_planner_agent.ipynb) | Scalable planner multi-agent system |
| [ReAct Multi-Agent System](tutorials/multi-agent-workflows/tutorial_react_agent.ipynb) | Adaptive ReAct multi-agent system |
| [Reflection Multi-Agent System](tutorials/multi-agent-workflows/tutorial_reflection_agent.ipynb) | Self-improving reflection agents |
| [Debate Multi-Agent System](tutorials/multi-agent-workflows/tutorial_debate_agent.ipynb) | Multi-agent debate pattern |
| [Manager Multi-Agent System](tutorials/multi-agent-workflows/tutorial_manager_agent.ipynb) | Manager-worker agent delegation |
| [Sequential Multi-Agent System](tutorials/multi-agent-workflows/tutorial_sequential_agent.ipynb) | Sequential agent pipeline |
| [LangGraph Tutorial](tutorials/langgraph/) | In-depth LangGraph integration with Flyte |

## MCP

| Tutorial | Description |
|----------|-------------|
| [MCP Recipe Assistant](tutorials/mcp/tutorial_recipe_mcp.ipynb) | Build and deploy a recipe assistant MCP server on Union |

## ML / AI

| Tutorial | Description |
|----------|-------------|
| [Stable Diffusion](tutorials/starter-examples/stable-diffusion/) | Image generation with SDXL Turbo |
| [Image Classifier](tutorials/starter-examples/image-classifier/) | Fine-tune ResNet18 on Beans dataset |

## Data

| Tutorial | Description |
|----------|-------------|
| [DuckDB ETL](tutorials/starter-examples/duckdb-etl/) | SQL-based data pipeline with DuckDB |
| [Snowflake ETL](tutorials/starter-examples/snowflake-etl/) | ETL with Snowflake connector |

---

## Setup

```bash
# Clone the repository
git clone https://github.com/unionai/workshops
cd workshops

# Navigate to any tutorial
cd tutorials/starter-examples/langgraph-react-agent

# Create virtual environment and install dependencies
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Flyte Cluster (for remote runs)

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks
```

Don't have a cluster? Request access at [flyte.org](https://flyte.org/).

### Run examples

**Remote:**
```bash
uv run flyte run langgraph_react_agent.py agent --request "What is 12 * 7 plus 3?"
```

**Local:**
```bash
uv run flyte run --local langgraph_react_agent.py agent --request "What is 12 * 7 plus 3?"
```

**Local with TUI:**
```bash
uv pip install textual

uv run flyte run --local --tui langgraph_react_agent.py agent --request "What is 12 * 7 plus 3?"
```

**Start TUI dashboard:**
```bash
uv run flyte start tui
```
