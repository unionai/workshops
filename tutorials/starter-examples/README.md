# Starter Examples

Short, self-contained examples showing how to use Flyte for agents, ML, and data workflows. Each example is a single `.py` file with its own dependencies.

| Example | What it does | Key concepts |
|---------|-------------|--------------|
| [flyte-basics](flyte-basics/) | Get started with Flyte 2 fundamentals | `TaskEnvironment`, `ReusePolicy`, `map()` |
| [langgraph-react-agent](langgraph-react-agent/) | ReAct agent with LangGraph + OpenAI | `create_react_agent`, `@flyte.trace`, tools |
| [stable-diffusion](stable-diffusion/) | Generate images from text prompts | GPU tasks, `flyte.io.File`, `flyte.report` |
| [image-classifier](image-classifier/) | Fine-tune ResNet18 on HuggingFace dataset | PyTorch training, multi-task pipeline |
| [duckdb-etl](duckdb-etl/) | Extract and transform CSV data with SQL | DuckDB, pandas, Flyte reports |
| [snowflake-etl](snowflake-etl/) | ETL pipeline with Snowflake connector | Snowflake plugin, batch insert |

## Quick start

```bash
# Pick an example
cd tutorials/starter-examples/langgraph-react-agent

# Install dependencies
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Run locally
uv run flyte run --local langgraph_react_agent.py agent --request "What is 12 * 7 plus 3?"

# Run on a remote Flyte cluster
uv run flyte run langgraph_react_agent.py agent --request "What is 12 * 7 plus 3?"
```

For remote runs, configure your Flyte cluster first:

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks
```

Don't have a cluster? Request access at [flyte.org](https://flyte.org/).