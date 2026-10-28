# An agent whose model and tools run on your Mac

A LangGraph agent answers questions about a database. Its brain is an open model served on your Mac's GPU by a Flyte app; its tools are Flyte tasks, so every query the model writes becomes a durable child action you can inspect in the run graph. No API key, no cloud, one laptop. Then the same agent runs on a cluster by changing one environment variable, or swaps its brain for Claude by changing another.

```bash
flyte start devbox --apple-silicon
flyte create config --devbox
python serve_model.py                                    # Qwen3-4B on the Mac's GPU, OpenAI-compatible
flyte run tools.py run_sql --query "SELECT COUNT(*) FROM Customer"    # a tool, with no model anywhere
flyte run agent.py ask --question "Which artist has the most albums?"
flyte run agent.py evaluate                              # 16 questions with known answers, fanned out
```

## Results

Sixteen questions about the [Chinook](https://github.com/lerocha/chinook-database) music store database (11 tables), scored against answers computed directly in SQL. Measured on an M5 Max, flyte 2.12.0, four questions in flight at a time.

| Brain | Correct | Tool calls per question | Seconds per question | Wall clock |
|---|---|---|---|---|
| Qwen3-4B on `mps` (default) | **13 / 16 (81%)** | 1.1 | 7.3 | 41 s |
| Qwen3-1.7B on `mps` | 7 / 16 (44%) | 1.8 | 8.9 | 47 s |

The 4B model gets the question into one correct query almost every time. The 1.7B model writes plausible SQL that answers the wrong question (`COUNT(*)` of the wrong table, a filter on the wrong column) and reports ids where names were asked. Two of its answers also hit a transient connection error from the app under concurrent load. Both numbers are with the schema handed to the model up front, which is the single biggest lever: without it, the 1.7B model spent 6.8 tool calls per question exploring tables and scored 31%.

What the three misses of the 4B model look like is in the **evaluate** report: an id instead of a name, a `HAVING` that should have been there, and one hallucinated table name it then corrected too late.

## Files

| File | What it does |
|---|---|
| `config.py` | The knobs: `ACCELERATOR` (where the model runs), `AGENT_MODEL` (which brain), `LLM_BASE_URL` (for local runs); `llm_base_url()` |
| `openai_server.py` | The OpenAI-compatible server from the serving tutorial: streaming, tool-call parsing, thinking off |
| `serve_model.py` | Deploys that server as an app with `gpu="AppleSilicon"` |
| `tools.py` | `list_tables`, `schema`, `describe_table`, `run_sql`: read-only SQLite tasks over Chinook |
| `agent.py` | The ReAct loop (`create_react_agent`), the `ask` and `evaluate` tasks, the question set and the scorer |
| `report_helpers.py` | Report CSS and SVG charts (shared with the sibling tutorials) |
| `apple-silicon-agent-tutorial.ipynb` | The same steps as a notebook |

## Setup

You need an Apple Silicon Mac (M1 or later), Docker Desktop, [`uv`](https://docs.astral.sh/uv/) and `flyte` 2.12 or later.

```bash
cd tutorials/apple-silicon-agent
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

flyte start devbox --apple-silicon
flyte create config --devbox
```

Qwen3-4B is ungated and about 8 GB in bf16, downloaded into your own `~/.cache/huggingface` on the app's first start. On a 16 GB Mac use the smaller model: `MODEL_NAME=Qwen/Qwen3-1.7B python serve_model.py`. The agent reads the same variable, so set it for `flyte run` too (or put it in `.env`).

## Run

### 0. Serve the brain

```bash
python serve_model.py
# Endpoint: http://apple-silicon-agent-llm-flytesnacks-development.localhost:30081
```

### 1. The tools, with no model anywhere

```bash
flyte run tools.py list_tables
flyte run tools.py schema
flyte run tools.py run_sql --query "SELECT Country, COUNT(*) n FROM Customer GROUP BY Country ORDER BY n DESC LIMIT 3"
```

Each tool is an `@env.task`: a function with typed inputs and outputs that Flyte runs as an action. `run_sql` refuses anything that is not a `SELECT` and caps results at 50 rows. The database downloads once per container (1 MB).

### 2. One question

```bash
flyte run agent.py ask --question "Which artist has the most albums?"
```

Open the run. The agent task has children: `schema` (called by the code, not the model, before the loop starts) and then each `run_sql` the model asked for, with the query as its input and the rows as its output. The task's report shows the same trace as a table.

### 3. Sixteen questions at once

```bash
flyte run agent.py evaluate
flyte run agent.py evaluate --concurrency 8
```

`evaluate` fans the questions out as concurrent `ask` actions. The model app serializes generation on its one GPU, so concurrency buys overlap between the model's turn and the tool's turn rather than raw parallelism; the report has the per-question table and the accuracy.

## Swap the brain

```bash
AGENT_MODEL=anthropic:claude-haiku-4-5 flyte run agent.py evaluate   # needs: flyte create secret ANTHROPIC_API_KEY --value sk-ant-...
AGENT_MODEL=openai:gpt-4.1-mini        flyte run agent.py evaluate   # needs: flyte create secret OPENAI_API_KEY --value sk-...
```

The agent only asks for the secret its provider needs; the local model needs none. Everything else (the tools, the loop, the scoring) is identical, which is the point of putting the model behind an OpenAI-compatible API.

## The same agent on a cluster

```bash
ACCELERATOR=L40s python serve_model.py        # the brain, on a CUDA card
ACCELERATOR=L40s flyte run agent.py evaluate  # the agent and its tools, in containers
```

The agent and tool tasks are ordinary CPU containers everywhere. Only the model app moves between a Mac GPU and a cluster GPU, and `config.py` is the only file that knows the difference.

## Things to know

- **Finding the model from a task.** The agent runs in a container and must reach the model app over the network. On a cluster, Flyte tells tasks how to address apps by name. On a devbox it does not (yet), and a pod cannot resolve the public `*.localhost:30081` names, so `config.llm_base_url()` falls back to the in-cluster service name. For `flyte run --local`, set `LLM_BASE_URL` to the app's public endpoint and the agent runs on your laptop against the model on your laptop's GPU.
- **Give small models the schema.** A 1.7B model that has to discover tables with `list_tables` and `describe_table` burns its context and its luck before it writes a query. Handing it the schema up front turned 31% into 44%; the 4B model then gets 81% with essentially one query per question. The tools are still there for when the schema is not enough.
- **Tool calls are plain text.** Qwen emits `<tool_call>{json}</tool_call>`; the server turns that into OpenAI `tool_calls`, and LangChain's `ChatOpenAI` never knows the difference. Thinking is off; an agent that reasons for forty seconds before each tool call is not what you want here.
- **The model runs unsandboxed on your Mac.** That is what `--apple-silicon` opts into. The tool tasks, on the other hand, stay in their containers.
