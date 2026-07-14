# Code mode: agents that write programs instead of calling tools

An intro to code-mode agents, built on Flyte's sandbox. By the end you'll have an agent that
answers questions about 3 million NYC taxi trips a month by writing a Python program — and
that program's loops turn into parallel, durable, retryable containers.

---

## What is a code-mode agent?

Almost every tool-using agent you've seen works like this: the model emits one tool call as
JSON, your harness runs it, the result goes back into the model's context, the model reasons,
and it emits the next one. **One tool per round-trip.**

A **code-mode agent** does something different. Given the same tools, it writes *one program*
that calls them — with loops, conditionals, and variables — and hands you the whole thing at
once. The program runs in a sandbox where those tools are in scope. Only the final result
comes back to the model.

Same question, same tools, two shapes:

**Sequential tool calling** — 12 round-trips, and all 12 results land in the context window:

```
model → {"tool": "query", "args": {"month": "2024-01", ...}}   → 4,000 rows back into context
model → {"tool": "query", "args": {"month": "2024-02", ...}}   → 4,000 rows back into context
model → ... ten more times ...
model → {"tool": "create_chart", "args": {"values": [22.6, 22.4, ...]}}
model → "The tip rate declined slightly..."
```

**Code mode** — one round-trip, and the rows never reach the model at all:

```python
months = ["2024-01", "2024-02", ..., "2024-12"]
sqls = []
for m in months:
    sqls.append(sql)

per_month = flyte_map("query", sqls, months)   # 12 queries, in parallel

rates = []
for rows in per_month:
    r = rows[0]
    rates.append(round(100.0 * r["tips"] / r["fares"], 2))

create_chart("line", "Tip rate by month, 2024", months, rates)
"Tip rate held in a tight 21.4%-22.6% band all year."
```

That is not pseudocode — it's very close to what the model actually writes in step 3.

### Why it works

- **Models are better at code than at tool-call JSON.** They were trained on billions of
  lines of real code and comparatively little synthetic function-calling data. Writing a loop
  is the most natural thing a model does; emitting the twelfth JSON object in a sequence is
  not.
- **The data stays out of the context window.** In the sequential version, every row of every
  query passes through the model — and you pay for it, twice (in and out). In code mode the
  rows live in the sandbox; only the answer comes back. That's the single biggest lever.
- **Control flow is free.** Loops, conditionals, filtering, ranking, arithmetic: the model
  writes them once instead of re-deciding each step. No round-trip per iteration.
- **Fewer turns means less drift.** Each round-trip is a chance to forget the goal.

We measure all of this in step 4. Same question, same tools, same model — the only difference
is `code_mode=True`:

| | Sequential | Code mode |
|---|---|---|
| Model turns | 9 | **2** |
| Tool calls | 22 | 0 |
| Total tokens | 91,185 | **10,320** |
| Wall clock | 158s | **20s** |

### The catch: you have to run untrusted code

The moment an LLM writes a program, you have to run a program an LLM wrote. That's the whole
problem, and it's why code mode lives or dies on its sandbox.

Flyte runs the generated program in [Monty](https://github.com/pydantic/monty), a Rust-based
Python interpreter that starts in microseconds and can do exactly two things: **pure Python
control flow**, and **call the tools you registered**. There are no imports, no filesystem, no
network, no OS access — not disabled by policy, but *absent*. `import os` isn't blocked; it
doesn't parse.

That trade is the point. You give up arbitrary Python and get an execution environment where
the worst a hostile program can do is call your tools in a silly order.

### The Flyte twist: your tools can be tasks

Here's the part you don't get from a generic code interpreter.

The tools in scope inside the sandbox can be **Flyte tasks**. When the model's program calls
one, Monty *pauses*, Flyte dispatches that task to a real container — with network, with
dependencies, with retries — and Monty *resumes* with the result. The sandboxed code can't
tell the difference; it just looks like a function call that returned.

So when the model writes `flyte_map("query", sqls, months)` to answer a twelve-month
question, that is not a for-loop. **It's twelve containers running at once, each retried
independently, each cached, all visible in the UI.** The model wrote a distributed fan-out
without knowing it wrote a distributed fan-out.

Meanwhile the cheap tools — the chart and metric renderers — stay in-process, where a
round-trip would only add latency. The model calls all of them identically. *Where each one
runs is your decision, not its.*

### When not to use code mode

It isn't free, and it isn't always right:

- **One tool call?** Don't. If the answer is a single lookup, a program is overhead.
- **You need per-tool human approval.** Tools are invoked from *inside* the sandbox, so
  Flyte's per-tool HITL approval doesn't apply in code mode. If a human must approve each
  `send_email`, use sequential tool calling.
- **Your tools have side effects you can't undo.** A model that writes a loop can write a
  loop that fires 500 times. Keep the blast radius in the tool.
- **The prompt is now part of the runtime.** The model has to know the sandbox's rules, and a
  generic list of restrictions is not enough — see [Design notes](#design-notes).

---

## What you'll build

Five steps, each a runnable Flyte task. They share one container image, so the first run
builds it and the rest start warm.

| Step | What it teaches |
|---|---|
| **0** `step0_download_data.py` | Warm the dataset cache in blob storage. Optional, but run it once before a workshop. *(No API key needed.)* |
| **1** `step1_sandbox.py` | The sandbox, with **no LLM anywhere**. You write the control flow; Monty runs it. Grounds the mechanism before any model shows up. *(No API key needed.)* |
| **2** `step2_generated_code.py` | The model writes the program. The whole generate → execute → retry loop, hand-rolled in ~40 lines. |
| **3** `step3_agent_report.py` | The real agent, and the payoff: generated loops become **parallel durable tasks**. |
| **4** `step4_compare_modes.py` | Sequential vs code mode, measured. The table above. |
| **5** `step5_chat_app.py` | Serve it as a chat app — locally, or deployed with a durable run per message. |

The data is real: the [NYC Yellow Taxi](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
public trip records — about 3 million trips per month, queried with DuckDB. It has a genuine
analytical trap in it, on purpose (see below).

---

## Setup

```bash
cd tutorials/code-mode-analysis

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Steps 2–5 call a model; step 1 does not.

**For runs on the cluster,** register the key as a Flyte secret. The name must be
`ANTHROPIC_API_KEY` — the same name the task reads it as.

**Secrets live in a specific project and domain**, and a task can only read a secret from the
one it runs in. Pass `--project` / `--domain` explicitly so the secret lands where your runs
will look for it, rather than wherever your config happens to default:

```bash
flyte create secret ANTHROPIC_API_KEY --project flytesnacks --domain development
# Enter secret value: sk-ant-...      ← prompted, so the key stays out of your shell history
```

Non-interactive forms, if you prefer (`--value` is a flag; the value is *not* positional):

```bash
flyte create secret ANTHROPIC_API_KEY --value sk-ant-... \
    --project flytesnacks --domain development

flyte create secret ANTHROPIC_API_KEY --from-file ./key.txt \
    --project flytesnacks --domain development
```

Confirm it landed in the right place — and note the same flags apply here:

```bash
flyte get secret --project flytesnacks --domain development
```

If you run a step and the task fails with a missing-secret error, this is almost always the
cause: the secret exists, but in a different project or domain than the run. Use the same
`--project` / `--domain` you point `flyte run` at.

Already have a shared key under a different name? Change the `flyte.Secret(key=...)` in
`config.py` to match it.

**For runs with `--local`,** there is no cluster, so the same `flyte.Secret` resolves from your
environment instead. Drop the key in a `.env` and `config.py` loads it for you:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env    # .env is gitignored
```

Point `flyte` at your cluster (a `.flyte/config.yaml` in this directory is picked up
automatically):

```bash
flyte create config --endpoint <your-endpoint> --builder remote \
    --project flytesnacks --domain development
```

Want a different Claude model? `config.py` reads `CODE_MODE_MODEL` (default
`claude-opus-4-8`).

Want a different *provider* entirely? `Agent` takes `call_llm` as a plain async callback, and
`config.counting_llm()` is our implementation of it — about sixty lines, calling the official
Anthropic SDK. Swap that one function for any chat-completions endpoint (OpenAI, or a model
you host yourself with vLLM) and nothing else in the tutorial changes. **That callback is the
only provider-specific code here.**

---

## Running

Every step is an ordinary `flyte run`, so the usual flags work:

```bash
# On the cluster (the default). Queries fan out across real containers.
uv run flyte run step3_agent_report.py analyze \
    --question "How did the tip rate move month by month through 2024? Chart it."

# On your laptop. No cluster, no image build — good for iterating on prompts.
uv run flyte run --local step3_agent_report.py analyze --question "..."

# With the live TUI, to watch tasks as they run.
uv run flyte run --local --tui step3_agent_report.py analyze --question "..."

# Follow the logs of a remote run.
uv run flyte run --follow step3_agent_report.py analyze --question "..."
```

`flyte run` takes `--project` / `--domain` too (short: `-p` / `-d`). They default from your
config, but pass them when you want to be sure — **a run can only read secrets from its own
project and domain**, so these must match what you used in `flyte create secret`:

```bash
uv run flyte run -p flytesnacks -d development step3_agent_report.py analyze \
    --question "How did the tip rate move month by month through 2024? Chart it."
```

Two things to know about `--local`:

- **Secrets come from your environment, not the cluster** (see Setup).
- **`flyte_map` runs sequentially**, and Flyte warns you so. The generated code is identical
  and the answer is identical — but the fan-out is only *really* a fan-out on the cluster.
  Run step 3 remotely at least once and open it in the UI.

To list the tasks a file exposes, drop the task name: `uv run flyte run step1_sandbox.py`.

---

## The steps

### 0. Download the dataset (optional, but do it before a workshop)

```bash
flyte run step0_download_data.py download
```

Pulls all twelve months into blob storage, serially. No API key needed, a few minutes, once
per project.

You can skip it — `fetch_trips` is cached on the *month* (not the SQL), so the months get
fetched on demand by whoever asks first and are served from blob storage to everyone
afterwards. Step 3 works fine cold.

What step 0 buys you is **who pays for the cold start**. Without it, the first person to ask a
twelve-month question waits a minute or two while the data downloads — and if forty people
start asking at once, that's forty cold fan-outs reaching for the same CDN. Run this before
the room shows up and everybody's first question is instant.

### 1. The sandbox, with no LLM anywhere

```bash
flyte run step1_sandbox.py monthly_tip_trend
```

Before letting a model write code, look at what runs it. `@env.sandbox.orchestrator` marks a
task whose body executes inside Monty. It loops over months and calls `load_month` — and when
it does, Monty **pauses**, Flyte runs that task in a real container (which *does* have network
access, and fetches a month of taxi data), and Monty **resumes** with the result.

To the sandboxed code it looks like an ordinary function call. That pause / dispatch / resume
is the entire mechanism. Everything else in this tutorial is a consequence of it.

### 2. The model writes the program

```bash
flyte run step2_generated_code.py analyze \
    --question "Did tipping change between January and December 2024?"
```

`orchestrate_local` is the same sandbox, except the program arrives as a **string** — which
means it doesn't have to exist until runtime, and doesn't have to be written by a human.

This step hand-rolls the entire loop in about forty lines: ask the model for a program, run
it, and if it raises, hand the error back and let the model fix its own code. Read it once,
because step 3 hides it behind a class.

### 3. The agent, and the fan-out

```bash
flyte run step3_agent_report.py analyze \
    --question "Rank the boroughs by tip rate and show how it moved through 2024"
```

`Agent(code_mode=True)` runs that loop for you. One thing changes, and it's the important one:
`query` is now an `@env.task`. Every query the model writes dispatches through the Flyte
controller as a **durable child task** — retried, cached, visible in the UI.

**Open the run in the UI and look at the child tasks.** That graph is the payoff: one
`flyte_map` in generated code, twelve containers on the cluster.

### 4. Why bother? Measure it.

```bash
flyte run step4_compare_modes.py compare \
    --question "Which borough tipped best in each quarter of 2024?"
```

The same question, the same tools, the same model — run twice, `code_mode=False` and
`code_mode=True`. The report charts turns and tokens side by side (numbers in the table up
top).

Watch the **turn count**. Sequential needs a round-trip per tool, so its cost grows with the
work; code mode writes one program, so it doesn't.

Both agents get the *same* tools — but sequential has a harder time *using* them. It commonly
fumbles `create_table`, because handing a list of row dicts to a JSON tool call is awkward,
while in code mode a list of dicts is just a list of dicts. That friction is itself a cost of
sequential tool calling. Both answers are printed in the report, side by side.

### 5. Serve it as a chat app (stretch)

`AgentChatAppEnvironment` gives you the chat UI, the tools sidebar, streaming, and the chat
endpoint in one declaration. It runs two ways, and the difference is exactly the point of
step 3.

**Locally** — no cluster, no image build, no deploy:

```bash
python step5_chat_app.py       # → http://localhost:8080  (reads .env)

PORT=8099 uv run python step5_chat_app.py   # if 8080 is taken
```

Starts in seconds. The right way to iterate on the prompt or the UI. What you give up is the
thing step 3 was about: the sandbox's `query` calls run in-process too, so there are no
durable child tasks and no real fan-out.

**Deployed** — the app runs on the cluster and every message becomes a durable run:

```bash
python step5_chat_app.py deploy
```

`task_entrypoint=answer` is what buys that. An app's request handler has no task context, so
calling the agent straight from it would run the sandboxed queries inside the app pod. With a
task entrypoint, **each chat message is a real `flyte.run`**: you get the run in the UI, one
child `query` task per query the model wrote, an HTML report per message, and the progress bar
is that run's phase changes streamed to the browser. `passthrough_auth=True` forwards the
signed-in user's credentials, so the analysis executes as *them*, not as a shared identity.

(Both commands are `python`, not `flyte run`: one serves an app, the other deploys one.)

---

## Questions to ask it

All of these were actually run. Numbers are from `step3_agent_report.py` on Opus 4.8, so you
know what to expect before typing one in front of people.

**The trap — the best single demo in the tutorial.** The naive reading of this data says
outer-borough riders are stingy:

> *"Do riders in Brooklyn and the Bronx really tip less than Manhattan, or is something else
> going on?"*

It filters to card payments, finds Brooklyn at 8.2% and the Bronx at 1.6% against Manhattan's
25.3% — then keeps digging: **93.5% of Bronx card trips record a literal $0 tip**, versus
under 5% in Manhattan. Then it *rules out* the obvious explanation, showing Brooklyn (4.8%
cash) and the Bronx (5.7%) actually take *less* cash than Manhattan (12.9%). Conclusion: a
data-capture artifact, not stingy passengers.

That's a real analytical finding, reached by writing code. For the sharpest version, delete
the cash-payment warning from `DATA_DESCRIPTION` in `dataset.py`, ask again, and compare — the
cheapest possible proof that **the tool description is the contract**.

**The twelve-month fan-out** — one `flyte_map`, twelve parallel query tasks:

> *"How did the tip rate move month by month through 2024? Chart it."*
> — 2 turns, ~9k tokens, ~12s. *"The tip rate barely moved all year, holding in a tight
> 21.4%–22.6% band."*

**Fan-out plus a zone join** — the richest report of the set (8 blocks):

> *"How do JFK and LaGuardia airport pickups compare on trip distance, fare, and tip rate
> across 2024?"*
> — *"JFK trips are longer and pricier — 15.9 miles and a \$64 fare versus LaGuardia's 9.7
> miles and \$44 — but LaGuardia riders tip better."*

**One month, so no fan-out.** Worth showing *because* it doesn't fan out — the model
parallelizes when there's something to parallelize, and not otherwise:

> *"What time of day do people tip best? Break March 2024 down by hour of pickup."*
> — *"People tip best on the evening commute: card trips picked up around 6–7 PM average about
> 27% tips, peaking at 19:00."*

**In the chat app, ask a follow-up.** This is what the chat gives you that the CLI doesn't —
the conversation is memory, so the next question leans on the last answer:

> **You:** *"Which pickup zones have the longest average trips in March 2024?"*
> **It:** *"The longest trips leave from the airports: JFK tops the list at 16.2 miles per
> trip (across ~149K rides)."*
>
> **You:** *"Now compare those to the shortest ones. Do longer trips tip better?"*
> **It:** *"No — longer trips tip **worse**. The 10 longest-trip zones average 11.9% tips
> versus 26.3% for the 10 shortest, and across all 78 zones distance and tip% are negatively
> correlated (**−0.47**)."*

It computed that correlation by writing code, not by eyeballing it.

---

## Design notes

Three decisions in this code that carry over to any code-mode agent you build.

### The prompt is part of the runtime

The model has to know the sandbox's rules, and the SDK's generic syntax prompt is not enough
on its own. `config.SANDBOX_RULES` spells out the three restrictions models most reliably
trip over:

- **No methods on primitives.** `str(x)`, not `x.__str__()`. Only `.append()` on a list.
- **`flyte_map` takes the name of a *registered task*** as a string, plus one list per
  argument of that task — so `query(sql, month)` needs `flyte_map("query", sqls, months)`, not
  a locally defined helper. Get this wrong and the fan-out quietly degrades into a sequential
  loop.
- **Every program runs in a fresh sandbox.** Nothing persists between turns. Without this
  stated plainly, a model will fetch data in one program, lose the variables, and then re-type
  the numbers as literals into the next — charting what it remembered rather than the data.

Naming these takes the twelve-month question from 4 turns / 27k tokens down to **2 turns / 9k
tokens**, and lands the fan-out in the first program every time. Prompt tuning here is not
cosmetic; it's the difference between a fan-out and a for-loop.

### The sandbox is one safety layer. The tools are another.

Monty confines the *orchestration* code, but the SQL string the model writes still runs against
a real DuckDB — which can read local files, install extensions, and reach the network. So
`query` adds two controls of its own (`tools.py`):

1. It uses **DuckDB's own parser** to classify the statement and rejects anything that isn't a
   single read-only `SELECT`. Letting the parser decide beats matching keywords by hand, which
   trips over a column named `deleted_at` and misses `/**/DROP`.
2. It loads the data with trusted SQL *first*, then sets `enable_external_access=false` and
   `lock_configuration=true` — so by the time the model's SQL runs there is no filesystem, no
   network, and no way to switch either back on.

Both layers are load-bearing: `SELECT * FROM read_csv('/etc/passwd')` parses as a perfectly
valid SELECT and sails past the first check. The second is what stops it. The general rule —
**accept the narrowest input that does the job**, because the model can only ever call the
tools you register.

### Cache the data where the fan-out can't stampede it

Fan-out changes the shape of your I/O, and it's easy to under-estimate by how much. Twelve
parallel `query` tasks means twelve pods starting at the same instant — and if each one
fetches its own month from the same host, that's a burst, not a trickle. Multiply by a room
of people and the source will (rightly) throttle you.

Two things stop it:

- **`fetch_trips` is cached on the month, not the SQL** (`analysis.py`). Each month is fetched
  from the TLC exactly once, parked in blob storage, and served from there to every later
  query — any SQL, any question, any user in the project. Step 0 warms that cache serially so
  nobody's first question is a stampede.
- **The agent is told to pass `concurrency=4` to `flyte_map`**, which bounds the burst even on
  a cold cache.

The tempting shortcut — point DuckDB at the source URL and read remotely per query — looks
fine in a single test, because a `count(*)` reads only the Parquet footer and returns in
100ms. But any query touching a *column* pulls that column across the network, every time.

### The dataset has a trap in it, on purpose

Tips are only recorded for *card* payments — cash tips are logged as zero. An agent that
doesn't filter `payment_type = 1` will confidently report that cash-heavy boroughs are stingy,
which is an artifact, not a fact. It's documented in `dataset.py`, which is handed to the model
verbatim, so a well-briefed agent navigates it. Remove the warning and it walks straight in —
the cheapest available demonstration that **the tool description is the contract**.

---

## Where things live

| File | What's in it |
|---|---|
| `config.py` | The Flyte environments: image, compute, secrets. Nothing else. |
| `llm.py` | Everything about talking to the model: the `call_llm` callback, the sandbox rules injected into the prompt, and the usage tally step 4 measures with. **The only provider-specific code in the tutorial.** |
| `dataset.py` | The dataset: URLs, load SQL, and the schema description handed to the model. **Swap this file to point at different data.** |
| `tools.py` | The tools the model may call: the guarded DuckDB `query`, plus the in-process chart/metric/table renderers. |
| `analysis.py` | The cached `fetch_trips` / `fetch_zones` tasks, the durable `query` task, the agent's instructions, and the agent. Shared by steps 3–5. |
| `report.py` | HTML rendering. Not conceptually interesting, but it's why the charts show up. |

## Going further

- **More tools.** Write a function with a docstring and add it to the agent's `tools` list.
  The prompt regenerates from the signatures — there is nothing else to wire up.
- **Move a tool across the boundary.** Make a cheap tool durable (or the reverse) by switching
  it between `@env.task` and a plain function. Nothing else changes — *not even the model's
  code*.
- **Add an MCP server.** `Agent(mcp_servers=[...])` registers a server's tools alongside the
  local ones, and the model calls them from its generated code identically.
- **Swap the dataset.** Rewrite `dataset.py`; the rest follows.
