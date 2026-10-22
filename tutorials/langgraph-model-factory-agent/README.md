# The model factory, run by an agent

The notebook is the workshop: step by step, against a cluster, with the code shown as
you go. This README is the full write-up: the same steps as CLI commands, the numbers
behind every claim, and the gotchas we hit building it.

There are two agents in this tutorial.

The first is the **support agent**: the thing in production. A ticket comes in, it routes
it to a queue, it drafts a reply. It is a mock, on purpose, but it has the shape of the
real one, and today its first step, routing, is an API call per ticket. The plan is to
move this agent onto open-source models, starting with the router, because routing is a
classification problem and a small model can do it faster and cheaper than a frontier one.

The second is the **ML engineer**: a LangGraph agent that gets the request

> Replace the support agent's router with a self-hosted open-source model: at least 95%
> accuracy on our tickets, under 150 ms per ticket on a T4. Baseline up to 5 candidates,
> fine-tune up to 3 of them, spend at most 12 runs.

and operates the **model factory** to fill it: evaluates candidates on T4s in parallel,
notices none pass, fine-tunes three of them, confirms, promotes the smallest model that
meets the bar as a Union artifact, deploys it as the `ticket-router` app, and tests the
live deployment. The support agent switches over. Then new kinds of tickets arrive, the
support agent starts misrouting them, the platform notices, and the engineer adapts the
model while nobody is watching.

Flyte runs every tool the engineer calls as its own durable container and records every
model turn, so a run that dies halfway replays what it already did, and the report scores
the agent against the record of what its tools actually returned rather than its prose.

The whole thing runs three ways: on a Union cluster, which is where the parallel GPUs, the
artifacts, the deployed app and the triggers live (the notebook
`langgraph-model-factory-agent-tutorial.ipynb` walks the same steps from Colab, against a
cluster or on its own); on your own
machine with a Flyte devbox, which on an Apple Silicon Mac puts the evals, the fine-tunes
and the router app on the Mac's GPU (see [Setup](#setup));
or with `flyte run --local` and no cluster at all, in Colab on its T4 or on a laptop (see [Setup](#setup)).

| Step | You will | Needs |
|---|---|---|
| 0 | Run the support agent as it is today, routing with the API model. The "before" | a model API key |
| 1 | Drive the factory yourself as a pipeline: the data, parallel evals, a fine-tune | nothing |
| 2 | Let the ML engineer build the router, deploy it, and test the deployment | a model API key |
| 3 | Switch the support agent to the self-hosted router. The "after" | same |
| 4 | Drift: a new kind of ticket arrives. See it, then turn the factory again by hand | same |
| 5 | The loop closes itself: new tickets land, a trigger checks production, the engineer retrains | same |
| 6 | Bake off Claude vs Haiku vs whatever as the engineer's brain (optional) | same |
| 7 | Kill the engineer mid-run and watch it resume without redoing work (optional) | same |

Plus two things that need no agent at all: an artifact trigger that validates every
promotion automatically, and a human approval gate in front of the deploy.

---

## What you're building

```
   ┌──────────── the support agent (support_agent.py), in production ─────────────┐
   │   ticket ─► route ─► draft reply                                              │
   │             │                                                                 │
   │             ├─ today:    the API model classifies it     ~1 s, $ per ticket   │
   │             └─ after:    POST ticket-router /classify    ~100 ms, self-hosted │
   └───────────────────────────────────────────────────────────────────────────────┘
                                            │  "replace the router with an OSS model"
                                            ▼
                       "open weights · ≥95% accuracy · ≤150 ms · ≤12 runs"
                                            │
                                            ▼
        ┌──────────── the ML engineer's graph (graph.py) ────────────────┐
        │                                                                │
        │   START ─► think ─► tools ─► think ─► tools ─► … ─► decision ─► END
        │             │         │                            │           │
        │          ai_node   parallel_tool_node        structured output │
        │        every turn   every call in a turn        typed Decision │
        │        recorded     runs concurrently                          │
        └─────────────────────────┬──────────────────────────────────────┘
                                  │  each tool call = a Flyte child action
                                  ▼
   ┌────────────────────────── Union cluster ──────────────────────────────┐
   │                                                                        │
   │   factory-agent (CPU)   the graph runs here, retries, replays          │
   │      │                                                                 │
   │      ├─► list_candidates ──────► factory-tools (CPU)                   │
   │      │                                                                 │
   │      ├─► run_eval ×4 ──────────► factory-gpu (T4) ┐                    │
   │      │                           factory-gpu (T4) ├─ at the same time  │
   │      │                           factory-gpu (T4) ┘  (×4)              │
   │      │                                                                 │
   │      ├─► fine_tune ×3 ─► train_model ─► factory-gpu (T4) ┐             │
   │      │                   train_model ─► factory-gpu (T4) ├ at once     │
   │      │                   train_model ─► factory-gpu (T4) ┘             │
   │      │                        └──► artifact  ticket-router-qwen2.5-0.5b-ft @v
   │      │                        └──► artifact  ticket-router-qwen2.5-1.5b-ft @v
   │      │                        └──► artifact  ticket-router-modernbert-base-ft @v
   │      │                                                                 │
   │      ├─► run_eval ×3 ◄── fetched by artifact reference                 │
   │      │                                                                 │
   │      ├─► promote ─► publish_router ──► artifact  ticket-router @vN     │
   │      │              └► flyte.serve(router_app) ─────────┐              │
   │      │                                                   ▼              │
   │      │                             ┌───────────────────────────────┐   │
   │      │                             │ ticket-router app (FastAPI)   │   │
   │      │                             │ mounts ArtifactValue at /tmp/model
   │      │                             │ POST /classify {"text": …}    │   │
   │      │                             └───────────────▲───────────────┘   │
   │      ├─► test_deployment ── real HTTP calls ────────┘                  │
   │      └─► rollback ── republish an earlier version (if needed)          │
   │                                                                        │
   │   on every new ticket-router version, whoever published it:            │
   │      trigger validate-on-promote ─► validate_router (T4) ─► report     │
   │   on every new support-tickets version:                                │
   │      trigger adapt-on-new-tickets ─► the engineer, "check production"  │
   └────────────────────────────────────────────────────────────────────────┘
```

The request has a built-in tension so the agent has to think rather than follow a
script. Measured on the demo cluster (T4 for the eval, a 2-vCPU node for the "CPU"
column, which is what the serving app runs on):

| candidate | params | zero-shot | after fine-tune | p50 on T4 | p50 on CPU | passes 95% / 150 ms? |
|---|---|---|---|---|---|---|
| smollm2-360m | 360M | 13% | | 110 ms | | no |
| qwen2.5-0.5b | 0.5B | 59% | 0.3 epoch: 92% · 1 epoch: 97.5% | 68 ms | 1310 ms | after a full epoch |
| qwen2.5-1.5b | 1.5B | 79% | 1 epoch: 97.5% | 83 ms | | after a full epoch, but 3× the weights |
| smollm2-1.7b | 1.7B | 36% | | 72 ms | | no (rambles instead of answering) |
| modernbert-base | 149M | 9% (untrained head) | 3 epochs, 16 s: 99.2% | 14 ms | 232 ms | yes, and smallest |

Nothing passes zero-shot. A light fine-tune of the 0.5B is not enough for the 95% bar, so
the agent has to read a result and decide how much training to ask for. And the one model
that is not a chat model at all, the encoder, turns out to be the best router by every
measure once it is trained. The agent has to get there from numbers it produces itself.

For scale, the models the support agent could route with today, zero-shot on the same
120 tickets (p50 here is a network call from the cluster, not a T4):

| router | zero-shot | p50 | per 1,000 routes | where it runs |
|---|---|---|---|---|
| Claude Opus 5 | 98.3% | 1.7 s | \$3.81 | API |
| GPT-4.1 | 97.5% | 634 ms | \$0.28 | API |
| Claude Haiku 4.5 | 95.8% | 606 ms | \$0.89 | API |
| Qwen3-8B (vLLM) | 94.2% | 386 ms | ~\$0.20 | one L40s, ours |
| modernbert-base, fine-tuned | 99.2% | 14 ms on a T4, 35 to 270 ms on the app's CPU pod | < \$0.01 | 149M params, ours |

The big API models clear the bar without training and the trained encoder beats all of
them at a hundredth of the latency. Cost is routing only, at list prices, from the tokens
each call used (the Claude calls carry the tool-use schema, GPT does not; Qwen is one
L40s at one request at a time). At 100k tickets a month that is \$381 on Opus, \$89 on
Haiku, and under a dollar of CPU time on the encoder. Qwen3-8B, an open model with 50× the encoder's
parameters, does not clear it zero-shot.

### What a run looks like

This is the shape of a cold run, with the tool calls the agent actually made:

```
 0:00  think     "list the candidates, then baseline the four chat models"
 0:02  tools     list_candidates                                       CPU     2 s
 0:05  tools     run_eval smollm2-360m   ┐
                 run_eval qwen2.5-0.5b   ├── four T4 pods at once      ~70 s
                 run_eval qwen2.5-1.5b   │   (the encoder is skipped:
                 run_eval smollm2-1.7b   ┘    an untrained head is not worth a run)
 1:20  think     "none pass; fine-tune the two Qwens for an epoch and the encoder for three"
 1:22  tools     fine_tune qwen2.5-0.5b  1ep ┐
                 fine_tune qwen2.5-1.5b  1ep ├── three T4 pods at once ~90 s
                 fine_tune modernbert    3ep ┘   three artifacts registered
 2:55  think     "confirm all three on the full set"
 2:56  tools     run_eval …0.5b-ft ┐
                 run_eval …1.5b-ft ├── three T4 pods at once            ~40 s
                 run_eval …bert-ft ┘
 3:40  think     "all three pass; the encoder is smallest and fastest, promote it"
 3:41  tools     promote ─► ticket-router@vN published, app deployed   ~60 s
 4:45  tools     test_deployment ─► 12/12 on the live app              ~30 s
 5:20  decision  typed Decision, checked against the request
```

Eight model turns, thirteen tool calls, about 13k tokens. The evals run every time (a
minute each, in parallel), so every run shows its work; the fine-tunes are cached, so the
second person in the room trains nothing.

In the Flyte UI the run graph shows the fan-out, and every action is named after what
it did: `run_eval · qwen2.5-0.5b` four across, then `fine_tune · modernbert-base · 3ep`
with its `train_model` child, then the rechecks, then `promote` with `publish_router`
under it, then `test_deployment`. Each eval and training action has its own report (numbers, a confusion matrix,
per-category accuracy, sample mistakes). The parent task's report is the decision page:
what was promoted, pass/fail chips against the request, everything the agent measured in
one table, the sequence of calls, and the rationale, plus an **Agent** tab with the turn
by turn timeline. The `ticket-router` artifact's **Lineage** tab leads from the app back
through the promotion to the fine-tune that produced the weights.

---

## Setup

Three ways to run it; pick one. The plain `flyte run` commands in the steps are for a devbox
or a cluster; the `--local` lines are for Colab or a laptop with neither. Each path is spelled
out in its own section below, folded.

- **A Flyte devbox on your machine (recommended).** Docker Desktop running, Python 3.10
  or newer, [`uv`](https://docs.astral.sh/uv/), `pip install -r requirements.txt`, then
  `flyte start devbox --apple-silicon` on an Apple Silicon Mac (`flyte start devbox`
  elsewhere: an NVIDIA GPU, or the CPU with the small models). Steps 0 to 4, the deployed
  router app, the bake-off, the approval gate and the crash (step 7) all run, the GPU work
  on your own GPU. Only step 5 and the validate-on-promote trigger stay out: artifact
  triggers are a Union feature.
- **The notebook on Google Colab (no GPU needed).** A Google account and a model API key;
  the [Colab badge](https://colab.research.google.com/github/unionai/workshops/blob/factory/tutorials/langgraph-model-factory-agent/langgraph-model-factory-agent-tutorial.ipynb)
  opens it on a free T4 with nothing to install. Slightly pared down: steps 0 to 4 and the
  bake-off, with a folder standing in for the deployed app; the triggers (step 5) and
  kill-and-resume (step 7) need a Flyte deployment.
- **The workshop's Union cluster (`tryv2`).** Any laptop, with Python 3.10 or newer, `uv`
  and `pip install -r requirements.txt`; then one `flyte create config` command (below) and
  a browser sign-in. The cluster's T4s do the work and every step runs, including step 5's
  triggers: the full workshop. The notebook does the same when its `LOCAL` switch is `False`.

The devbox and Colab paths also need a model API key (Anthropic by default, or OpenAI)
with a few dollars of credit; the workshop cluster already holds one. The devbox and
`--local` paths download about 5 GB of models on the first run (no Hugging Face account).

**Working with a coding agent?** Two skill sets teach Claude Code, Codex and friends the
Flyte 2 SDK and Union, so they stop guessing at the API while you follow along or extend
the tutorial:

- [flyteorg/flyte-agent-plugins](https://github.com/flyteorg/flyte-agent-plugins): 21 skills
  for building, deploying and running Flyte workflows, plus two MCP servers (Flyte docs
  search, and acting on a cluster). `uvx flyte-skills install`, or in Claude Code
  `/plugin marketplace add flyteorg/flyte-agent-plugins` then `/plugin install flyte@flyte-agent-plugins`.
- [unionai-oss/union-skills](https://github.com/unionai-oss/union-skills): skills for
  deploying and operating Union (connecting a cluster, running workflows, debugging one
  that won't come up). `uvx union-skills install`, or `/plugin marketplace add unionai-oss/union-skills`
  then `/plugin install union@union-skills`.

What you need on your machine, by path:

| | Union cluster | Devbox (your machine) | No cluster (`--local`) |
|---|---|---|---|
| Python | 3.10 or newer (3.11 used here) | same | same; Colab's is fine |
| [`uv`](https://docs.astral.sh/uv/) | yes, for the venv | yes: it also builds the per-task Python envs on a Mac | yes, for the venv (Colab: `pip`) |
| Docker | no | yes: Docker Desktop (or Docker Engine), running | no |
| `flyte` | 2.12 or newer, from `requirements.txt` | same | same |
| Python packages | `requirements.txt` | `requirements.txt` | `requirements-local.txt` (adds torch, transformers, peft, trl) |
| GPU | the cluster's T4s | the Mac's GPU (Apple Silicon), else CPU | Colab's T4, a Mac's GPU, else CPU |
| Model API key | as a Union secret | as a devbox secret | in `.env` or the environment |
| Disk | | ~10 GB: the devbox image, the task images, the model weights | ~5 GB of models in `~/.cache/huggingface` |

In every case, start here:

```bash
git clone https://github.com/unionai/workshops
cd workshops/tutorials/langgraph-model-factory-agent
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt            # the no-cluster path: requirements-local.txt instead
flyte --version                               # 2.12 or newer
```

Step 1 needs nothing else. For the agents, put a key in `.env`; `config.py` loads it:

```bash
ANTHROPIC_API_KEY=sk-ant-...
# or, to drive the agent with OpenAI instead:
# OPENAI_API_KEY=sk-...
# AGENT_MODEL=openai:gpt-4.1
```

<details id="setup-devbox">
<summary><b>On your own machine: a Flyte devbox (recommended)</b></summary>

A devbox is a Flyte control plane in one container on your machine (Linux, Windows with
WSL2, or a Mac). On an Apple Silicon Mac, start it with `--apple-silicon` and every task
that asks for a GPU runs natively on the Mac, PyTorch on `mps`, while the devbox keeps
the run graph, the cache, the retries, the logs and the reports:

```bash
flyte start devbox --apple-silicon        # a Mac; elsewhere: flyte start devbox
flyte create config --devbox              # writes .flyte/config.yaml pointing at localhost:30080
# keeping a cluster config too? write this one elsewhere and name it instead:
#   flyte create config --devbox -o .flyte/config.devbox.yaml && export FLYTECTL_CONFIG=.flyte/config.devbox.yaml
flyte create secret anthropic-api-key --value sk-ant-...
echo "ANTHROPIC_API_KEY_SECRET_NAME=anthropic-api-key" >> .env
flyte run model_factory.py model_factory --fine_tune_too      # same commands as the cluster
```

Needs Docker Desktop running, `uv` on the PATH (the Apple Silicon bridge builds a Python
environment per task image with it, about 3 minutes the first time, cached under
`~/.flyte/apple-silicon/envs/`), and `flyte` 2.12 or later. The one knob is `ACCELERATOR`
(`T4` | `L40s` | `apple` | `cpu`): it picks the pod shape for `run_eval` and `train_model`
and the torch device. On a Mac whose Flyte config (`.flyte/config.yaml`, or the file
`FLYTECTL_CONFIG` names) points at a devbox it defaults to `apple`; a config that points
at a cluster keeps the T4 default, and then the eval pods wait forever for an NVIDIA GPU
the devbox does not have; a devbox without a GPU wants `ACCELERATOR=cpu` in `.env`, and then the
encoder is the candidate to fine-tune (seconds, not the quarter hour a chat model takes
on a CPU). Then every step but 5 runs as written, with the T4 swapped for your GPU: steps
0 to 4, the bake-off, the approval gate and the crash. Measured on an M5 Max, same code,
same commands:

| | Union cluster (T4) | Apple Silicon devbox (M5 Max) |
|---|---|---|
| qwen2.5-0.5b zero-shot, p50 | 68 ms | 27 ms |
| one LoRA epoch on the 0.5B | 42 s | 26 s, bf16 |
| modernbert-base, 3 epochs, fine-tuned | 99.2%, 14 ms | 99.2%, 6 ms |
| the support agent on the router (step 3) | 100%, 170 ms via a CPU pod | 100%, 15 ms, the app on `mps` |
| drift and the fix by hand (step 4) | 88.9% on v2, then 100% | 88.1% on v2, then 100%; the engineer retrains the encoder on v2 and redeploys |
| the bake-off (step 6), Haiku vs GPT-4.1 | 10 and 7 runs | both meet the request: Haiku 11 runs, 54k tokens; GPT-4.1 8 runs, 49k tokens |
| the engineer on an open model (step 6), Qwen3-8B | on an L40s vLLM app: met, 12 runs, 33k tokens, 173 s | served on the Mac itself at ~22 tokens/s: met, 10 to 12 runs, 35k tokens, 5 to 7 min; promotes a fine-tuned smollm2-1.7b at 99 to 100%, 50 ms |
| three brains at once (step 6), Haiku, GPT-4.1, Qwen3-8B | | all three meet it in one run: Haiku 8 runs, 52k tokens, 133 s; GPT-4.1 7 runs, 27k tokens, 108 s (both promote the encoder at 99.2%, 6 ms); Qwen 10 runs, 35k tokens, 295 s (the 1.7B at 100%, 50 ms) |
| crash and resume (step 7) | fails once, succeeds on the retry | the same: two attempts, nothing retrained, promoted and live-tested 12/12 at 8 ms |
| the agent, step 2, fine-tunes cached | 45 s warm, 4 to 5 min cold | 2 min 47 s warm: evals rerun, promote, deploy, live test (9 runs) |
| first run | ~6 min image build | ~5 min: a linux/arm64 image plus a Mac Python env with torch |

Three things are different on a devbox, and the code notices each one at run time:

- **No artifact registry.** That is a Union feature. `tools.registry()` asks once; when
  the answer is no, `fine_tune` hands back the storage path of its output directory as
  the model reference, `promote` mounts that directory into the app directly, and
  `production_status` asks the live app what it is serving. Same files, no versions to
  roll back through (rollback says so, and promote the earlier reference again), no
  lineage tab, and no artifact triggers: step 5 and the validate-on-promote trigger need
  Union. `FACTORY_ARTIFACTS=0` forces this mode anywhere.
- **One GPU.** `FACTORY_MAX_PARALLEL_GPU` defaults to 1 on a Mac: the engineer still asks
  for four evals in one turn and they take turns. Raise it on a Mac with memory to spare.
- **Promote runs on the Mac.** A devbox signs upload URLs for `localhost`, has no remote
  image builder, and reports an app as plain ACTIVE with no revision history. So in
  `apple` mode the tool tasks run on the host alongside the GPU ones, and `promote`
  builds the app image with the local builder and takes the app's `/` as proof of the
  rollout. The agent does not know any of this; its tools return the same text.

Memory, not the pod request, is the limit: the pod only proxies a process on the Mac.
The 0.5B LoRA needs about 8 GB of unified memory; the 1.5B several times that. On a
16 GB Mac, give step 2 `--candidates_to_screen 2 --max_fine_tunes 1`, or let the engineer
screen everything and watch the encoder win on a few hundred megabytes. The notebook
works against a devbox as well: skip its "connect to the cluster" cell.

</details>

<details id="setup-colab">
<summary><b>No devbox and no cluster: the notebook on Colab, or <code>flyte run --local</code></b></summary>

Every step's `--local` line runs the task in your own process: Colab on its T4, or a
laptop (Apple Silicon uses `mps`, anything else the CPU). No cluster, no devbox, no
Docker, and still every tool call a recorded child action. It needs the training stack
in the environment, which the cluster paths get from the image:

```bash
uv pip install -r requirements-local.txt       # requirements.txt plus torch, transformers, peft, trl
# (it also pins torchao>=0.16: Colab preinstalls 0.10, which peft's LoRA code refuses)
flyte run --local model_factory.py model_factory --n 20 --models '["smollm2-360m"]'
flyte run --local ml_engineer.py ml_engineer_agent --candidates_to_screen 2 --max_fine_tunes 1
flyte run --local support_agent.py agent_handle_tickets --n 10 --router oss
```

The notebook runs this way by default (`LOCAL = True` in its setup cell; set it to
`False` for a cluster): the
[Colab badge](https://colab.research.google.com/github/unionai/workshops/blob/factory/tutorials/langgraph-model-factory-agent/langgraph-model-factory-agent-tutorial.ipynb)
opens it on a T4, where steps 0 to 4 and the bake-off run end to end with nothing but a
model API key.

What stands in for the platform, in `tools.py`, decided at run time by `flyte.ctx()`:

- **A folder is the registry and the app.** `promote` copies the winner into
  `.factory/ticket-router/<version>/` next to the code and points `current.json` at it
  (`FACTORY_LOCAL_DIR` moves it). `production_status` reads that, `test_deployment` loads
  the current version in-process and measures it, and `rollback` makes an earlier version
  current again. The support agent's `--router oss` loads the same directory through the
  `Router` class that the app itself wraps, so steps 3 and 4 are the same code with no HTTP.
- **One process, one GPU.** `FACTORY_MAX_PARALLEL_GPU` defaults to 1 here as on a Mac:
  the engineer still asks for several evals in one turn and they take turns.
- **The reports, in the notebook.** Every task that writes a report on the cluster writes
  the same `report.html` locally, into the run's metadata folder. In the notebook, `run()`
  renders the step's own report under the cell, and `show_reports(all=True)` renders every
  report of that run: each eval's table and confusion matrix, each fine-tune's loss curve,
  the engineer's timeline and decision (`utils/report_viewer.py`). A cached fine-tune did
  not run again, so it has no new report.
- **The run graph, in the terminal.** Add `--tui` to any `--local` command and the SDK
  opens its terminal UI on the run as it goes: the actions, their logs and outputs, the
  same tree the console shows for a cluster run (`flyte[tui]` is in `requirements.txt`).
- **Not here.** Step 5's triggers and the validate-on-promote trigger need artifacts, and
  step 7's retry needs the platform.

Measured on an M5 Max with `--local` (Colab's T4 is in the same range): step 2 with
`--candidates_to_screen 2 --max_fine_tunes 1` took 2 min 39 s (the engineer screened five
candidates anyway, fine-tuned the 1.5B two epochs to 97.5% at 43 ms, promoted it, and the
in-process test passed 12/12); step 3 on that router 100% at 52 ms; step 4 86.5% on v2 with
14 tickets in an unknown queue; the by-hand fix 83 s (retrained on v2 to 97.5%, 3 runs), after which the support agent routes
the 126 v2 tickets at 97.6% with none in an unknown queue. The notebook itself, every
cell in local mode, ran end to end in 5 minutes on the same machine (that time the engineer
promoted the encoder: 97.5% at 6 ms, then 99.2% on v2 after the by-hand fix).

On a laptop CPU, keep to the small models (`--models '["smollm2-360m"]'`, or let the
engineer find the encoder) and `--n 20`; a chat-model fine-tune on a CPU takes a quarter
of an hour.

</details>

<details id="setup-cluster">
<summary><b>On the workshop cluster (<code>tryv2</code>), or your own Union cluster</b></summary>

The workshop runs in the `factorysea` project on `tryv2.hosted.unionai.cloud`. From
a laptop, the whole setup is the config file:

```bash
flyte create config --endpoint tryv2.hosted.unionai.cloud --project factorysea --domain development --builder remote
flyte get secret                    # ANTHROPIC_API_KEY is already there; nothing to create
flyte run model_factory.py model_factory --fine_tune_too       # the first run signs you in
```

The config lands in `.flyte/config.yaml` next to the code, so every `flyte run` and the
notebook's `run()` go to the cluster from then on (the `--local` lines keep working
alongside). The first call opens your browser to sign in with the account the host gave
you. Anywhere without a browser to hand off to (a hosted notebook, a remote shell) add
`--auth-type headless`: the first call then prints a login URL and a code to paste back.
Without it the browser flow fails silently and the first upload dies with an empty
`ConnectError`. The notebook already does all of this when `LOCAL = False`: its connect
cell writes the same config with the headless flag.

On your own cluster, point the config at your endpoint and project and create the key
yourself (`--value` matters: without it the command waits for input):

```bash
flyte create config --endpoint <your-endpoint> --project flytesnacks --domain development --builder remote
flyte create secret ANTHROPIC_API_KEY --value sk-ant-...
```

On a shared cluster such as `tryv2`, set `FACTORY_TAG=<your handle>` in `.env`. It namespaces what has
to be yours: the serving app (`ticket-router-<tag>`), the production artifacts, and the
environments that carry triggers. The GPU training task stays shared, so each fine-tune
is trained once for the whole room. The API key stays shared: on `tryv2` the project's
`ANTHROPIC_API_KEY` is the one secret everyone uses, so leave `FLYTE_SECRET_PREFIX`
unset. The prefix is for a cluster where each person brings their own key
(`FLYTE_SECRET_PREFIX=SAGE_` makes the code ask for `SAGE_ANTHROPIC_API_KEY`); a secret
with another name entirely is `OPENAI_API_KEY_SECRET_NAME=my-openai-key`.

The first cluster run builds two images. The one with the training stack takes about six
minutes, once.

</details>

---

## Step 0: the support agent, as it is today

```bash
flyte run --local support_agent.py agent_handle_tickets --n 10 --router llm    # Colab or a laptop with no devbox: 10 tickets
flyte run support_agent.py agent_handle_tickets --router llm                   # devbox or cluster: 30 tickets
```

**What you'll see.** Thirty held-out tickets go through the support agent: the router picks
a queue, then the queue's drafting agent (the same model, prompted as that team) writes a
two-line reply for the team to send. The report has
the numbers that matter for the request: routing accuracy, p50 routing latency, tokens,
and cost per 1,000 tickets, plus where the tickets landed and every draft.

The router is accurate, slow, and priced per ticket. We ran the same thirty tickets
through three API models; the cost column is routing and replies together, at list price:

| Support agent model | Routing accuracy | Route p50 | Per 1,000 tickets |
|---|---|---|---|
| Claude Opus 5 | 96.7% | 2.28 s | \$10.28 |
| Claude Haiku 4.5 (the default) | 96.7% | 596 ms | \$1.18 |
| GPT-4.1 | 93.3% | 594 ms | \$0.64 |

Opus thinks before it routes, which is why it is four times slower than the others at
the same accuracy. That is the "before." Pick the model with `--model`, or `AGENT_MODEL`.

**What just happened.** `support_agent.py` is a two-node LangGraph graph, `route → draft`.
Each ticket is a `flyte.group` in the run graph holding two `flyte.trace` steps, route and
draft, whose arguments are the ticket itself: the graph shows every ticket's text going in
and the queue or reply coming out, thirty small agents side by side, and a batch that dies
halfway replays what it already did.
`--router llm` classifies with the API model; `--router oss` calls the `ticket-router`
app instead, which does not exist yet. Building it is the request.

---

## Step 1: the factory as a pipeline

This is the factory with you at the controls: you pick the candidates and the epochs, the same tools do the work, and you look at what comes back. In step 2 the ML engineer takes over the controls. In the Flyte UI the evals appear side by side in a group called `baseline-evals`, and the fine-tune plus its recheck in `fine-tune-and-recheck`.

```bash
flyte run --local model_factory.py model_factory --n 20 --models '["smollm2-360m"]'   # Colab or a laptop with no devbox: one quick eval
flyte run model_factory.py model_factory --fine_tune_too                           # devbox or cluster: the full matrix
```

**What you'll see.** The task report has a **Data** tab with the synthetic tickets (800
for training, 120 held out, eight categories, generated from templates with a fixed seed,
so every container and every attendee gets the same split with nothing to download) and a
main tab with the eval matrix scored against the request's constraints. In the terminal,
the tool transcript:

```
$ run_eval(model='smollm2-360m', n=120)
smollm2-360m: accuracy 14.2% on 120 tickets; latency p50 86 ms, p95 89 ms per ticket (cuda, batch size 1); model load 10s.
per category: billing 100%, refund 0%, shipping 0%, ...           ← it answers "Billing" to everything
$ run_eval(model='qwen2.5-0.5b', n=120)
qwen2.5-0.5b: accuracy 55.8% on 120 tickets; latency p50 67 ms, p95 70 ms ...
$ run_eval(model='qwen2.5-1.5b', n=120)
qwen2.5-1.5b: accuracy 79.2% on 120 tickets; latency p50 80 ms, p95 85 ms ...
$ fine_tune(model='qwen2.5-0.5b')
fine-tuned qwen2.5-0.5b with LoRA for 100 steps (1 epoch(s)) in 42s on cuda; loss 2.67 -> 0.17.
new model reference: artifact:ticket-router-qwen2.5-0.5b-ft@1ep-1789605305
$ run_eval(model='artifact:ticket-router-qwen2.5-0.5b-ft@1ep-1789605305', n=120)
artifact:...: accuracy 97.5% on 120 tickets; latency p50 68 ms, p95 69 ms ...
```

**What just happened.** `model_factory.py` called the same tool objects the agent will use,
with `asyncio.gather` around the evals. Each `.ainvoke()` became a Flyte child
action: on the cluster, a T4 container. `fine_tune` launched `train_model` on a T4, which
saved the merged weights and registered them as a Union artifact with a model card; the
last `run_eval` fetched that artifact by reference. Open the run in the Flyte UI and you
can see the evals side by side.

**Look at** the numbers before moving on. They are the ground the agent will stand on.
The evals are not cached, so the engineer measures again in step 2 and you see it happen;
the fine-tune is, so when the engineer asks for the same one it gets the weights back in
seconds, and so does everyone else in the project.

---

## Step 2: let the ML engineer build the router

```bash
flyte run --local ml_engineer.py ml_engineer_agent --candidates_to_screen 2 --max_fine_tunes 1   # Colab or a laptop with no devbox: keep it small
flyte run ml_engineer.py ml_engineer_agent                                                       # devbox or cluster
```

**What you'll see.** The terminal prints the run URL and, at the end, the decision as
JSON. The report's **Agent** tab is the timeline: a `think` row, then the tool calls it
requested, their results, the next `think`, and so on. The decision tab shows the promoted
model, the numbers the agent measured, its evidence and rationale, and a row of checks:

```
Against the request   ✓ accuracy ≥ 95%   ✓ p50 ≤ 150 ms   ✓ runs ≤ 12   ✓ promoted   ✓ reported honestly
tools used: list_candidates, run_eval, run_eval, run_eval, fine_tune, fine_tune, run_eval, run_eval, promote
```

The last two checks matter. Tool errors come back to the model as text, and a model will
happily write up a failed promotion as a success. The graph records which tools failed and
the score compares the model's claim with the record.

**What just happened.** `graph.py` is a small LangGraph `StateGraph`. Two of its parts come
from `flyteplugins.agents.langgraph`: `ai_node` records every model turn with
`flyte.trace`, and `@tool` turns each `@env.task` into a LangChain tool that runs as a
durable child action. The `tools` node is ours, a variant of the plugin's that launches
every call in a turn concurrently. The `decision` node is a structured-output call that
returns a typed `Decision`.

On the cluster, `promote` did two more things: it published the winner as a new version
of the `ticket-router` artifact (decision and numbers on the card, searchable attributes)
and deployed `router_app.py`, a FastAPI app whose model parameter is
`ArtifactValue(name="ticket-router")`, resolved and pinned at deploy time.

**The agent tests its own deployment.** After `promote`, the agent calls
`test_deployment`, which finds the live app, waits for it to come up, and sends it a dozen
held-out tickets over HTTPS. If that fails, `rollback` republishes an earlier version's
weights as the newest `ticket-router` version and redeploys. Production is always the
latest version of the artifact, by construction, and a rollback is visible in the
artifact's history rather than a quiet re-pointing of the app. In the run above it passed 12/12, on the app's CPU pod at about 100 ms a ticket.

**One run, unplanned.** In one of our runs the OpenAI API returned a 404 in the middle of
the write-up, five minutes in, after four evals, three fine-tunes and a deploy. Flyte
retried the task. Every model turn replayed from its record, every child action was
already done, and attempt two finished in 13 seconds with nothing redone. That is step 7,
happening by accident.

**Watch it be wrong.** With a 95% bar, an earlier version of this agent promoted a model at
94.2% and wrote that it "exceeds 95% by rounding conventions". The report caught it (the
accuracy chip fails and "reported honestly" fails, because the decision claims the
constraints were met), and the prompt now says the bar is the bar. Keep that report; it is
the best argument for scoring an agent against the record rather than reading its prose.

**Try the result.** The app's endpoint is on its page in the Flyte UI, or:

```bash
flyte get app ticket-router
curl -X POST "$ENDPOINT/classify" -H 'content-type: application/json' \
     -d '{"text": "I was charged twice this month, please refund one"}'
# {"label":"refund","raw":"refund","latency_ms":1075.8}
```

That is a 2-CPU pod, hence a second per ticket; the T4 numbers are the eval's. The app
scales to zero after 30 idle minutes.

**About that second attendee.** Step 2's request says "treat this as a fresh build, ignore
what is deployed", so thirty people running it get thirty full investigations rather than
twenty-nine "production already meets the bar" in 35 seconds. Steps 4 and 5 say the
opposite, on purpose.

**Change the problem.** The request is the task's inputs. `--min_accuracy 0.95
--max_latency_ms 75` rules out the 1.5B on latency and makes the agent work harder;
`--budget 4` forces it to skip a baseline. Every combination is a different investigation.

---

## Step 3: switch the support agent to the self-hosted router

```bash
flyte run support_agent.py agent_handle_tickets --router oss
```

**What you'll see.** The same thirty tickets, the same draft replies, but routing is now
an HTTP call to the app the engineer deployed. The route p50 includes the HTTP round trip
to a CPU pod; what is left of the cost is the reply drafts. The cost column counts API
tokens only: the router pod is not metered, and at 2 CPUs, a few cents an hour on any
cloud, it adds about a tenth of a cent per thousand tickets:

| Support agent model | Routing accuracy | Route p50 | Per 1,000 tickets |
|---|---|---|---|
| Claude Opus 5 | 96.7% → 100% | 2.28 s → 128 ms | \$10.28 → \$6.47 |
| Claude Haiku 4.5 | 96.7% → 100% | 596 ms → 170 ms | \$1.18 → \$0.30 |
| GPT-4.1 | 93.3% → 96.7% | 594 ms → 200 ms | \$0.64 → \$0.34 |

Put this report next to step 0's.

**What just happened.** Nothing in the support agent changed except a flag. The router
app is a FastAPI service that mounts the promoted artifact; `--router oss` posts the
ticket to `/classify` and reads back the queue and the model's confidence. That is the
first piece of this agent running on an open model.

---

## Step 4: drift, and the fix by hand

```bash
flyte run support_agent.py agent_handle_tickets --router oss --dataset v2 --n 126      # see it
flyte run drift.py day_two                                                  # fix it
flyte run support_agent.py agent_handle_tickets --router oss --dataset v2 --n 126 --labels_from v2   # see it fixed
```

The support team adds a ninth category, `data_request`, for privacy and data-subject
requests that used to be filed under `other`. Our tickets are now dataset v2. The router in
production was trained on v1 and has never seen the label.

**What you'll see.** First the support agent, routing the v2 tickets with the v1 router:
on our cluster 87.3% on the full split, with 9 of the 14 GDPR tickets filed under
`account_access` (not `other`: "delete my account and all my data" looks like an account
ticket to a model that knows eight queues). The report flags them as tickets in a queue
the router does not know about. That is the drift, in the agent that suffers it. Then the
engineer: it starts with `production_status` (which artifact version
is live), evaluates it on v2 (it cannot say `data_request`, so it lands around 88%),
fine-tunes on v2, confirms, promotes, tests the deployment. Then the support agent again:
97.2%, and the GDPR tickets route to `data_request`. The `ticket-router` artifact gains a version; its Versions tab is now
the router's history, and Lineage leads from the app through each promotion to the
training run that produced it. In our run: 5 runs spent, the v2 encoder at 99.2%,
deployment test 100% on v2 tickets.

**What just happened.** Same graph, same tools, a different request: `Request` carries a
`situation` (what changed, what is in production, and "retrain the base that is in
production first, it serves on a CPU pod") and `dataset="v2"`. That last clause is there
because the first time we ran this, the engineer fine-tuned a chat model that cleared the
T4 latency bar and then served at 1.1 s per ticket on the CPU app. The bar is measured
where the request says; the request has to say where production runs. Every eval, fine-tune
and promote takes the dataset version; the serving app reads the label set from the
artifact's metadata, so the endpoint switches to nine labels when the v2 model lands.

### Detecting drift, honestly

The report can say "that is drift" because the batch is labeled. Production is not, so
the question is what you would watch instead. The usual answer is the router's own
confidence, and the app returns one for the encoder (`confidence` on `/classify`), so the
support agent reports it: mean confidence and the share of routes under 90%. Here is what
it showed on the deployed v1 encoder, measured on the two full test splits:

| batch | accuracy | mean confidence | routes under 90% | misroutes at 90%+ confidence |
|---|---|---|---|---|
| v1 tickets (in distribution) | 98.3% | 97.2% | 7.5% | 0 of 2 |
| v2 tickets (new category) | 87.3% | 95.8% | 11.1% | 7 of 16 |

Accuracy fell eleven points; confidence fell one and a half. Half of the GDPR tickets were
misrouted with 90%+ confidence (a third with another fine-tune of the same encoder on tryv2:
88.9% accuracy, 97.1% mean confidence, 5 of 14 misroutes at 90%+). A softmax over eight labels has no way to say "none of
these"; it does not know what it does not know. So confidence is a weak alarm for a new
category, and we chose not to build the loop on it.

What catches this kind of drift is labels, and in production those arrive on their own
with a lag: every ticket the support team re-routes by hand is a label, and the re-route
rate is accuracy monitoring without a labeling project. That is the manual step this
tutorial keeps: someone notices tickets in the wrong queue, adds the category, labels the
batch, publishes it (step 5). Everything after that is automatic. If you want a label-free
alarm as well, two signals beat confidence. Novelty: the distance of a ticket's encoder
embedding from the training tickets, which a new kind of ticket raises whatever the softmax
says. And a judge: the API model that routed tickets in step 0 can shadow the router on a
sample (one ticket in fifty, say) with one extra option the router does not have, "none of
these queues". Disagreement is an accuracy estimate with no labels, and the extra option is
exactly what the softmax cannot express. It spends a little of the API budget the router
saved, on a sample, so the arithmetic still favors the router. Either signal can publish the
`support-tickets` artifact itself and start the same loop.

**Try it with a ticket the old model could not route:**

```bash
curl -X POST "$ENDPOINT/classify" -H 'content-type: application/json' \
     -d '{"text": "Under GDPR I would like a copy of all the data you hold on me"}'
# before day two: {"label":"cancellation", ...}    after: {"label":"data_request", ...}
```

## Step 5: the loop closes itself

```bash
flyte deploy adaptive_loop.py agent_env               # registers the trigger, once
flyte run adaptive_loop.py publish_tickets --version v2
```

Step 4 had you run the factory when the data changed. Here the only human act is
publishing the labeled tickets; the rest nobody does:

```
publish_tickets ─► artifact support-tickets @v2
                          │  trigger adapt-on-new-tickets
                          ▼
adapt ─► the engineer agent, told: "new tickets landed, check production first"
      ─► production_status ─► run_eval(that version, v2)         one T4
      ─► meets the bar?  ─► keeps it, promotes nothing, reports its numbers
      ─► below the bar?  ─► fine_tune on v2 ─► promote ─► artifact ticket-router @vM
                                                            │  trigger validate-on-promote
                                                            ▼
                                                      validate_router ─► report
```

**What you'll see.** Publishing the tickets is one short run. A minute later a run of
`adapt` appears that you did not start. It is the same engineer task as steps 2 and 4,
with the same Agent tab and the same decision page, just started by the platform with a
different situation. If the model in production still meets the bar on the new tickets,
the decision page says **Kept in production** with its measured numbers and nothing else
happens. If not, the agent retrains, promotes and tests the deployment, and a run of
`validate_router` appears that nobody started either. The `support-tickets` artifact is
the history of the data; the `ticket-router` artifact is the history of the model; each
version of the second points back at the run that made it, and that run at the version
of the first that caused it.

**What just happened.** Two artifact triggers chained through an agent. The trigger task
knows nothing about models; it reads the manifest of what landed and hands the situation
to the engineer graph. Keeping production is a first-class outcome of the decision
(`action: kept_production`), scored like any other: the numbers it reports have to be the
numbers it measured.

In our run: `adapt` asked `production_status` which version was live, measured it at
87.5% on v2, spent 6 runs (baselines, two fine-tunes in parallel, confirmations), promoted
a v2 encoder at 99.2%, the deployment test passed 12/12, and `validate_router` fired and
passed. Zero clicks after the publish. Published the same tickets again with the new model
in production, and `adapt` measured it, kept it, and stopped after one run.

This is the adaptive cycle the tutorial is named for: data changes, the platform notices,
the agent decides whether the model has to change, the platform validates and serves
whatever it decides, and the whole thing is recorded as artifacts and runs you can walk
back through.

## Step 6: bake off the engineer's brain (optional)

```bash
flyte run --local bakeoff.py bakeoff --models '["anthropic:claude-opus-5", "anthropic:claude-haiku-4-5"]'
flyte run bakeoff.py bakeoff --models '["anthropic:claude-opus-5", "anthropic:claude-haiku-4-5", "vllm:qwen3-8b"]' --trials 2
```

**What you'll see.** One child action per (model, trial), in parallel. The report scores
each: met the constraints, stayed in budget, runs spent, tokens, wall clock. Each child
action has its own decision page and Agent tab, so you can read how each one argued for
the encoder over the 1.5B.

**What just happened.** Same request, same tools, different `model` string. The
fine-tunes are cached, so after the first agent the others train nothing, and the score
that matters is runs spent and tokens rather than wall clock. Promotion publishes
artifacts but does not deploy here (`FACTORY_DEPLOY=0`). On our cluster Claude Opus 5 met
the request in 10 runs and 63k tokens; Claude Haiku 4.5 in 10 runs and 53k tokens;
GPT-4.1 in 7 runs and 23k tokens; GPT-4.1-mini in 11 runs and 29k tokens. Same tools,
different amount of flailing. On an Apple Silicon devbox the same bake-off (Haiku and GPT-4.1,
the fine-tunes cached) came out the same way: both met the request, GPT-4.1 in 8 runs and
49k tokens, Haiku in 11 runs and 54k tokens.

**The engineer on an open model.** `vllm:qwen3-8b` is the same agent driven by Qwen3-8B,
served by vLLM on a Union app with one L40s. Deploy it once (it prefetches the weights
into object storage, then streams them to the GPU on every cold start, and scales to zero
when idle):

```bash
flyte create secret VLLM_API_KEY --value <any string>     # vLLM's own --api-key
python serve_model.py                                    # prints the endpoint
```

Then put `VLLM_BASE_URL=https://<app>/v1` in `.env` and add `vllm` to `FACTORY_PROVIDERS`.
On a shared cluster, attendees point at the instructor's app and its key by name
(`VLLM_API_KEY_SECRET_NAME=<prefix>_VLLM_API_KEY`); nobody needs their own L40s. The app
scales to zero after fifteen idle minutes and takes two or three minutes to stream the
weights back, so hit `/v1/models` once before a room does. If the
open model clears the bar, the ML engineer is off the API too, and it is a reasonable
place to start the whole workshop from. On our
cluster Qwen3-8B met the request too: it fine-tuned three candidates, promoted the 1.5B at
100% and 84 ms, and passed the deployment test, in 12 runs, 33k tokens and 173 seconds. It
spent the whole budget and picked a model ten times the size of the encoder that also
passed, which is exactly the kind of judgment the report lets you compare.

**The engineer on an open model, on your Mac.** On an Apple Silicon devbox the same thing
runs on the Mac's GPU with no vLLM and no L40s:

```bash
flyte create secret VLLM_API_KEY --value anything     # the agent's environment asks for one
python serve_model_apple.py                           # prefetch Qwen3-8B, serve it on mps, print the endpoint
# VLLM_PREFETCH_RUN=<run name> python serve_model_apple.py                          # redeploy, same weights
# VLLM_HF_REPO=Qwen/Qwen3-4B VLLM_MODEL_ID=qwen3-4b python serve_model_apple.py   # a 16 GB Mac
```

The weights are fetched once by a task (`flyte.prefetch.hf_model`, the same one the Union
script uses), so the download is a run in the UI with the weights as its output, and the
app mounts that output rather than pulling from Hugging Face at startup. Put the printed
endpoint in `.env` as `VLLM_BASE_URL=http://.../v1`, add `vllm` to `FACTORY_PROVIDERS`,
and bake off `vllm:qwen3-8b`. `openai_server.py` is the same small
transformers server as the `apple-silicon-serving` tutorial: the model's own chat template
renders the tools, the Hermes-style `<tool_call>` blocks Qwen emits are parsed back into
OpenAI tool calls, thinking is off. One difference from vLLM is that this server has no
JSON-schema mode, so the engineer asks for its typed decision through a tool call instead
(`llm.py` does that for every self-served model; vLLM is fine with it too). Generation is
one stream at a time, so a bake-off with several open-model agents takes turns.

On an M5 Max, Qwen3-8B in bf16 answers at about 22 tokens a second, and the engineer it
drives meets the request. In a three-way bake-off on the devbox, Haiku, GPT-4.1 and Qwen
all met it in the same run: the two API models promoted the fine-tuned encoder (99.2%,
6 ms) in 8 and 7 runs; Qwen screened five candidates, fine-tuned three, and promoted the
fine-tuned smollm2-1.7b (100%, 50 ms) in 10 runs and 35k tokens, fewer tokens than Haiku's
52k but 295 s against 133 s, nearly all of it generation on the Mac. A competent engineer,
not the sharpest: it clears the bar with a bigger model than it needed. One thing the open
model does that the API models do not: it writes `accuracy: 97.5` where the schema says
`0.975`, so `graph.Decision` accepts a percentage and divides. Without that the whole
trial failed at the last step, three attempts in a row, after all the work was done.

Which means the engineer itself needs no API key. On a devbox with the Qwen app up:

```bash
flyte run ml_engineer.py ml_engineer_agent --model vllm:qwen3-8b
```

is the factory, the router and the engineer's brain, all on one laptop.

Models are strings from `llm.py`: `anthropic:<model>`, `openai:<model>`, or
`vllm:<model>@<url>/v1` for anything OpenAI-compatible you serve yourself.

## Step 7: crash it, and watch it resume (optional)

```bash
flyte run crash_resume.py resilient_engineer
```

**What you'll see.** The run fails once and succeeds on the retry. In the pod logs,
attempt 0 prints three `live model call` lines and then the simulated crash; attempt 1
prints four, for a seven-turn run. The missing three are the replay. In the run graph
exactly one `think:model` is red: the crash itself, with the message `simulated worker
crash after 3 model calls`; everything after it belongs to attempt 1. In the Flyte UI,
attempt 1's `run_eval` and `fine_tune` children are already done: no T4 started twice, no
model was trained twice.

**What just happened.** The task dies after its third live model call on the first
attempt, which for this agent is right after the fine-tune results come back. Flyte
retries it in a fresh container. The turns it already paid for replay from their durable
records; the tool calls it already made are not redone; the decision is produced once.

One detail that took a real run to find: a turn is looked up by a fingerprint of the
whole transcript, message ids included, and LangGraph assigns a random id to any message
that arrives without one. With random ids every attempt looks new and nothing replays.
The graph gives its seed messages and tool results stable ids (`graph.py`). If you build
your own graph on these nodes, do the same.

---

## Every promotion validated, with no agent

```bash
flyte deploy validate_on_promote.py validate_env
```

`validate_router` is a task with an artifact trigger: whenever a new version of
`ticket-router` lands, whoever published it (the agent, a person on the CLI, another
pipeline), Union starts a run of it with that version bound to the `model` input, and it
evaluates the model on every dataset version on a T4 and writes a report. Deploy it once,
then promote anything and look at the artifact's **Triggers** tab. In a room of thirty
people promoting, it fires thirty times.

Its task version is pinned at deploy time. If you change the eval code, redeploy the
trigger, or it keeps running the old code (we found that out when the pinned version tried
to load the encoder as a chat model).

## A human in the loop

```bash
FACTORY_APPROVAL=1 flyte run ml_engineer.py ml_engineer_agent
```

With `FACTORY_APPROVAL=1`, `promote` creates a `flyte.new_condition` before it publishes
anything: the run pauses, the Flyte UI shows a prompt with the model, its numbers and the
agent's reasoning, and a person approves or declines. Decline, and the agent is told the
promotion was refused and writes that up. No extra infrastructure; it is a native Flyte
condition, and it works the same on a devbox. From a terminal, the run's condition action
shows as PAUSED, and

```bash
flyte signal condition <run name> <action name> true      # false declines
```

prints the prompt and releases it. On the devbox the whole thing took 2 min 24 s of agent
time plus however long the approver takes: eight runs, the encoder promoted at 99.2% and
6 ms, the app rolled out after the signal, and the live test passed 12/12. The same gate
applies wherever `promote` runs, including the trigger-started retrain in step 5. A local
run (`--local`) has no conditions and skips the gate: its promote is a copy into a folder.

## Measuring on CPU

`run_eval_cpu` is `run_eval` on a small CPU node. Training always goes to the GPU, but the
serving app is a CPU pod, so the CPU number is what production will look like. The
fine-tuned encoder classifies a ticket in about 230 ms on the demo cluster's 2-vCPU nodes
(14 ms on the T4); the fine-tuned 0.5B chat model takes about 1.3 s. The agent is told
which candidates are small enough for this, and the decision page labels CPU rows so the
T4 latency bar is not applied to them.

---

## Cheat sheet

| File | What it is |
|---|---|
| `tickets.py` | The synthetic dataset: templates, splits, the router prompt |
| `factory.py` | `evaluate()` and `fine_tune()` over transformers, peft and trl. No Flyte in it |
| `tools.py` | The eight tools as Flyte tasks, plus `train_model` and `publish_router` behind them |
| `graph.py` | The LangGraph graph, the request, the typed decision, the parallel tool node |
| `llm.py` | The agent's model, from a string |
| `config.py` | Environments, images, secrets, and the `ACCELERATOR` knob |
| `router_app.py` | The serving app that mounts the promoted artifact, and the `Router` class a local run uses in-process |
| `report.py` | HTML for the task reports |
| `utils/workshop.py` | `run(task)` and `show()` for the notebook: submit, print the URL, wait, return outputs |
| `support_agent.py` | The support agent: route, draft a reply. Steps 0, 3 and 4 |
| `model_factory.py` | Step 1: the factory as a pipeline |
| `ml_engineer.py` | Step 2: the ML engineer agent |
| `drift.py` | Step 4: the fix by hand (`day_two`) |
| `adaptive_loop.py` | Step 5: `publish_tickets`, the trigger, and `adapt` |
| `bakeoff.py` | Step 6: the bake-off |
| `crash_resume.py` | Step 7: crash and resume |
| `validate_on_promote.py` | The artifact trigger: validate every new `ticket-router` version |
| `serve_model.py` | Optional: serve an open model on a Union vLLM app to drive the agent with |
| `serve_model_apple.py`, `openai_server.py` | The same on a Mac's GPU: a small OpenAI-compatible transformers server as a devbox app |
| `requirements.txt`, `requirements-local.txt` | The venv; the second adds the training stack for runs with no cluster (`flyte run --local`, the notebook's `LOCAL = True`) |

| Knob | Where | Default |
|---|---|---|
| the request | `--min_accuracy --max_latency_ms --candidates_to_screen --max_fine_tunes --budget` | 0.95, 150, 5, 3, 12 |
| the agent's model | `AGENT_MODEL` in `.env`, or `--model` | `anthropic:claude-haiku-4-5` |
| providers a task may use | `FACTORY_PROVIDERS` in `.env`: which secrets every agent task asks for. The agent's own provider is always included; add another before passing `model=` from it (step 6, or any step) | the agent's provider |
| where the GPU work runs | `ACCELERATOR`: `T4`, `L40s`, `apple` (the Mac's GPU, on an Apple Silicon devbox), `cpu` | `T4`; `apple` on a Mac pointed at a devbox |
| artifact registry | `FACTORY_ARTIFACTS=0` passes models around as task outputs instead (what a devbox does on its own) | detected |
| the local registry | `FACTORY_LOCAL_DIR`: where a run with no cluster keeps what it promotes | `.factory/` next to the code |
| GPU tools in flight per run | `FACTORY_MAX_PARALLEL_GPU`: the engineer still asks for five evals in one turn, but only this many run at once. Set 1 or 2 for a room on one GPU pool | `0` (no limit); `1` on a Mac and in a local run |
| deploy on promote | `FACTORY_DEPLOY` | `1` (step 6 sets `0`) |
| the router app's pod | sized to the promoted model: the encoder 2 CPU / 2Gi, a chat model up to 0.5B 2 CPU / 4Gi, larger ones a T4. `ROUTER_CPU`, `ROUTER_MEMORY`, `ROUTER_GPU` override | by model |
| human approval before deploy | `FACTORY_APPROVAL` | `0` |
| your namespace on a shared cluster | `FACTORY_TAG` | none (app and artifacts are then `ticket-router`) |
| a note for the engineer | `--situation "..."` on step 2 | none (steps 4 and 5 set their own) |
| fresh build vs. keep production | `Request.fresh` | `True` for steps 1 to 3, 6 and 7 ("ignore what is deployed, build one"); `False` for 4 and 5 ("check production first") |
| test the deployment training | `FACTORY_MAX_STEPS=2` | off |

Timings on the demo cluster: cold agent run 4 to 5 minutes, warm 45 seconds, image
build 6 minutes once, one LoRA epoch on the 0.5B 42 seconds. Two things stretch that: GPU
nodes booting adds a few minutes to the first evals, and the app rollout inside `promote` has
taken 10 to 15 minutes on the demo cluster (90 seconds elsewhere), so a 25-minute agent run
there is normal, not stuck.

---

## Notes

- `run_eval` measures latency at batch size 1 with up to eight generated tokens. A base
  model that answers "The category is billing." pays for every token; a fine-tuned one
  answers `billing` and stops. That is why fine-tuning lowers latency too.
- The T4 has no bf16. Training loads fp32 weights and uses fp16 autocast; at these sizes
  that fits easily. On Apple Silicon the same code uses bf16 autocast and ships bf16 weights.
- `fine_tune` is a CPU-side wrapper around `train_model`, the GPU task that produces the
  artifact. A tool has to return text for the model, and an artifact has to be a task's
  top-level output, so they are two tasks. `promote` and `publish_router` split the same way.
- Only `train_model` is cached. Evals used to be, and a cache hit returns the number
  without the report, which in a room reads as "nothing happened". `cache="auto"` keys on
  the inputs and the task's own source, not on files the task reads, so `fine_tune` passes
  a fingerprint of the training tickets as an input: edit the data and the fine-tune is
  retrained, edit anything else and it is not. And never cache on an unversioned
  reference: a task given `artifact:ticket-router` would return whatever "latest" meant
  the first time it ran. Resolve the version first, as `adapt` does.
- A child task's spec is serialized where it is launched. For the agent's tools that is
  inside the agent's pod, which has no `.env`, so every knob `config.py` reads from the
  environment is also passed to every environment as `env_vars` (`PROPAGATED`). Forget
  that and a pod quietly falls back to the defaults and asks for the wrong secret.
- On a cluster `flyte.serve()` blocks until the app reports activated, with no deadline.
  A revision that crashes on startup therefore hangs whatever called it. `promote` and
  `rollback` wrap the deploy in a 10-minute timeout so a bad revision becomes a tool error
  the agent can read and roll back from. Related: the app's code bundle only carries the
  modules loaded when `serve()` runs, so `router_app.py` imports its helpers at the top of
  the file, not inside the startup hook.
- From a notebook, `flyte.run` detects IPython and switches to "interactive mode": it
  pickles the task instead of bundling the source files. The pickle carries the task's own
  module by value but only references the modules it imports, so the pod dies on the first
  one with `No module named 'llm'`. `utils/workshop.py` runs with
  `flyte.with_runcontext(interactive_mode=False)` so the source is shipped, and pins
  `root_dir` to the tutorial folder so the bundle does not depend on the kernel's working
  directory. Do the same in any notebook code that calls `flyte.run` itself.
- `promote` sizes the serving pod to the model it is deploying (`serving_resources` in
  `router_app.py`): the encoder gets a small CPU pod, a chat model above 0.5B gets a T4, so
  the latency the request was judged on is the latency production sees (the fine-tuned
  1.5B: 84 ms in the eval, 85 ms from the deployed app). On a cluster with small CPU
  nodes, `ROUTER_MEMORY` and `ROUTER_CPU` are the knobs; "Insufficient memory" in the app's
  revision log is the symptom. Keep the encoder's 2 CPUs if you can: on one core it answers
  in about 600 ms instead of 200.
- The app opens its port before the model is loaded and loads it in the background
  (`/` reports `loaded`, `/classify` waits). A pod that is still reading a 6GB checkpoint
  when the platform's readiness check gives up is restarted forever. On a GPU the weights
  stream straight to the device (`device_map`); staging them in CPU memory first got the
  12Gi pod OOM-killed. `promote` polls until the new pod reports the right model *and*
  `loaded`, and fails loudly if it never does, rather than letting `test_deployment`
  measure the previous revision.
- In a room of attendees on one project, every promotion republishes `ticket-router` and
  redeploys the same app: last promotion wins. Fine for a demo; key the app name on the
  run name in `router_app.py` if everyone should get their own.
