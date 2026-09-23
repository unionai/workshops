# The model factory, run by an agent, watched by Grafana

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
support agent starts misrouting them, you see it in Grafana, the platform notices, and the
engineer adapts the model while nobody is watching.

Flyte runs every tool the engineer calls as its own durable container. Grafana Agent
Observability sees both agents: every model turn, every tool call, tokens, cost, the
transcripts. The router app reports what it serves to the same stack, which is where the
drift shows up first.

The whole thing runs three ways: in Colab with `flyte run --local` (the fine-tunes use
Colab's T4), on your laptop (CPU, slow, use small `--n`), or on a Union cluster, which is
where the parallel GPUs, the artifacts, the deployed app, the triggers and the Grafana
links live. The notebook `langgraph-grafana-agent-tutorial.ipynb` walks the same steps.

| Step | You will | Needs |
|---|---|---|
| 0 | Run the support agent as it is today, routing with the API model. The "before" | a model API key |
| 1 | Drive the factory yourself as a pipeline: the data, parallel evals, a fine-tune | nothing |
| 2 | Let the ML engineer build the router, deploy it, and test the deployment | a model API key |
| 3 | Switch the support agent to the self-hosted router. The "after" | same |
| 4 | Watch the engineer's run in Grafana | + a Grafana token |
| 5 | Kill the engineer mid-run and watch it resume without redoing work | same |
| 6 | Bake off Claude vs Haiku vs whatever as the engineer's brain | same |
| 7 | Drift: a new kind of ticket arrives. See it in Grafana, then turn the factory again by hand | same |
| 8 | The loop closes itself: new tickets land, a trigger checks production, the engineer retrains | same |

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

   meanwhile, on the side (config.py, one init() call):

      every model turn ──► generation   (model, tokens, cost, prompt, answer)   ┐
      every tool call  ──► step                                                 │  Grafana Agent
      the Flyte run    ──► conversation (one per run, for both agents)          │  Observability
      the task         ──► agent name, agent version                            ┘  + Tempo trace
      the router app   ──► predictions per label, confidence, latency  ─► Grafana metrics (the drift dashboard)
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

| router | zero-shot | p50 | where it runs |
|---|---|---|---|
| Claude Opus 5 | 98.3% | 1.7 s | API |
| GPT-4.1 | 97.5% | 634 ms | API |
| Claude Haiku 4.5 | 95.8% | 606 ms | API |
| Qwen3-8B (vLLM) | 94.2% | 386 ms | one L40s, ours |
| modernbert-base, fine-tuned | 99.2% | 14 ms on a T4, 232 ms on the app's CPU pod | 149M params, ours |

The big API models clear the bar without training and the trained encoder beats all of
them at a hundredth of the latency. Qwen3-8B, an open model with 50× the encoder's
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

Eight model turns, thirteen tool calls, about 13k tokens. Warm, with the eval matrix and
the fine-tunes cached, the same run is under a minute, which is what the second person in
the room sees.

In the Flyte UI the run graph shows the fan-out, and every action is named after what
it did: `run_eval · qwen2.5-0.5b` four across, then `fine_tune · modernbert-base · 3ep`
with its `train_model` child, then the rechecks, then `promote` with `publish_router`
under it, then `test_deployment`. Each eval and training action has its own report (numbers,
per-category accuracy, sample mistakes). The parent task's report is the decision page:
what was promoted, pass/fail chips against the request, everything the agent measured in
one table, the sequence of calls, and the rationale, plus an **Agent** tab with the turn
by turn timeline. The `ticket-router` artifact's **Lineage** tab leads from the app back
through the promotion to the fine-tune that produced the weights.

---

## Before you start

```bash
git clone https://github.com/unionai/workshops
cd workshops/tutorials/langgraph-grafana-agent
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

Step 1 needs nothing else. For the agents, put a key in `.env`; `config.py` loads it:

```bash
ANTHROPIC_API_KEY=sk-ant-...
# or, to drive the agent with OpenAI instead:
# OPENAI_API_KEY=sk-...
# AGENT_MODEL=openai:gpt-4.1
```

### If you have a Union cluster (recommended)

```bash
flyte create config --endpoint <your-endpoint> --project flytesnacks --domain development --builder remote
flyte create secret ANTHROPIC_API_KEY --value sk-ant-...
```

The first call opens your browser to sign in. Anywhere without a browser to hand off to
(Colab, a remote shell) add `--auth-type headless`: the first call then prints a login URL
and a code to paste back. Without it the browser flow fails silently and the first upload
dies with an empty `ConnectError`; the notebook already passes the flag.

On a shared cluster, set `FACTORY_TAG=<your handle>` in `.env`. It namespaces what has
to be yours: the serving app (`ticket-router-<tag>`), the production artifacts, and the
environments that carry triggers. The GPU eval and training tasks stay shared, so the eval
matrix is cached once for the whole room. Prefix your secret names the same way
(`FLYTE_SECRET_PREFIX=SAGE_` makes the code ask for `SAGE_ANTHROPIC_API_KEY`), or point at
an existing secret by name (`OPENAI_API_KEY_SECRET_NAME=my-openai-key`).

The first cluster run builds two images. The one with the training stack takes about six
minutes, once.

### If you have a Grafana Cloud stack (step 4 onward, and the support agent's conversations)

One token, four plain values. An admin enables **Observability → Agent Observability**
on the stack once. Then collect:

| Put in `.env` | Where it comes from |
|---|---|
| `GRAFANA_HOST` | your stack URL, `https://<stack>.grafana.net` |
| `AGENTO11Y_ENDPOINT` | Agent Observability → Configuration → API URL |
| `AGENTO11Y_AUTH_TENANT_ID` | same page → Instance ID |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | your stack's page on grafana.com → OpenTelemetry card |
| `GRAFANA_TOKEN` | an access policy token with scopes `sigil:write` and `traces:write` |

On the cluster the token is a secret; the other four ride along as environment variables:

```bash
flyte create secret GRAFANA_TOKEN --value glc_...
```

`config.py` derives the OTLP basic-auth header from the tenant id and token, and sets
agento11y to HTTP with Basic auth (its defaults are gRPC and no auth, which Grafana Cloud
answers with a 401 and the token never sent). Without these values, every step still
runs; they just print that nothing is being exported. The Instance ID is also readable
inside the token itself: the `glc_` payload is base64 JSON whose `n` field names
`stack-<id>-…`.

---

## Step 0: the support agent, as it is today

```bash
flyte run --local support_agent.py agent_handle_tickets --n 10 --router llm    # laptop
flyte run support_agent.py agent_handle_tickets --router llm                   # cluster, 30 tickets
```

**What you'll see.** Thirty held-out tickets go through the support agent: the router picks
a queue, then the queue's drafting agent (the same model, prompted as that team) writes a
two-line reply for the team to send. The report has
the numbers that matter for the request: routing accuracy, p50 routing latency, tokens,
and cost per 1,000 tickets, plus where the tickets landed and every draft. With Grafana
configured, the batch is a conversation in Agent Observability with one generation per
ticket per step.

The router is accurate, slow, and priced per ticket. We ran the same thirty tickets
through three API models; the cost column is routing and replies together, at list price:

| Support agent model | Routing accuracy | Route p50 | Per 1,000 tickets |
|---|---|---|---|
| Claude Opus 5 (the default) | 96.7% | 2.28 s | $10.28 |
| Claude Haiku 4.5 | 96.7% | 596 ms | $1.18 |
| GPT-4.1 | 93.3% | 594 ms | $0.64 |

Opus thinks before it routes, which is why it is four times slower than the others at
the same accuracy. That is the "before." Pick the model with `--model`, or `AGENT_MODEL`.

**What just happened.** `support_agent.py` is a two-node LangGraph graph, `route → draft`.
Each model call is a `flyte.trace` step whose arguments are the ticket itself, so the run
graph shows every ticket's text going in and the queue or reply coming out, and a batch
that dies halfway replays what it already did.
`--router llm` classifies with the API model; `--router oss` calls the `ticket-router`
app instead, which does not exist yet. Building it is the request.

---

## Step 1: the factory as a pipeline

This is the factory with you at the controls: you pick the candidates and the epochs, the same tools do the work, and you look at what comes back. In step 2 the ML engineer takes over the controls. In the Flyte UI the evals appear side by side in a group called `baseline-evals`, and the fine-tune plus its recheck in `fine-tune-and-recheck`.

```bash
flyte run --local model_factory.py model_factory --n 20 --models '["smollm2-360m"]'   # laptop: one quick eval
flyte run model_factory.py model_factory --fine_tune_too                           # cluster: the full matrix
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
And they are cached: when the engineer asks for the same eval in step 2, it gets the answer
in a second, and so does everyone else in the project.

---

## Step 2: let the ML engineer build the router

```bash
flyte run --local ml_engineer.py ml_engineer_agent --candidates_to_screen 2 --max_fine_tunes 1   # laptop: keep it small
flyte run ml_engineer.py ml_engineer_agent                                                       # cluster
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
already done, and attempt two finished in 13 seconds with nothing redone. That is step 8,
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

**About that second attendee.** Step 1's request says "treat this as a fresh build, ignore
what is deployed", so thirty people running it get thirty full investigations rather than
twenty-nine "production already meets the bar" in 35 seconds. Steps 5 and 6 say the
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
to a CPU pod; what is left of the cost is the reply drafts:

| Support agent model | Routing accuracy | Route p50 | Per 1,000 tickets |
|---|---|---|---|
| Claude Opus 5 | 96.7% → 100% | 2.28 s → 128 ms | $10.28 → $6.47 |
| Claude Haiku 4.5 | 96.7% → 100% | 596 ms → 170 ms | $1.18 → $0.30 |
| GPT-4.1 | 93.3% → 96.7% | 594 ms → 200 ms | $0.64 → $0.34 |

Put this report next to step 0's. In Grafana it is a second conversation from the same
agent, with the routing generations gone.

**What just happened.** Nothing in the support agent changed except a flag. The router
app is a FastAPI service that mounts the promoted artifact; `--router oss` posts the
ticket to `/classify` and reads back the queue and the model's confidence. That is the
first piece of this agent running on an open model.

---

## Step 4: watch both agents in Grafana

Nothing to run. Open the step 2 run in the Flyte UI: the task carries two links.
**Grafana Agent Observability** opens this run's conversation: each `think` is a
generation with its prompt, answer, model, tokens and cost; each tool call is a step; the
header has the totals. **Grafana trace** opens the same run in Tempo, where a `fine_tune`
span is a minute and a half wide and the three of them overlap. The support agent's runs
from steps 0 and 3 are conversations too, side by side: one full of routing generations,
one without them.

**What just happened.** Nothing changed in either agent. `config.py` calls
`flyteplugins.agento11y.init()` once, at module scope, when the Grafana values are
present. The plugin binds the Flyte run name as the conversation id and the task name as
the agent name, and hands the graph a callback handler through the adapter. That is the
whole integration.

`init()` has to be at module scope: the Flyte task span opens before the task body runs,
and the binding that names the run as the conversation rides on that span.

---

## Step 5: drift, and the fix by hand

```bash
flyte run support_agent.py agent_handle_tickets --router oss --dataset v2      # see it
flyte run drift.py day_two                                          # fix it
flyte run support_agent.py agent_handle_tickets --router oss --dataset v2 --labels_from v2   # see it fixed
```

The support team adds a ninth category, `data_request`, for privacy and data-subject
requests that used to be filed under `other`. Our tickets are now dataset v2. The router in
production was trained on v1 and has never seen the label.

**What you'll see.** First the support agent, routing v2 tickets with the v1 router: on
our cluster 88.9% and every GDPR ticket filed under `other`, which the report flags as
"tickets in a queue the router does not know about". That is the drift, in the agent that
suffers it. Then the engineer: it starts with `production_status` (which artifact version
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

**Try it with a ticket the old model could not route:**

```bash
curl -X POST "$ENDPOINT/classify" -H 'content-type: application/json' \
     -d '{"text": "Under GDPR I would like a copy of all the data you hold on me"}'
# before day two: {"label":"cancellation", ...}    after: {"label":"data_request", ...}
```

### Seeing the drift in Grafana

The router app reports three OpenTelemetry metrics to the same stack the agents export
to: `ticket_router_predictions` (a counter by predicted label, model and dataset),
`ticket_router_confidence` (a histogram, 0 to 1) and `ticket_router_latency_ms`. They need
the token to carry `metrics:write`. `grafana/ticket-router-drift.json` is a dashboard for
them: import it (Dashboards → New → Import), pick your Prometheus datasource, and run the
support agent over v2 tickets. The share routed to `other` climbs, confidence's p10 drops,
and after the engineer promotes a v2 model the `data_request` series appears and the
"trained on" panel flips from v1 to v2. Detection happens in Union (the trigger in step
8); seeing it happen is Grafana's job.

## Step 6: the loop closes itself

```bash
flyte deploy adaptive_loop.py observed_env            # registers the trigger, once
flyte run adaptive_loop.py publish_tickets --version v2
```

Step 5 had you run the factory when the data changed. Here nobody does:

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
`adapt` appears that you did not start. It is the same engineer task as steps 2 and 5,
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

## Step 7: bake off the engineer's brain (optional)

```bash
flyte run --local bakeoff.py bakeoff --models '["anthropic:claude-opus-5", "anthropic:claude-haiku-4-5"]'
flyte run bakeoff.py bakeoff --models '["anthropic:claude-opus-5", "anthropic:claude-haiku-4-5", "vllm:qwen3-8b"]' --trials 2
```

**What you'll see.** One child action per (model, trial), in parallel. The report scores
each: met the constraints, stayed in budget, runs spent, tokens, wall clock. In Grafana
each agent model is its own agent version, so the conversations sit side by side and you
can read how each one argued for the encoder over the 1.5B.

**What just happened.** Same request, same tools, different `model` string. Because the
evals and fine-tunes are cached, after the first agent the rest mostly hit cache, so the
bake-off measures the agents rather than the GPUs. Promotion publishes artifacts but does
not deploy here (`FACTORY_DEPLOY=0`). With a warm matrix, Claude Opus 5 met the
request in 10 runs, 63k tokens and 99 seconds; Claude Haiku 4.5 met it in 10 runs, 53k
tokens and 56 seconds; GPT-4.1 in 7 runs, 23k tokens and 49 seconds; GPT-4.1-mini in 11
runs, 29k tokens and 100 seconds. Same tools, same cache, different amount of flailing.

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

Models are strings from `llm.py`: `anthropic:<model>`, `openai:<model>`, or
`vllm:<model>@<url>/v1` for anything OpenAI-compatible you serve yourself.

## Step 8: crash it, and watch it resume (optional)

```bash
flyte run crash_resume.py resilient_engineer
```

**What you'll see.** The run fails once and succeeds on the retry. In the pod logs,
attempt 0 prints three `live model call` lines and then the simulated crash; attempt 1
prints four, for a seven-turn run. The missing three are the replay. In the run graph
exactly one `think:model` is red: the crash itself, with the message `simulated worker
crash after 3 model calls`; everything after it belongs to attempt 1. In the Flyte UI,
attempt 1's `run_eval` and `fine_tune` children are cache hits: no T4 started twice, no
model was trained twice. In Tempo, both attempts are one trace, and the replayed steps
are marked `flyte.replayed`. A stock OpenTelemetry setup would show two unrelated traces
with holes where the replays are.

**What just happened.** The task dies after its third live model call on the first
attempt, which for this agent is right after the fine-tune results come back. Flyte
retries it in a fresh container. The turns it already paid for replay from their durable
records; the tool calls it already made are cache hits; the decision is produced once.

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
agent's reasoning, and a person approves or declines (or signals it from the CLI with
`flyte signal condition`). Decline, and the agent is told the promotion was refused and
writes that up. No extra infrastructure; it is a native Flyte condition.

## Measuring on CPU

`run_eval_cpu` is `run_eval` on a small CPU node. Training always goes to the T4, but the
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
| `config.py` | Environments, images, secrets, the Grafana `init()` |
| `router_app.py` | The serving app that mounts the promoted artifact |
| `report.py` | HTML for the task reports |
| `utils/workshop.py` | `run(task)` and `show()` for the notebook: submit, print the URL, wait, return outputs |
| `support_agent.py` | The support agent: route, draft a reply. Steps 0, 3 and 5 |
| `model_factory.py` | Step 1: the factory as a pipeline |
| `ml_engineer.py` | Step 2: the ML engineer agent |
| `drift.py` | Step 5: the fix by hand (`day_two`) |
| `adaptive_loop.py` | Step 6: `publish_tickets`, the trigger, and `adapt` |
| `bakeoff.py` | Step 7: the bake-off |
| `crash_resume.py` | Step 8: crash and resume |
| `validate_on_promote.py` | The artifact trigger: validate every new `ticket-router` version |
| `serve_model.py` | Optional: serve an open model on a Union vLLM app to drive the agent with |

| Knob | Where | Default |
|---|---|---|
| the request | `--min_accuracy --max_latency_ms --candidates_to_screen --max_fine_tunes --budget` | 0.95, 150, 5, 3, 12 |
| the agent's model | `AGENT_MODEL` in `.env`, or `--model` | `anthropic:claude-opus-5` |
| providers a task may use | `FACTORY_PROVIDERS` in `.env`: which secrets every agent task asks for. The agent's own provider is always included; add another before passing `model=` from it (step 7, or any step) | the agent's provider |
| deploy on promote | `FACTORY_DEPLOY` | `1` (step 7 sets `0`) |
| the router app's pod | sized to the promoted model: the encoder 2 CPU / 2Gi, a chat model up to 0.5B 2 CPU / 4Gi, larger ones a T4. `ROUTER_CPU`, `ROUTER_MEMORY`, `ROUTER_GPU` override | by model |
| human approval before deploy | `FACTORY_APPROVAL` | `0` |
| your namespace on a shared cluster | `FACTORY_TAG` | none (app and artifacts are then `ticket-router`) |
| a note for the engineer | `--situation "..."` on step 2 | none (steps 5 and 6 set their own) |
| fresh build vs. keep production | `Request.fresh` | `True` for steps 1 to 4, 7 and 8 ("ignore what is deployed, build one"); `False` for 5 and 6 ("check production first") |
| test the deployment training | `FACTORY_MAX_STEPS=2` | off |

Timings on the demo cluster: cold agent run 4 to 5 minutes, warm 45 seconds, image
build 6 minutes once, one LoRA epoch on the 0.5B 42 seconds.

---

## Notes

- `run_eval` measures latency at batch size 1 with up to eight generated tokens. A base
  model that answers "The category is billing." pays for every token; a fine-tuned one
  answers `billing` and stops. That is why fine-tuning lowers latency too.
- The T4 has no bf16. Training loads fp32 weights and uses fp16 autocast; at these sizes
  that fits easily.
- `fine_tune` is a CPU-side wrapper around `train_model`, the GPU task that produces the
  artifact. A tool has to return text for the model, and an artifact has to be a task's
  top-level output, so they are two tasks. `promote` and `publish_router` split the same way.
- `cache="auto"` versions a task by its own source. Editing `tickets.py` does not
  invalidate cached evals; edit `tools.py` (or bump `n`) if you need fresh numbers. And
  never cache on an unversioned reference: `run_eval("artifact:ticket-router")` returns
  whatever "latest" meant the first time it ran. Resolve the version first, as `adapt` does.
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
- `flyteplugins-agento11y` is installed from the flyte-sdk repository; it is not on PyPI
  yet. The image adds `git` for that reason.
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
