# Fine-tuning open models with agents: from eval to deployment

*Durable, self-healing agents on Union.*

Hands-on, 90 minutes. Bring a laptop; everything runs in a hosted notebook against a cluster we give you access to.

## What we'll build

You have a support agent in production. It routes tickets and drafts replies, and today its routing step is an API call per ticket: accurate, slow, and billed by someone else. We're going to replace that call with a model you own, and we're not going to pick the model ourselves.

A second agent does it: an ML Engineer Agent built with LangGraph, running it's own model factory on Union. It evaluates open-source candidates in parallel, fine-tunes the ones worth it, and picks the smallest model that clears the bar. It registers the winner as an artifact, deploys it, tests the live deployment, and the support agent switches over. Routing goes from 600 ms to 200 ms, the API bill for it goes to zero, and the weights are yours.

Then the data changes. A new kind of ticket appears, the router starts misfiling it (confidently, which is its own lesson), and the support agent's report shows it. The team labels the new tickets and publishes them; a trigger notices, the ML Engineer Agent automatically retrains, the platform validates and redeploys, and the support agent is fine again. Nobody clicked anything. Along the way we'll kill the engineer mid-run and watch it resume without retraining a thing.

## Why this pattern matters

The ticket router is one example. The same agent loop works for a reranker, a fraud
scorer, a PII detector, an intent model, or for drafting replies and extracting fields.
Anywhere an agent leans on a frontier model for a narrow job and you have the data, a
smaller model can be faster, cheaper, yours, and often better at that one job. We think a
lot of agent work is going this way: agents that kick off real training and deployment,
and get called back when the data shifts.

Where we see it being used:

- **A model factory with an agent at the controls.** A product needs a dozen narrow models:
  routers, classifiers, rerankers, extractors. Each one is a request ("open weights, 95%,
  150 ms, 12 runs") handed to the engineer agent, which screens candidates, fine-tunes,
  registers the winner as an artifact and deploys it. The platform keeps the record: every
  eval, every fine-tune, the lineage from the model in production back to the run and the
  data that made it. One request per model, no notebooks to hand around.
- **Taking the frontier model out of an agent's hot path.** Routing, intent, moderation,
  PII detection, field extraction, reranking retrieval results, tagging entities, drafting
  in a house style. Each of these is an API call per event today. Each is a classification
  or short-generation job a small model learns from a few hundred examples, and the agent
  that depends on it swaps a model string for an endpoint it owns.
- **Retraining that runs itself when the data shifts.** A new ticket type, a new product
  line, a new locale. Labeled examples land as a versioned artifact, a trigger wakes the
  engineer, it measures the live model against the new data, retrains, validates and
  redeploys. A human approval gate sits in front of the promotion where the stakes call for
  one. Nobody schedules a retrain, and the report shows what changed and why.
- **Per-customer or per-locale models from one request.** Thirty attendees each promote a
  router in this workshop and every promotion is validated. The same mechanism trains a
  router per tenant, a scorer per market, or a model per language, each with its own
  artifact and its own history, from a template request and the customer's data.
- **Choosing the agent's own brain on evidence.** The bake-off runs the same request
  across Claude, GPT and a self-hosted open model and scores each on constraints met, runs
  spent, tokens and time. The question "is the cheap model good enough for this job" gets a
  number instead of an opinion, and the open model that serves the router can also run the
  engineer, with no API key anywhere in the loop.
- **Everything on one machine, when the data cannot leave.** The Apple Silicon path runs the
  evals, the fine-tunes, the deployed router and the engineer's model on a laptop, with the
  same code and the same records as the cluster. Regulated data, air-gapped environments,
  and a demo that works on a plane.
- **Durable, resumable training jobs under an agent.** Fine-tunes are child actions with
  retries and a cache, and every model turn is recorded. Kill the engineer mid-run and it
  resumes from the last tool result without paying for the work again. That is what makes
  it safe to let an agent spend GPU hours.

## What you'll learn

- Running a LangGraph agent on Union.ai so every tool call is a durable container: parallel GPU jobs, retries, replay after a crash, a cache the whole room shares
- Union artifacts as the handoff between training, serving and validation, with lineage from the deployed model back to the run that made it
- Artifact triggers and a human approval gate, so the loop can close on its own or wait for a person
- What an agent gets wrong when nobody scores it: we'll show one round 94.2% up to a 95% bar, and how the report caught it

## Bring

A laptop with a browser. A Claude or OpenAI API key if you want to drive the agent yourself; we'll have a shared cluster and GPUs ready.

## Who it's for

Engineers building agents who want them to do real work with real compute, and teams weighing open models against API calls who want to see the switch happen live. You should be comfortable reading Python.
