# Fine-tuning open models with agents: from eval to deployment

*Durable, self-healing agents on Union, with end-to-end observability in Grafana.*

Hands-on, 90 minutes. Bring a laptop; everything runs in Colab or on a cluster we give you access to.

## What we'll build

You have a support agent in production. It routes tickets and drafts replies, and today its routing step is an API call per ticket: accurate, slow, and billed by someone else. We're going to replace that call with a model you own, and we're not going to pick the model ourselves.

A second agent does it: an ML Engineer Agent built with LangGraph, running it's own model factory on Union. It evaluates open-source candidates in parallel, fine-tunes the ones worth it, and picks the smallest model that clears the bar. It registers the winner as an artifact, deploys it, tests the live deployment, and the support agent switches over. Routing goes from 600 ms to 200 ms, the API bill for it goes to zero, and the weights are yours.

Then the data changes. A new kind of ticket appears, the router starts misfiling it, and you watch that happen in Grafana. A trigger notices, the ML Engineer Agent automatically retrains, the platform validates and redeploys, and the support agent is fine again. Nobody clicked anything. Along the way we'll kill the engineer mid-run and watch it resume without retraining a thing.

## Why this pattern matters

The ticket router is one example. The same agent loop works for a reranker, a fraud
scorer, a PII detector, an intent model, or for drafting replies and extracting fields.
Anywhere an agent leans on a frontier model for a narrow job and you have the data, a
smaller model can be faster, cheaper, yours, and often better at that one job. We think a
lot of agent work is going this way: agents that kick off real training and deployment,
and get called back when the data shifts.

## What you'll learn

- Running a LangGraph agent on Union.ai so every tool call is a durable container: parallel GPU jobs, retries, replay after a crash, a cache the whole room shares
- Grafana Agent Observability for agents: one conversation per run, every generation and tool call with tokens and cost, both agents side by side, and the router's own metrics for spotting drift
- Union artifacts as the handoff between training, serving and validation, with lineage from the deployed model back to the run that made it
- Artifact triggers and a human approval gate, so the loop can close on its own or wait for a person
- What an agent gets wrong when nobody scores it: we'll show one round 94.2% up to a 95% bar, and how the report caught it

## Bring

A laptop with a browser. A Claude or OpenAI API key if you want to drive the agent yourself; we'll have a shared cluster and GPUs ready. A Grafana Cloud stack is optional; we'll show ours.

## Who it's for

Engineers building agents who want them to do real work with real compute, and teams weighing open models against API calls who want to see the switch happen live. You should be comfortable reading Python.
