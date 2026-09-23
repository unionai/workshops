"""Every eval the factory runs, as a scored trial in Grafana Agent Observability.

Agent Observability already shows each run as a conversation. This adds the other half:
an *experiment* per run, with one trial per `run_eval` the engineer (or you, in step 1)
asked for, scored on accuracy and latency against the request's bar. In Grafana the
experiment sits next to the conversation, under the same id, so the eval numbers and the
agent's reasoning about them are one click apart.

Nothing here can fail a run: without a Grafana stack it does nothing, and with one it
logs and moves on if the export fails.
"""

from __future__ import annotations

import os
import re

import flyte

from config import GRAFANA_CONFIGURED, GRAFANA_HOST

EVAL_PATTERN = re.compile(
    r"accuracy (?P<accuracy>[\d.]+)% on (?P<n>\d+) tickets; latency p50 (?P<p50>\d+) ms, p95 (?P<p95>\d+) ms"
)


def parse_eval(model: str, text: str) -> dict | None:
    """The numbers out of a run_eval reply, or None if the reply is an error."""
    m = EVAL_PATTERN.search(text)
    if not m:
        return None
    return {
        "model": model,
        "n": int(m["n"]),
        "accuracy": float(m["accuracy"]) / 100,
        "latency_p50_ms": float(m["p50"]),
        "latency_p95_ms": float(m["p95"]),
    }


def experiment_id() -> str | None:
    """The run name, which is also the conversation id in Agent Observability."""
    ctx = flyte.ctx()
    if ctx is None or ctx.mode == "local":
        return None
    run = ctx.action.run_name
    return run if ctx.action.name == "a0" else f"{run}-{ctx.action.name}"


def record_evals(
    results: list[dict],
    *,
    min_accuracy: float,
    max_latency_ms: float,
    dataset: str,
    agent_model: str,
    name: str = "model factory",
) -> str | None:
    """Export one trial per eval. Returns the experiment's Grafana URL, or None."""
    if not GRAFANA_CONFIGURED or not results:
        return None
    exp_id = experiment_id()
    if exp_id is None:
        return None
    try:
        from agento11y.experiments import Candidate, Client, Evaluator, Experiment, TestCase, TestSuite

        client = Client(
            os.environ["AGENTO11Y_ENDPOINT"],
            tenant_id=os.environ.get("AGENTO11Y_AUTH_TENANT_ID", ""),
            ingest_token=os.environ["AGENTO11Y_AUTH_TOKEN"],
            grafana_url=GRAFANA_HOST,
        )
        held_out = Evaluator(evaluator_id="held-out-tickets", version=dataset, kind="deterministic")
        cases = [
            TestCase(
                test_case_id=r["model"],
                name=r["model"],
                category=dataset,
                input=f"{r['n']} held-out tickets, dataset {dataset}",
                expected=f">= {min_accuracy:.0%} accuracy, <= {max_latency_ms:.0f} ms p50 on a T4",
            )
            for r in results
        ]
        suite = TestSuite(suite_id=f"ticket-router-{dataset}", name=f"ticket router, dataset {dataset}", test_cases=cases)
        candidate = Candidate(
            agent_name=name, model_provider=agent_model.split(":")[0], model_name=agent_model.split(":", 1)[-1]
        )
        with Experiment(
            client,
            experiment_id=exp_id,
            name=f"{name} · {exp_id}",
            suite=suite,
            candidate=candidate,
            default_evaluator=held_out,
            tags=["model-factory", dataset],
        ) as exp:
            for r, case in zip(results, cases, strict=True):
                acc_ok = r["accuracy"] >= min_accuracy
                lat_ok = r["latency_p50_ms"] <= max_latency_ms
                with exp.trial(case) as trial:
                    trial.bind_conversation(exp_id.split("-")[0])
                    trial.check_score(
                        "accuracy",
                        passed=acc_ok,
                        value=round(r["accuracy"], 4),
                        explanation=f"{r['accuracy']:.1%} on {r['n']} tickets",
                    )
                    trial.check_score("latency_p50_ms", passed=lat_ok, value=r["latency_p50_ms"])
                    trial.score("latency_p95_ms", r["latency_p95_ms"])
                    trial.final_score(
                        round(r["accuracy"], 4),
                        passed=acc_ok and lat_ok,
                        explanation="meets the request" if acc_ok and lat_ok else "below the bar",
                    )
        return client.experiment_url(exp_id)
    except Exception as exc:  # noqa: BLE001  (never fail a run over telemetry)
        print(f"grafana evals: export skipped: {type(exc).__name__}: {exc}")
        return None
