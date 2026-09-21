"""Step 1: the model factory as a plain pipeline, no agent anywhere.

This is the factory itself: show the tickets, baseline the candidates (in parallel),
optionally fine-tune one and re-check it. You pick the models and the epochs; in step 1
the agent does. Every call is a Flyte task, a child action of this one: on the cluster
`run_eval` is a T4 container; in Colab under `--local` it uses the notebook's GPU; on a
laptop the CPU.

The evals and the fine-tune are cached, so once one person has run this, the agent in
step 1 gets those results back in a second, for everyone in the project.

No API key. No Grafana.

    flyte run --local step1_factory.py model_factory --n 20 --models '["smollm2-360m"]'   # laptop: one quick eval
    flyte run step1_factory.py model_factory                                # cluster: full evals on T4s
    flyte run step1_factory.py model_factory --fine_tune_too           # also fine-tune the 0.5B

"""

from __future__ import annotations

import asyncio
import re

import flyte
import flyte.report

from config import tools_env
from graph import Request
from report import dataset_html, evals_html
from tickets import load_split
from tools import fine_tune, list_candidates, run_eval, run_eval_cpu


def _parse(summary: str, model: str, n: int) -> dict:
    """Pull the numbers back out of a tool's text reply, for the report."""
    acc = float(re.search(r"accuracy ([\d.]+)%", summary).group(1)) / 100
    p50 = float(re.search(r"p50 (\d+) ms", summary).group(1))
    p95 = float(re.search(r"p95 (\d+) ms", summary).group(1))
    load = float(re.search(r"model load (\d+)s", summary).group(1))
    dev = re.search(r"\((cuda|cpu|mps)", summary).group(1)
    cats = dict(re.findall(r"(\w+) (\d+)%", summary.split("per category:")[1].split("\n")[0]))
    return {
        "model": model,
        "n": n,
        "accuracy": acc,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "load_seconds": load,
        "device": dev,
        "per_category": {k: int(v) / 100 for k, v in cats.items()},
        "mistakes": [],
    }


@tools_env.task(report=True)
async def model_factory(
    n: int = 120,
    models: list[str] = ["smollm2-360m", "qwen2.5-0.5b", "qwen2.5-1.5b"],
    fine_tune_too: bool = False,
    epochs: float = 1.0,
    dataset: str = "v1",
    tune_model: str = "qwen2.5-0.5b",
    device: str = "gpu",
) -> str:
    """The factory, driven by you: show the data, baseline the candidates, optionally fine-tune one and re-check it."""
    train, test = load_split("train", dataset), load_split("test", dataset)
    tab = flyte.report.get_tab("Data")
    tab.log(dataset_html(train, test))
    await flyte.report.flush.aio()

    transcript = [f"$ list_candidates()\n{await list_candidates.ainvoke({})}\n"]
    results = []
    # The evals are independent, so launch them together: on the cluster, one T4 each.
    evaluator = run_eval_cpu if device == "cpu" else run_eval
    with flyte.group("baseline-evals"):
        outs = await asyncio.gather(*(evaluator.ainvoke({"model": m, "n": n, "dataset": dataset}) for m in models))
    for m, out in zip(models, outs, strict=True):
        transcript.append(f"$ run_eval(model={m!r}, n={n})\n{out}\n")
        results.append(_parse(out, m, n))
    if fine_tune_too:
        with flyte.group("fine-tune-and-recheck"):
            out = await fine_tune.ainvoke({"model": tune_model, "epochs": epochs, "dataset": dataset})
            transcript.append(f"$ fine_tune(model={tune_model!r}, epochs={epochs})\n{out}\n")
            path = out.split("new model reference: ")[-1].strip()
            out = await evaluator.ainvoke({"model": path, "n": n, "dataset": dataset})
            transcript.append(f"$ run_eval(model={path!r}, n={n})\n{out}\n")
            results.append(_parse(out, f"{tune_model} (fine-tuned, {epochs:g} ep)", n))

    await flyte.report.replace.aio(evals_html(results, Request(dataset=dataset)))
    await flyte.report.flush.aio()
    text = "\n".join(transcript)
    print(text)
    return text


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(model_factory)
    print(run.url)
