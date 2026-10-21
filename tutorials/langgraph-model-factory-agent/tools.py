"""The ML engineer's tools. Each one is a Flyte task.

`@tool` on top of `@env.task` makes a function two things at once: a LangChain
`StructuredTool` the model can call (schema from the type hints and docstring), and a
durable Flyte task. When the graph invokes it on a cluster, it runs as its own child
action in its own container: `run_eval` and `fine_tune` on a T4, the others on a small
CPU pod. Locally it just runs. When the model asks for several tools in one turn, the
graph launches them together, so three evals are three T4s at once.

Models move between tools as Union artifacts. `fine_tune` registers its output as a
named, versioned model artifact; `run_eval` and `promote` accept `artifact:<name>@<version>`
and fetch it. `promote` publishes the winner as a new version of the `ticket-router`
artifact and deploys the serving app that mounts it; `test_deployment` calls the live app;
`rollback` redeploys an earlier version. Lineage in the Union UI then runs from the
training run, through the artifact, to the running app.

Evals are not cached, on purpose: a cache hit returns the number but not the report, and
in a workshop a run whose children all say "cached" looks like a run where nothing
happened. An eval is a minute on a T4 and they run in parallel, so every run shows its
work. Training is cached (`cache="auto"` on `train_model`): the same fine-tune of the same
model is the same weights, and a room full of people should not each train the same
three models. The cache key is the inputs plus a fingerprint of the training tickets, so
editing the data retrains; editing an unrelated file does not.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import tempfile
import time

import flyte
import flyte.artifacts as artifacts
import flyte.report
from flyte.io import Dir
from flyteplugins.agents.langgraph import tool

from config import (
    FACTORY_APPROVAL,
    FACTORY_ARTIFACTS,
    FACTORY_DEPLOY,
    GPU_ARTICLE,
    GPU_LABEL,
    LOCAL_DIR,
    cpu_env,
    gpu_env,
    tagged,
    tools_env,
)
from factory import CANDIDATES, evaluate
from factory import fine_tune as _fine_tune
from report import (  # module scope: the code bundle only carries modules loaded at serialization
    evals_html,
    training_html,
)
from router_app import make_router_app
from tickets import DATASET_VERSIONS, load_split

ROUTER_ARTIFACT = tagged("ticket-router")  # the production artifact; every promotion is a new version
APP_NAME = tagged("ticket-router")  # the serving app
FT_PREFIX = "ticket-router"  # fine-tune artifacts are shared across tags: same inputs, same weights


def _is_local() -> bool:
    ctx = flyte.ctx()
    return ctx is None or ctx.mode == "local"


_REGISTRY: str | None = None


async def registry() -> str:
    """`artifacts` on a platform with an artifact registry (Union), `outputs` everywhere else.

    Decided once per process, by asking. A devbox accepts `artifacts.new(...)` on a task
    output without complaint and then cannot find the artifact, so the first lookup is the
    only reliable test. In `outputs` mode the fine-tunes and the promoted router are passed
    around as the storage paths of task outputs, and the app mounts the output directly:
    the same files, no registry, no lineage tab.
    """
    global _REGISTRY
    if _REGISTRY is None:
        if _is_local() or not FACTORY_ARTIFACTS:
            _REGISTRY = "outputs"
        else:
            from flyte.remote import Artifact

            try:
                async for _ in Artifact.listall.aio(name=ROUTER_ARTIFACT):
                    break
                _REGISTRY = "artifacts"
            except Exception as e:  # noqa: BLE001
                if "not found" not in str(e).lower() and "unimplemented" not in str(e).lower():
                    raise
                _REGISTRY = "outputs"
    return _REGISTRY


# ── The local registry: what a run with no platform promotes to ──────────────
#
# `flyte run --local` (a laptop, Colab) has no artifact registry and no app. `promote`
# copies the winner into LOCAL_DIR/ticket-router/<version>/ and points `current.json` at
# it; the support agent's `--router oss` and `test_deployment` load that in-process.


def _local_registry() -> str:
    return os.path.join(LOCAL_DIR, ROUTER_ARTIFACT)


def local_current() -> dict | None:
    """The version in production in a local run: {version, path, model, accuracy, ...} or None."""
    p = os.path.join(_local_registry(), "current.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _local_versions() -> list[dict]:
    reg = _local_registry()
    if not os.path.isdir(reg):
        return []
    out = []
    for name in sorted(os.listdir(reg), reverse=True):
        p = os.path.join(reg, name, "router.json")
        if os.path.exists(p):
            with open(p) as f:
                out.append({**json.load(f), "path": os.path.join(reg, name)})
    return out


def _local_publish(model: str, accuracy: float, latency_p50_ms: float, reason: str, dataset: str) -> dict:
    import shutil

    version = f"v{int(time.time())}"
    dest = os.path.join(_local_registry(), version)
    shutil.copytree(model, dest)
    entry = {
        "model": model,
        "accuracy": accuracy,
        "latency_p50_ms": latency_p50_ms,
        "reason": reason,
        "dataset": dataset,
        "version": version,
    }
    with open(os.path.join(dest, "router.json"), "w") as f:
        json.dump(entry, f, indent=2)
    with open(os.path.join(_local_registry(), "current.json"), "w") as f:
        json.dump({**entry, "path": dest}, f, indent=2)
    return {**entry, "path": dest}


def _local_test(n: int, dataset: str) -> str:
    """test_deployment with no app: the current local version, loaded in-process."""
    current = local_current()
    if not current:
        return "FAIL: nothing is in production. Nothing has been promoted in this local registry; promote first."
    from router_app import Router

    r = Router(current["path"])
    tickets = load_split("test", dataset)[-max(1, min(n, 40)) :]
    hits, latencies, misses = 0, [], []
    for t in tickets:
        body = r.classify(t.text)
        latencies.append(body["latency_ms"])
        if body["label"] == t.label:
            hits += 1
        elif len(misses) < 3:
            misses.append({"ticket": t.text[:70], "expected": t.label, "got": body["label"]})
    acc = hits / len(tickets)
    lat = sorted(latencies)[len(latencies) // 2]
    text = (
        f"local router (no app off-platform; the promoted model loaded in-process from {current['path']}): "
        f"{hits}/{len(tickets)} correct ({acc:.0%}) on dataset {dataset}, p50 {lat:.0f} ms per ticket on {r.device}. "
        f"Serving ticket-router@{current['version']}: {r.factory.get('base', '?')} trained on dataset {r.dataset}."
    )
    if misses:
        text += "\nmisses: " + json.dumps(misses)
    return ("PASS" if acc >= 0.75 else "FAIL") + ": " + text


def reach(endpoint: str) -> str:
    """The URL a *task* uses to call an app.

    A task on a devbox only knows the app's public URL, `http://<app>-<project>-<domain>.localhost:30081`,
    which a pod cannot resolve. Inside a pod, swap it for the devbox's in-cluster service name.
    """
    from urllib.parse import urlparse

    host = urlparse(endpoint).hostname or ""
    if host.endswith(".localhost") and os.getenv("KUBERNETES_SERVICE_HOST"):
        return f"http://{host.removesuffix('.localhost')}.flyte.svc.cluster.local"
    return endpoint


def _looks_like_artifact(ref: str) -> bool:
    """Bare artifact names the model tends to pass: `ticket-router-<base>-ft`, `ticket-router@v…`."""
    name = ref.split("@", 1)[0]
    return name.startswith(FT_PREFIX) and name not in CANDIDATES


async def _materialize(model: str) -> str:
    """A candidate id passes through; an artifact reference or remote path is downloaded.

    Accepts `artifact:<name>@<version>`, and forgives the two shapes the model produces
    when it drops the prefix: `<name>@<version>` and a bare `<name>` (latest version).
    """
    if model in CANDIDATES or os.path.isdir(model):
        return model
    if model.startswith("artifact:") or _looks_like_artifact(model):
        from flyte.remote import Artifact

        name, _, version = model.removeprefix("artifact:").partition("@")
        art = Artifact.get(name, version=version or "latest")
        d = await art.to_python(Dir)
        return await d.download()
    if "://" in model:
        return await Dir.from_existing_remote(model).download()
    return model  # let factory.resolve produce the error


def _dataset(version: str) -> str:
    if version not in DATASET_VERSIONS:
        raise ValueError(f"unknown dataset {version!r}; known: {', '.join(DATASET_VERSIONS)}")
    return version


# ── Look ─────────────────────────────────────────────────────────────────────


@tool
@tools_env.task
async def list_candidates(dataset: str = "v1") -> str:
    """The open-weight base models the factory can evaluate or fine-tune, with size and license, and the dataset they are measured on."""
    _dataset(dataset)
    lines = [
        f"{k:16} {v['params']:>5}  {v['license']:11} {v['hf']}{'  [small enough for CPU]' if v.get('cpu_ok') else ''}\n{'':16} {v['notes']}"
        for k, v in CANDIDATES.items()
    ]
    lines.append(
        f"\nrun_eval and fine_tune use {GPU_ARTICLE}. Models marked [small enough for CPU] can also be evaluated with "
        "run_eval_cpu, which is what production looks like: the router app serves on a CPU pod. "
        f"The request's latency bar is measured on {GPU_ARTICLE}."
    )
    cats = DATASET_VERSIONS[dataset]
    lines.append(
        f"\nDataset {dataset}: {len(load_split('test', dataset))} held-out support tickets across {len(cats)} categories "
        f"({', '.join(cats)}). Train set: {len(load_split('train', dataset))} tickets, used by fine_tune."
    )
    return "\n".join(lines)


@tool
@tools_env.task
async def production_status() -> str:
    """What is in production right now: the ticket-router artifact versions (newest first) and the deployed app."""
    if _is_local():
        current = local_current()
        if not current:
            return (
                "IN PRODUCTION NOW: nothing. Local run: the local registry is empty; nothing has been promoted. "
                "Model references here are directory paths, as returned by fine_tune."
            )
        others = [v for v in _local_versions() if v["version"] != current["version"]]
        text = (
            f"IN PRODUCTION NOW: {current['path']}  (version {current['version']}; weights from {current['model']}; "
            f"measured on dataset {current['dataset']} when promoted: accuracy {current['accuracy']}, p50 {current['latency_p50_ms']} ms).\n"
            "Local run: no app; the support agent loads this directory in-process. Evaluate the live model by this exact path."
        )
        if others:
            text += "\nEarlier versions (rollback takes a version): " + ", ".join(v["version"] for v in others)
        return text
    from flyte.remote import App, Artifact

    if await registry() == "outputs":
        return await _production_status_from_app()

    lines = []
    try:
        versions = [a async for a in Artifact.listall.aio(name=ROUTER_ARTIFACT)]
        # listall makes no ordering promise; production is the most recently created version.
        versions.sort(key=lambda a: a.to_dict().get("createdAt", ""), reverse=True)
    except Exception:
        versions = []
    if not versions:
        lines.append(
            f"IN PRODUCTION NOW: nothing. The {ROUTER_ARTIFACT} artifact has no versions; nothing has been promoted."
        )
    else:

        def describe(v) -> str:
            attrs = (v.to_dict().get("spec", {}).get("info", {}) or {}).get("userMetadata", {}) or {}
            return (
                f"artifact:{ROUTER_ARTIFACT}@{v.version}  (weights from {attrs.get('source_model', '?')}; "
                f"measured on dataset {attrs.get('dataset', 'v1')} when promoted: accuracy {attrs.get('accuracy', '?')}, p50 {attrs.get('latency_p50_ms', '?')} ms)"
            )

        lines.append(f"IN PRODUCTION NOW: {describe(versions[0])}")
        lines.append(
            "This is the only version that is live. Evaluate it by this exact reference; the numbers above were measured at promotion time, on that dataset, and may not hold on newer data."
        )
        if len(versions) > 1:
            lines.append(f"History ({len(versions) - 1} earlier version(s), newest first; rollback targets):")
            for v in versions[1:5]:
                lines.append("  " + describe(v))
    try:
        app = App.get(APP_NAME)
        lines.append(f"app {APP_NAME}: {app.endpoint}")
    except Exception:
        lines.append(f"app {APP_NAME}: not deployed")
    return "\n".join(lines)


# ── Measure ──────────────────────────────────────────────────────────────────


async def _eval(model: str, n: int, dataset: str) -> str:
    _dataset(dataset)
    path = await _materialize(model)
    r = evaluate(path, n=n, version=dataset)
    r.model = model
    # The task's own report: the numbers, per category, and the mistakes. The agent only
    # sees the text below; you can open this in the Flyte UI afterwards.
    await flyte.report.replace.aio(evals_html([r.__dict__]))
    await flyte.report.flush.aio()
    text = r.summary() + f"\ndataset: {dataset}"
    if r.mistakes:
        text += "\nsample mistakes: " + json.dumps(r.mistakes[:4])
    return text


async def _production_status_from_app() -> str:
    """No artifact registry (a devbox): production is whatever the router app says it is serving."""
    import httpx
    from flyte.remote import App

    await flyte.init_in_cluster.aio()
    try:
        endpoint = reach(App.get(APP_NAME).endpoint)
    except Exception:  # noqa: BLE001  (no app yet)
        return f"IN PRODUCTION NOW: nothing. The {APP_NAME} app has not been deployed; nothing has been promoted."
    try:
        info = httpx.get(endpoint + "/", timeout=60).json()
    except Exception as e:  # noqa: BLE001
        return f"The {APP_NAME} app exists at {endpoint} but did not answer ({str(e)[:80]}). Treat production as unknown."
    r = info.get("router") or {}
    return (
        f"IN PRODUCTION NOW: {r.get('model', '?')}  (version {info.get('version', '?')}; "
        f"measured on dataset {info.get('dataset', 'v1')} when promoted: accuracy {r.get('accuracy', '?')}, "
        f"p50 {r.get('latency_p50_ms', '?')} ms; served from {info.get('model_dir')} on the app's {info.get('device', '?')}).\n"
        "This platform has no artifact registry: models are referenced by the storage path of the task output "
        "that produced them, as printed by fine_tune and promote. Evaluate the live model by that exact reference."
    )


@tool
@gpu_env.task(retries=2, report=True)
async def run_eval(model: str, n: int = 120, dataset: str = "v1") -> str:
    """Classify n held-out tickets with a model (a candidate id, or a model reference returned by fine_tune) on a dataset version, on the factory's GPU, and measure accuracy plus per-ticket latency. This is the latency the request is judged on."""
    return await _eval(model, n, dataset)


@tool
@cpu_env.task(retries=2, report=True)
async def run_eval_cpu(model: str, n: int = 120, dataset: str = "v1") -> str:
    """Same as run_eval but on a small CPU node: no GPU needed, cheaper, slower per ticket. Sensible for models marked small enough for CPU. Latency here is CPU latency, not the GPU number the request asks for."""
    return await _eval(model, n, dataset)


# ── Build ────────────────────────────────────────────────────────────────────


def _live_training_report(model: str, epochs: float, lora_r: int, dataset: str):
    """A Trainer callback that redraws the task report every logging step: a live loss curve."""
    from transformers import TrainerCallback

    from factory import loss_history

    class _Report(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                flyte.report.replace(
                    training_html(
                        model, epochs, lora_r, dataset, loss_history(state.log_history), total_steps=state.max_steps
                    )
                )
                flyte.report.flush()

    return _Report()


async def _train(model: str, epochs: float, lora_r: int, dataset: str) -> tuple[Dir, str]:
    _dataset(dataset)
    path = await _materialize(model)
    base = model if model in CANDIDATES else os.path.basename(path.rstrip("/"))
    local_out = os.path.join(tempfile.mkdtemp(prefix="factory-"), f"{base}-ft")
    max_steps = int(os.environ.get("FACTORY_MAX_STEPS", "-1"))  # deployment tests only
    live = [] if _is_local() else [_live_training_report(model, epochs, lora_r, dataset)]
    r = _fine_tune(path, local_out, epochs=epochs, max_steps=max_steps, lora_r=lora_r, version=dataset, callbacks=live)
    if _is_local():
        # One process for the whole run, and training seeds Python's `random`, so two
        # fine-tunes would draw the same "random" output path from the SDK. Name it ourselves.
        dest = os.path.join(tempfile.mkdtemp(prefix="factory-weights-"), f"{base}-ft")
        weights = await Dir.from_local(local_out, remote_destination=dest)
        # No registry and no upload endpoint off-platform: the directory is the output.
        summary = r.summary()
        await flyte.report.replace.aio(
            training_html(
                model, epochs, lora_r, dataset, r.history or [], summary=summary, seconds=r.seconds, device=r.device
            )
        )
        await flyte.report.flush.aio()
        return weights, summary
    weights = await Dir.from_local(local_out)
    name = f"{FT_PREFIX}-{base}-ft"
    version = f"{dataset}-{epochs:g}ep-r{lora_r}-{int(time.time())}"
    card = await artifacts.Card.create_from.aio(
        content=(
            f"# {name}\n\nLoRA fine-tune of `{model}` on {len(load_split('train', dataset))} synthetic "
            f"support tickets, dataset {dataset} ({', '.join(DATASET_VERSIONS[dataset])}).\n\n"
            f"- epochs: {epochs:g}\n- lora_r: {lora_r}\n- steps: {r.steps}\n"
            f"- train loss: {r.train_loss_start:.2f} -> {r.train_loss_end:.2f}\n- trained in {r.seconds:.0f}s on {r.device}\n"
        ),
        format="md",
        card_type="model",
    )
    meta = artifacts.Metadata.create_model_metadata(
        name=name,
        version=version,
        description=f"Ticket router fine-tuned from {model} on dataset {dataset}",
        framework="transformers",
        architecture=CANDIDATES.get(base, {}).get("hf", base),
        task="text-classification",
        serial_format="safetensors",
        card=card,
        attrs={"base_model": model, "dataset": dataset, "epochs": f"{epochs:g}", "lora_r": str(lora_r)},
    )
    summary = f"{r.summary()}\nartifact: {name}@{version}"
    # The final report: the whole loss curve, the numbers, the artifact reference.
    await flyte.report.replace.aio(
        training_html(
            model, epochs, lora_r, dataset, r.history or [], summary=summary, seconds=r.seconds, device=r.device
        )
    )
    await flyte.report.flush.aio()
    return artifacts.new(weights, meta), summary


def _data_fingerprint(dataset: str) -> str:
    """A hash of the training tickets, so the cache key covers the data and not only its name."""
    import hashlib

    rows = sorted(f"{t.text}\t{t.label}" for t in load_split("train", dataset))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


@gpu_env.task(cache="auto", retries=1, produces_artifacts=True, report=True)
async def train_model(
    model: str, epochs: float = 1.0, lora_r: int = 16, dataset: str = "v1", data: str = ""
) -> tuple[Dir, str]:
    """Train a candidate on the factory's GPU and register the weights as a model artifact.

    `data` is the fingerprint of the training tickets. It is not used here; it is an input
    so that it is part of the cache key: edit the tickets and the fine-tune is retrained.
    """
    return await _train(model, epochs, lora_r, dataset)


@tool
@tools_env.task(retries=1)
async def fine_tune(model: str, epochs: float = 1.0, lora_r: int = 16, dataset: str = "v1") -> str:
    """Fine-tune a model on the training tickets of a dataset version, on the factory's GPU, and register the result as a model artifact. Chat models get LoRA (about a minute per epoch); the encoder gets a full fine-tune of its untrained head (seconds per epoch, so give it 2-3). `epochs` and `lora_r` are yours to choose. Returns a model reference (artifact:<name>@<version>, or a storage path on a platform without a registry) to pass to run_eval and promote."""
    short = model.split("@")[0].removeprefix("artifact:ticket-router-") if model.startswith("artifact:") else model
    named = dataclasses.replace(
        train_model, short_name=f"train_model · {short} · {epochs:g}ep" + (f" · {dataset}" if dataset != "v1" else "")
    )
    weights, summary = await named(model, epochs, lora_r, dataset, _data_fingerprint(dataset))
    if await registry() == "outputs":
        # No artifact registry: the model is the output directory (local, or in the devbox's
        # object store); its path is the reference.
        return summary.split("\nartifact:")[0] + f"\nnew model reference: {weights.path}"
    name_version = summary.rsplit("artifact: ", 1)[-1].strip()
    return summary.split("\nartifact:")[0] + f"\nnew model reference: artifact:{name_version}"


# ── Ship ─────────────────────────────────────────────────────────────────────


@tools_env.task(produces_artifacts=True)
async def publish_router(
    model: str, accuracy: float, latency_p50_ms: float, reason: str, dataset: str
) -> tuple[Dir, str]:
    """Publish a model as a new version of the production `ticket-router` artifact.

    A task output wrapped with `artifacts.new` is the supported way to register a version
    from inside a run: Union stamps the producing action on it, so the artifact's lineage
    view leads back to this promotion, and from there to the fine-tune that made the weights.
    """
    local = await _materialize(model)
    version = f"v{int(time.time())}"
    # The app reads this at startup and reports it on `/`, so a deploy can wait for *this*
    # version to be the one answering. The base model name is not enough: a retrain of
    # the same base looks identical to the pod it is replacing.
    entry = {
        "model": model,
        "accuracy": accuracy,
        "latency_p50_ms": latency_p50_ms,
        "reason": reason,
        "dataset": dataset,
        "version": version,
    }
    with open(os.path.join(local, "router.json"), "w") as f:
        json.dump(entry, f)
    weights = await Dir.from_local(local)
    card = artifacts.Card.create_from(
        content=f"# ticket-router {version}\n\n{reason}\n\n```json\n{json.dumps(entry, indent=2)}\n```",
        format="md",
        card_type="model",
    )
    meta = artifacts.Metadata.create_model_metadata(
        name=ROUTER_ARTIFACT,
        version=version,
        description=f"Production ticket router: {model}",
        framework="transformers",
        architecture="causal-lm classifier",
        task="text-classification",
        serial_format="safetensors",
        card=card,
    )
    # Metadata is frozen; add the decision's numbers as searchable attrs on a copy.
    meta = dataclasses.replace(
        meta,
        attrs={
            **(meta.attrs or {}),
            "source_model": model,
            "accuracy": f"{accuracy:.4f}",
            "latency_p50_ms": f"{latency_p50_ms:.1f}",
            "dataset": dataset,
        },
    )
    # Return the version too: "latest" is not a safe way to find out what you just
    # published when other runs may be promoting at the same time.
    return artifacts.new(weights, meta), version


async def _ask_for_approval(model: str, accuracy: float, latency_p50_ms: float, reason: str) -> bool:
    """Pause the run until a person approves the deployment in the Flyte UI (or the CLI)."""
    condition = await flyte.new_condition.aio(
        "deploy-approval",
        prompt=(
            "## Deploy to production?\n\n"
            f"The agent wants to promote **{model}**.\n\n"
            f"| accuracy | p50 latency |\n|---|---|\n| {accuracy:.1%} | {latency_p50_ms:.0f} ms |\n\n"
            f"{reason}\n\n"
            "Approve to publish the artifact and deploy the app."
        ),
        prompt_type="markdown",
        data_type=bool,
    )
    return bool(await condition.wait.aio())


@tool
@tools_env.task
async def promote(model: str, accuracy: float, latency_p50_ms: float, reason: str, dataset: str = "v1") -> str:
    """Promote one model to production: publish it as the `ticket-router` artifact and deploy the serving app that mounts it. Call this once, at the end, with the numbers you measured. Follow it with test_deployment."""
    _dataset(dataset)
    if _is_local():
        # No platform: copy the winner into the local registry and make it current. The
        # support agent and test_deployment load it from there, in-process.
        entry = _local_publish(await _materialize(model), accuracy, latency_p50_ms, reason, dataset)
        return (
            f"promoted {model} as ticket-router {entry['version']} "
            f"(accuracy {accuracy:.1%}, p50 {latency_p50_ms:.0f} ms). Local run: no app to deploy; the model is in "
            f"production at {entry['path']} and the support agent loads it in-process. "
            "Run test_deployment to check it before you finish."
        )

    from flyte.remote import Artifact

    # 0. Optionally, a person has to say yes first. The run pauses here until they do.
    if FACTORY_APPROVAL and not await _ask_for_approval(model, accuracy, latency_p50_ms, reason):
        return f"promotion of {model} was DECLINED by the approver. Nothing was published or deployed."

    if await registry() == "outputs":
        # No artifact registry (a devbox): the promoted weights are a task output, and the
        # app mounts that output directory. Same files, no versions to roll back through.
        weights, published = await publish_router(model, accuracy, latency_p50_ms, reason, dataset)
        text = (
            f"promoted {model} as ticket-router {published} (accuracy {accuracy:.1%}, p50 {latency_p50_ms:.0f} ms); "
            f"weights at {weights.path}."
        )
        if not FACTORY_DEPLOY:
            return text + " Deployment skipped (FACTORY_DEPLOY=0)."
        url = await _deploy(make_router_app(published, base=_base_of(model), weights=weights), expect_version=published)
        return text + f" Deployed the {APP_NAME} app: {url}. Run test_deployment to check it before you finish."

    # Remember what was in production before, so rollback has somewhere to go.
    try:
        previous = Artifact.get(ROUTER_ARTIFACT).version
    except Exception:
        previous = None

    # 1. The winner becomes a new version of the production artifact, with the decision on it.
    _, published = await publish_router(model, accuracy, latency_p50_ms, reason, dataset)
    text = f"promoted {model} as {ROUTER_ARTIFACT}@{published} (accuracy {accuracy:.1%}, p50 {latency_p50_ms:.0f} ms)."
    if previous:
        text += f" Previous production version: {previous} (use rollback if the deployment test fails)."

    # 2. Ship it: deploy the app that mounts the latest ticket-router artifact.
    if not FACTORY_DEPLOY:
        return text + " Deployment skipped (FACTORY_DEPLOY=0)."
    url = await _deploy(make_router_app(published, base=_base_of(model)), expect_version=published)
    return text + f" Deployed the {APP_NAME} app: {url}. Run test_deployment to check it before you finish."


def _base_of(model: str) -> str | None:
    """`artifact:ticket-router-modernbert-base-ft@…` -> `modernbert-base`; a candidate id -> itself."""
    if model in CANDIDATES:
        return model
    name = model.split("@")[0].removeprefix("artifact:")
    if name.startswith(f"{FT_PREFIX}-") and name.endswith("-ft"):
        return name[len(FT_PREFIX) + 1 : -3]
    return None


def _host_builder():
    """Build the app's image where this task runs, when that is a Mac.

    Inside a pod the SDK uses the platform's remote image builder, which a devbox does not
    have. On an Apple Silicon devbox this task runs natively on the Mac, where Docker and
    the devbox registry are, so the local builder applies; the image was already built for
    the GPU tasks, so this is a registry check, not a build. Anywhere else: unchanged.
    """
    import contextlib
    import platform

    if platform.system() != "Darwin":
        return contextlib.nullcontext()
    from flyte._initialize import get_init_config, init_config_context

    # The host process gets a minimal PATH; Docker Desktop's CLI, which the local builder
    # uses to check the registry, lives in /usr/local/bin.
    if "/usr/local/bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + ":/usr/local/bin"
    # The devbox registry is `localhost:30000`; the in-cluster config does not say so, and
    # without it the image is looked up under a bare name and never found.
    return init_config_context(get_init_config().replace(image_builder="local", image_registry="localhost:30000"))


async def _deploy(app_env, expect_version: str, timeout_s: int = 900) -> str:
    """Deploy the router app and wait, with a deadline, until the new revision is the one answering.

    `serve()` returns as soon as the app is active, and an app with an older revision
    already running is active before the new pod has even scheduled. So we watch the
    app's own status until the revision we just created reports ready (or fails, which
    surfaces "Insufficient memory" and friends as a tool error the agent can read), and
    only then ask `/` to confirm the artifact version. Polling `/` during the rollout
    would wake the old pod on a scaled-to-zero app and slow the new one down.
    """
    import httpx
    from flyte.remote import App
    from google.protobuf.json_format import MessageToDict

    await flyte.init_in_cluster.aio()
    try:
        before = App.get(APP_NAME).revision
    except Exception:  # noqa: BLE001  (first deploy: no app yet)
        before = 0
    try:
        with _host_builder():
            handle = await asyncio.wait_for(
                flyte.with_servecontext(interactive_mode=False).serve.aio(app_env), timeout=timeout_s
            )
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"app {APP_NAME} was deployed but did not become active within {timeout_s}s; "
            "check its logs in the Union UI, then rollback to the previous version or fix and promote again"
        ) from None

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        conds = MessageToDict(App.get(APP_NAME).pb2).get("status", {}).get("conditions", [])
        if conds and not any("revision" in c for c in conds):
            # A devbox reports plain ACTIVE conditions with no revision history. The version
            # check below is the proof that the new pod is the one answering.
            if any(c.get("deploymentStatus") == "DEPLOYMENT_STATUS_ACTIVE" for c in conds):
                break
            await asyncio.sleep(10)
            continue
        new = [c for c in conds if int(c.get("revision", 0)) > before]
        if new:
            latest_id = max(new, key=lambda c: int(c.get("revision", 0))).get("deploymentId")
            mine = [c for c in new if c.get("deploymentId") == latest_id]
            failed = [
                c
                for c in mine
                if c.get("deploymentStatus") == "DEPLOYMENT_STATUS_FAILED" or "RevisionFailed" in c.get("message", "")
            ]
            if failed:
                raise RuntimeError(
                    f"app {APP_NAME} could not deploy {expect_version}: {failed[-1].get('message', '')[:400]}"
                )
            if any(
                c.get("deploymentStatus") == "DEPLOYMENT_STATUS_ACTIVE" and c.get("substate") in ("RUNNING", "")
                for c in mine
            ):
                break
        await asyncio.sleep(10)
    else:
        raise RuntimeError(
            f"app {APP_NAME} did not finish rolling out {expect_version} within {timeout_s}s. "
            "Check the app's revision log in the Union UI, then rollback or fix and promote again"
        )

    endpoint = reach(App.get(APP_NAME).endpoint)
    for _ in range(12):  # the new revision is ready; confirm it is the one answering
        try:
            info = httpx.get(endpoint + "/", timeout=60).json()
            if info.get("load_error"):
                raise RuntimeError(f"app {APP_NAME} started but its model failed to load: {info['load_error']}")
            if info.get("version") == expect_version and info.get("loaded", True):
                return handle.url
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(10)
    raise RuntimeError(
        f"app {APP_NAME} rolled out but still answers as a different version than {expect_version}; "
        "check the app in the Union UI, then rollback or promote again"
    )


async def _wait_for_app(timeout_s: int = 300) -> str:
    """The app scales from zero and loads a model; poll until it answers."""
    import httpx
    from flyte.remote import App

    endpoint = reach(App.get(APP_NAME).endpoint)
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            r = httpx.get(endpoint + "/", timeout=30)
            if r.status_code == 200 and "model_dir" in r.text:
                return endpoint
            last = f"{r.status_code} {r.text[:80]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)[:80]
        await asyncio.sleep(10)
    raise RuntimeError(f"app {APP_NAME} did not become ready within {timeout_s}s (last: {last})")


@tool
@tools_env.task
async def test_deployment(n: int = 12, dataset: str = "v1") -> str:
    """Send n held-out tickets to the LIVE ticket-router app and report its accuracy and latency. Call this after promote. If it fails, use rollback."""
    _dataset(dataset)
    if _is_local():
        return _local_test(n, dataset)
    import httpx

    await flyte.init_in_cluster.aio()
    endpoint = await _wait_for_app()
    info = httpx.get(endpoint + "/", timeout=30).json()
    tickets = load_split("test", dataset)[-max(1, min(n, 40)) :]  # the tail, so it is not the rows the eval prints
    hits, latencies, misses = 0, [], []
    async with httpx.AsyncClient(timeout=120) as client:
        for t in tickets:
            r = await client.post(endpoint + "/classify", json={"text": t.text})
            r.raise_for_status()
            body = r.json()
            latencies.append(body["latency_ms"])
            if body["label"] == t.label:
                hits += 1
            elif len(misses) < 3:
                misses.append({"ticket": t.text[:70], "expected": t.label, "got": body["label"]})
    acc = hits / len(tickets)
    device = (httpx.get(endpoint + "/", timeout=30).json().get("device")) or "cpu"
    lat = sorted(latencies)[len(latencies) // 2]
    text = (
        f"live app {APP_NAME} at {endpoint}: {hits}/{len(tickets)} correct ({acc:.0%}) on dataset {dataset}, "
        f"p50 {lat:.0f} ms per ticket on the app's {device} pod{f' (a CPU pod is not comparable to the {GPU_LABEL} eval)' if device == 'cpu' else ''}. "
        f"The app is serving {ROUTER_ARTIFACT}@{info.get('version', '?')}: "
        f"{(info.get('factory') or {}).get('base', '?')} trained on dataset {info.get('dataset')}."
    )
    if misses:
        text += "\nmisses: " + json.dumps(misses)
    verdict = "PASS" if acc >= 0.75 else "FAIL"
    return f"{verdict}: " + text


@tool
@tools_env.task
async def rollback(version: str) -> str:
    """Roll production back to an earlier `ticket-router` artifact version (from production_status or promote's reply): the old weights are republished as the newest version, so production is always the latest version, and the app is redeployed. Use when a deployment test fails."""
    if _is_local():
        match = [v for v in _local_versions() if v["version"] == version]
        if not match:
            return f"no local version {version!r}; production_status lists the ones there are."
        old = match[0]
        entry = _local_publish(
            old["path"], old["accuracy"], old["latency_p50_ms"], f"Rollback to {version}.", old["dataset"]
        )
        return f"rolled back: {version} republished as {entry['version']} and made current ({entry['path']}). Run test_deployment to confirm."
    if await registry() == "outputs":
        return (
            "This platform has no artifact registry, so there is no version history to roll back through. "
            "To go back, call promote again with the earlier model reference "
            "(the storage path fine_tune returned for it)."
        )
    from flyte.remote import Artifact

    old = Artifact.get(ROUTER_ARTIFACT, version=version)  # raises if it does not exist
    attrs = (old.to_dict().get("spec", {}).get("info", {}) or {}).get("userMetadata", {}) or {}
    # A rollback of a rollback points at another ticket-router version; follow the chain
    # to the fine-tune artifact so the deploy knows which base model to wait for.
    source = str(attrs.get("source_model", ""))
    for _ in range(10):
        if not source.startswith(f"artifact:{ROUTER_ARTIFACT}@"):
            break
        prev = Artifact.get(ROUTER_ARTIFACT, version=source.split("@", 1)[1])
        source = str(
            ((prev.to_dict().get("spec", {}).get("info", {}) or {}).get("userMetadata", {}) or {}).get(
                "source_model", ""
            )
        )
    # Production is defined as the latest version of the artifact. A rollback therefore
    # publishes a new version whose weights are the old ones, with the reason on its card,
    # rather than quietly pointing the app at history. The lineage view shows the loop.
    _, published = await publish_router(
        f"artifact:{ROUTER_ARTIFACT}@{version}",
        float(attrs.get("accuracy", 0) or 0),
        float(attrs.get("latency_p50_ms", 0) or 0),
        f"Rollback to {ROUTER_ARTIFACT}@{version} (source {attrs.get('source_model', '?')}).",
        attrs.get("dataset", "v1"),
    )
    url = await _deploy(make_router_app(published, base=_base_of(source)), expect_version=published)
    return (
        f"rolled back: republished {ROUTER_ARTIFACT}@{version} as {ROUTER_ARTIFACT}@{published}; "
        f"{APP_NAME} now serves it: {url}. Run test_deployment to confirm."
    )


TOOLS = [list_candidates, production_status, run_eval, run_eval_cpu, fine_tune, promote, test_deployment, rollback]
