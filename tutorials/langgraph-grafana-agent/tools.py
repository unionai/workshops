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

Two tools are cached (`cache="auto"`): the same eval of the same model is the same
answer, so a second call by anyone in the project returns in a second instead of a
minute. In a room full of people running the same agent, the first person pays for each
cell of the eval matrix and everyone else hits cache.
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

from config import FACTORY_APPROVAL, FACTORY_DEPLOY, cpu_env, gpu_env, tagged, tools_env
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
        "\nrun_eval and fine_tune use a T4. Models marked [small enough for CPU] can also be evaluated with "
        "run_eval_cpu, which is what production looks like: the router app serves on a CPU pod. "
        "The request's latency bar is measured on the T4."
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
        return "Local run: no artifact registry or app off-cluster. Nothing is in production."
    from flyte.remote import App, Artifact

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


@tool
@gpu_env.task(cache="auto", retries=2, report=True)
async def run_eval(model: str, n: int = 120, dataset: str = "v1") -> str:
    """Classify n held-out tickets with a model (a candidate id, or an artifact:<name>@<version> reference) on a dataset version, on a T4, and measure accuracy plus per-ticket latency. This is the latency the request is judged on."""
    return await _eval(model, n, dataset)


@tool
@cpu_env.task(cache="auto", retries=2, report=True)
async def run_eval_cpu(model: str, n: int = 120, dataset: str = "v1") -> str:
    """Same as run_eval but on a small CPU node: no GPU needed, cheaper, slower per ticket. Sensible for models marked small enough for CPU. Latency here is CPU latency, not the T4 number the request asks for."""
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
    weights = await Dir.from_local(local_out)
    name = f"{FT_PREFIX}-{base}-ft"
    version = f"{dataset}-{epochs:g}ep-r{lora_r}-{int(time.time())}"
    card = artifacts.Card.create_from(
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


@gpu_env.task(cache="auto", retries=1, produces_artifacts=True, report=True)
async def train_model(model: str, epochs: float = 1.0, lora_r: int = 16, dataset: str = "v1") -> tuple[Dir, str]:
    """Train a candidate on a T4 and register the weights as a model artifact."""
    return await _train(model, epochs, lora_r, dataset)


@tool
@tools_env.task(retries=1)
async def fine_tune(model: str, epochs: float = 1.0, lora_r: int = 16, dataset: str = "v1") -> str:
    """Fine-tune a model on the training tickets of a dataset version, on a T4, and register the result as a model artifact. Chat models get LoRA (about a minute per epoch); the encoder gets a full fine-tune of its untrained head (seconds per epoch, so give it 2-3). `epochs` and `lora_r` are yours to choose. Returns an artifact:<name>@<version> reference to pass to run_eval and promote."""
    short = model.split("@")[0].removeprefix("artifact:ticket-router-") if model.startswith("artifact:") else model
    named = dataclasses.replace(
        train_model, short_name=f"train_model · {short} · {epochs:g}ep" + (f" · {dataset}" if dataset != "v1" else "")
    )
    weights, summary = await named(model, epochs, lora_r, dataset)
    if _is_local():
        # No artifact registry off-cluster: the model is a local directory, pass its path.
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
    weights = await Dir.from_local(local)
    version = f"v{int(time.time())}"
    entry = {
        "model": model,
        "accuracy": accuracy,
        "latency_p50_ms": latency_p50_ms,
        "reason": reason,
        "dataset": dataset,
        "version": version,
    }
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
        entry = {
            "model": model,
            "accuracy": accuracy,
            "latency_p50_ms": latency_p50_ms,
            "reason": reason,
            "dataset": dataset,
        }
        path = os.path.join(tempfile.mkdtemp(prefix="registry-"), "ticket-router.json")
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)
        return f"promoted {model} (accuracy {accuracy:.1%}, p50 {latency_p50_ms:.0f} ms). Local run: registry written to {path}; no artifact or app off-cluster."

    from flyte.remote import Artifact

    # 0. Optionally, a person has to say yes first. The run pauses here until they do.
    if FACTORY_APPROVAL and not await _ask_for_approval(model, accuracy, latency_p50_ms, reason):
        return f"promotion of {model} was DECLINED by the approver. Nothing was published or deployed."

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
    url = await _deploy(make_router_app(published, base=_base_of(model)), expect_base=_base_of(model))
    return text + f" Deployed the {APP_NAME} app: {url}. Run test_deployment to check it before you finish."


def _base_of(model: str) -> str | None:
    """`artifact:ticket-router-modernbert-base-ft@…` -> `modernbert-base`; a candidate id -> itself."""
    if model in CANDIDATES:
        return model
    name = model.split("@")[0].removeprefix("artifact:")
    if name.startswith(f"{FT_PREFIX}-") and name.endswith("-ft"):
        return name[len(FT_PREFIX) + 1 : -3]
    return None


async def _deploy(app_env, expect_base: str | None = None, timeout_s: int = 600) -> str:
    """Deploy the router app and wait, with a deadline, until the new revision is the one answering.

    On a cluster `serve()` blocks until the app reports activated, and a revision that
    crashes on startup never does; a deadline turns that into a tool error the agent can
    read (and roll back from). And "activated" is not "switched over": the previous pod
    keeps answering during the rollout, so we also poll `/` until it reports the model we
    just deployed (by base name), which is what test_deployment will measure.
    """
    import httpx

    await flyte.init_in_cluster.aio()
    try:
        handle = await asyncio.wait_for(
            flyte.with_servecontext(interactive_mode=False).serve.aio(app_env), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"app {APP_NAME} was deployed but did not become healthy within {timeout_s}s; "
            "check its logs in the Union UI, then rollback to the previous version or fix and promote again"
        ) from None
    if expect_base:
        from flyte.remote import App

        endpoint = App.get(APP_NAME).endpoint
        deadline, seen = time.time() + timeout_s, 0
        while time.time() < deadline:
            try:
                info = httpx.get(endpoint + "/", timeout=30).json()
                seen = seen + 1 if (info.get("factory") or {}).get("base") == expect_base else 0
                if seen >= 3:  # three consecutive answers from the new revision
                    break
            except Exception:  # noqa: BLE001
                seen = 0
            await asyncio.sleep(5)
    return handle.url


async def _wait_for_app(timeout_s: int = 300) -> str:
    """The app scales from zero and loads a model; poll until it answers."""
    import httpx
    from flyte.remote import App

    endpoint = App.get(APP_NAME).endpoint
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
        return "Local run: no app to test off-cluster. Skipping."
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
    lat = sorted(latencies)[len(latencies) // 2]
    text = (
        f"live app {APP_NAME} at {endpoint}: {hits}/{len(tickets)} correct ({acc:.0%}) on dataset {dataset}, "
        f"p50 {lat:.0f} ms per ticket on the app's {info.get('device', 'cpu')} pod (a CPU pod is not comparable to the T4 eval). "
        f"The app is serving a model trained on dataset {info.get('dataset')} from {(info.get('factory') or {}).get('base', '?')}."
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
        return "Local run: nothing to roll back off-cluster."
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
    url = await _deploy(make_router_app(published, base=_base_of(source)), expect_base=_base_of(source))
    return (
        f"rolled back: republished {ROUTER_ARTIFACT}@{version} as {ROUTER_ARTIFACT}@{published}; "
        f"{APP_NAME} now serves it: {url}. Run test_deployment to confirm."
    )


TOOLS = [list_candidates, production_status, run_eval, run_eval_cpu, fine_tune, promote, test_deployment, rollback]
