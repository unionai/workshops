# Batch inference on a spot GPU market that keeps going when nodes come and go

Some GPU providers sell whole 8-GPU nodes on a market: you set a limit price, a node is granted while the market price
is under it, and the node is taken back when the price goes over. That is cheap capacity for work that can wait, as
long as the work survives losing a node at any moment.

This tutorial runs that kind of work: **gpt-oss-120b writes a synthetic coding dataset**, thousands of answers to
prompts like "write a Rust function that merges overlapping intervals and include unit tests". Every prompt is
independent and nobody waits on a single answer, so the only goal is to finish the whole set at the lowest cost. The
run goes faster when more nodes are granted, waits when there are none, and never regenerates finished work.

```bash
uv run gpu_probe.py                                                  # can a whole node be scheduled? reports its 8 GPUs
uv run prefetch.py openai/gpt-oss-120b                               # weights into the GPU pool's own object store, once
uv run batch_inference.py --model <model dir> --num-prompts 2000     # the batch, with a live report
uv run batch_inference.py --model <model dir> --crash-shard 3        # the same, killing a replica mid-shard
```

Adapted from a draft tutorial in unionai-examples (#352), with the fixes it needed to run on a GPU.

## What happens when it runs

1. **The driver** (`batch`, a CPU pod) makes the prompts from templates (16 tasks × 4 languages × 4 styles, so the run
   can be any size), writes them as shards of 256, and calls `infer_shard` once per shard.
2. **The GPU pool** (`infer_shard`) is a *reusable* task environment: long-running containers that Flyte keeps warm
   between calls. One replica is one whole node, and the pool scales from 1 to `MAX_NODES` replicas with the work and
   the nodes the market grants. Up to four shards share a replica, so its server always has requests queued.
3. **Each replica starts one vLLM server** the first time a shard lands on it, with eight engines on the node's eight
   GPUs (`--data-parallel-size 8`; a model that needs the whole node, such as Kimi K3, runs with
   `--tensor-parallel 8`). The weights stream from the pool's object store into GPU memory through the
   `flyteplugins-vllm` model streamer, so nothing is downloaded to the node's disk first.
4. **A shard sends its prompts in sub-batches of 128, and each sub-batch is a `@flyte.trace`.** Flyte records each
   finished sub-batch, so when a shard is retried it replays the finished ones from the record instead of generating
   them again, and only does the rest.
5. **When a replica dies** (a node taken back, an OOM kill, or `--crash-shard N`), the shards it was running fail
   with "Executor process terminated", and each is retried (up to 10 times) on whichever replica is free. The pool
   starts a replacement, which loads the model again. Finished shards are cached and never rerun.
6. **When every node is gone**, the shards wait in the queue for resources. The driver keeps the results of every
   finished shard and its report keeps refreshing, so a pause is visible. When nodes come back, the run finishes.

`--crash-shard N` kills the replica halfway through shard N on its first attempt: the retry replays the first half of
that shard from its traces. A reusable container's `FLYTE_ATTEMPT_NUMBER` is set when the container starts, not per
task, so the shard counts its own attempts in the run's storage.

## The live report

The driver's report refreshes as shards finish, and every 30 seconds otherwise, so a pause shows up even when nothing
completes. It is the screen to watch during a demo:

- shards done, prompts done, elapsed time
- tokens per second overall, and over the last two minutes: it falls when a node is taken back and recovers when one returns
- active replicas: replicas that finished a shard in the last two minutes; zero while every node is taken back
- retried shards: attempts, how many sub-batches were replayed from traces, and which replica finished each one
- the median model load time, which decides how long a returning node takes to contribute
- **allocated compute**: one row per granted node, as the shards saw it (its name, its replica, its GPUs, GPU memory
  used of total, shards done, when it started and whether it is still active), plus nodes in use, GPUs allocated,
  node-hours used and, with `PRICE_PER_NODE_HOUR` set, an estimated cost. Grants and reclaims happen outside the run,
  so they show indirectly: a node appears when the market grants it and goes quiet when it is taken back. The node
  name comes from the pod template (`NODE_NAME` from `spec.nodeName`), the GPU memory from the kernel's
  `/sys/class/drm/card*/device/mem_info_vram_*`

## Setup

**The GPU pool is its own cluster pool.** A cluster pool is a separate data plane (object store, secret store and
image registry), and pools are isolation boundaries. Three consequences shape this tutorial:

| | What it means here |
|---|---|
| A run cannot span pools | A child task on another pool's queue is refused ("action crosses cluster pool boundary"), so the whole run, driver included, goes to the GPU queue: every script submits with `flyte.with_runcontext(queue=GPU_QUEUE)` |
| Weights must be in that pool's store | Its nodes cannot read the default pool's bucket, and `flyte prefetch` has no queue option, so `prefetch.py` runs the same prefetch task on the GPU queue. It still records the model artifact |
| Images must be pullable from its nodes | If the pool has no image registry of its own, use public images (below), or set `IMAGE_REGISTRY` and `IMAGE_REGISTRY_SECRET` to a registry the default builder can push to and the nodes can pull from |

**Whole nodes are requested in a pod template.** The market admits only pods that ask for a multiple of 8
`amd.com/gpu`, on the container, with no node selector or toleration (the queue adds them). Asking through
`flyte.Resources(gpu=flyte.Device(device_class="AMD_GPU", ...))` makes Union add a cloud accelerator affinity these
nodes never match, so the task waits forever. `common.full_node_pod_template()` builds the request; it needs the
`kubernetes` client wherever task specs are built, which includes the driver pod, so the images carry it.

**The worker image must not keep vLLM's entrypoint.** The stock vLLM images start `vllm serve` as their entrypoint.
Union starts a reusable worker by passing its launcher as the container arguments, so with that entrypoint vLLM reads
the launcher as a model name and every replica exits with code 2. Flyte's image builder cannot clear an entrypoint, and
a `command` in the pod template is dropped, so the worker image is built from `docker/worker-rocm.Dockerfile`, which
clears it. It also installs into the ROCm image's system Python, which is marked externally managed (PEP 668).

```bash
docker login -u <you>
REGISTRY=docker.io/<you> ./docker/build-public.sh        # builds for linux/amd64, pushes, prints the two URIs
export VLLM_IMAGE=docker.io/<you>/batch-llm-vllm-rocm:<tag> DRIVER_IMAGE=docker.io/<you>/batch-llm-driver:<tag>
```

With `VLLM_IMAGE` and `DRIVER_IMAGE` set, the scripts build nothing. Without them, each script builds its images
without a queue first (on the default builder) and then submits to the GPU queue with the URIs.

| Variable | What it is |
|---|---|
| `GPU_QUEUE` | the queue that routes to the GPU pool |
| `VLLM_IMAGE`, `DRIVER_IMAGE` | prebuilt worker and driver images the GPU nodes can pull |
| `MAX_NODES` | the most replicas (whole nodes) the pool may hold, default 2 |
| `IMAGE_REGISTRY`, `IMAGE_REGISTRY_SECRET` | only when building the images with Flyte's builder instead |

**A node needs a limit price.** Someone with access to the market's console sets a limit price or capacity for the
cluster, so GPU pods get a node. A grant is a whole node with a minimum billing period, so batch tests into one grant:
replicas stay warm for 15 minutes when idle (`idle_ttl=900`), and a second run inside that window reuses them.

**The console may not show inputs, outputs or logs** for runs on a separate pool, because it reads them from the
default pool's bucket. The driver's report is the view of the run until that is fixed.

## Run a demo

1. Set the limit price above market and wait for two nodes; `uv run gpu_probe.py` should report 8 GPUs.
2. Warm up with `--num-prompts 256`, and note the model load time and tokens per second.
3. Size the real run to last about 15 minutes from those numbers (`--num-prompts`, `--max-tokens`).
4. Start it with `--crash-shard 3`. Shard 3's replica exits halfway; its shards retry, replay their finished
   sub-batches and carry on.
5. Lower the limit price so one node is taken back: its shards retry on the other node and tokens per second halves.
6. Lower it below both: the shards wait, the report shows zero active replicas, and nothing finished is lost.
7. Raise it: nodes return, each new replica loads the model, and the remaining shards finish.

Each crash or preemption uses one of a shard's 10 retries, and the wait after nodes return is the model load time,
which is what decides how large a model the demo can use.

## Files

| File | What it does |
|---|---|
| `batch_inference.py` | prompts, the reusable vLLM worker (`infer_shard`), traced sub-batches, the driver and its live report |
| `common.py` | the queue, the registry settings, the whole-node pod template, and building images without the queue |
| `gpu_probe.py` | asks for one whole node and reports the GPUs the pod sees |
| `prefetch.py` | runs `flyte.prefetch`'s task on the GPU queue, so the weights land in that pool's store |
| `docker/` | the worker (ROCm vLLM, entrypoint cleared) and driver images, and the script that pushes them to a public registry |

## Status

RESULTS_PENDING: the first GPU runs of this version are in progress. Not yet known: whether the `flyteplugins-vllm`
weight streamer works with the ROCm vLLM build, and whether eight data-parallel engines start cleanly on the node.
