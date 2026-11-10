# Batch inference on a warm GPU pool: Kimi K3 (or a stand-in) in long-running containers

No model server. The model lives inside a **reusable task environment**: a pool of long-running GPU containers
that each load the weights once and keep them in GPU memory between task calls. A CPU driver task splits the rows
into chunks and calls the `generate` task once per chunk; Flyte routes each call to a warm replica, and each replica
hands its chunk to vLLM as one batch. Kill a replica in the middle and the run still finishes: the unfinished call is
retried on a live replica and the pool starts a replacement.

```bash
flyte run workflow.py batch                   # 200 clinical notes, chunks of 25: both replicas load the model
flyte run workflow.py batch                   # again, straight away: the pool is still warm, no model load
flyte run workflow.py batch --crash_chunk 3   # chunk 3 kills its replica on the first try; the run finishes anyway
```

Part of the [Kimi K3 series](../KIMI_K3.md). The endpoint version of the same job is
[`kimi-k3-batch-inference`](../kimi-k3-batch-inference/): same notes, same scorer (`extract.py`), same engine
helpers (`engine.py`), so the numbers compare.

## Results

Measured on the Union demo cluster (2026-11-10): Qwen3-8B on a pool of two L40S replicas, 200 clinical notes in
chunks of 25, thinking off, greedy, weights from a `flyte.prefetch.hf_model` run.

| | Cold pool | Warm pool, straight after | Chunk 3 kills its replica |
|---|---|---|---|
| Run | `ux8lrnpm5k6dkx6gnvm2` | `usxckqx7vlw9ppzn792f` | `ust48wkhvbpnftdh72xs` |
| Rows scored / valid JSON | 200 / 200 | 200 / 200 | 200 / 200 |
| Mean field score | 70.5% | 70.4% | 70.5% |
| Wall clock | 782 s | **20.7 s** | 952 s |
| Rows per minute | 15 | **581** | 13 |
| Model loads in this run | 2 (226 s, 228 s) | **0** | 3 (154 s, 159 s, 227 s) |
| Replicas that served chunks | 2 | the same 2 | 3: two, and the one that replaced the killed replica |
| Chunks retried | none | none | chunk 3, finished on the retry |
| Prefix-cache hits (input tokens) | 21,264 | 66,000 | 21,056 |

What the table says. Generating a chunk of 25 notes takes about 5.5 s; everything else in the cold run is pods
starting and an 8B model loading, twice. The warm run is the same job with none of that: the engine stayed in GPU
memory between runs, so 200 notes took 21 seconds. The crash run lost a replica mid-batch, and the batch did not
notice beyond one chunk: the call was retried on a live replica, the pool started a replacement (the third load),
and every row came back scored. It ran cold because the code change made it a new pool. Per field, the model gets
age and sex almost always (96 and 98%) and lists of symptoms and treatments about half the time (47 and 41%).

Where does a retry go? Run `usgc8bxvlc69pd84zhcf` (demo cluster, Qwen3-8B, 2 L40S replicas, 1,000 notes in 40 chunks,
chunk 3 kills its replica): both replicas worked side by side (one 25-row chunk per ~5 s each); chunk 3 was picked up at
about 420 s, its replica died, and the retry ran on the **surviving warm replica** (no model load) and finished 17 s
after the crash. The survivor then finished the queue alone, so throughput halved; the run ended (541 s, 1,000 / 1,000,
71.2%) before a replacement had loaded. Retries go to whichever replica is free first. The run summary now carries
`chunks_per_replica`, `retries` and `chunk_finish_s_by_replica`.

On a whole-node AMD GPU pool (2026-11-10): Qwen3-0.6B, one replica on one 8× MI355X node, 50 notes in chunks of 10,
images from a public registry (`docker/build-public.sh`), weights prefetched into that pool's store with `prefetch.py`.

| | Cold pool | Warm pool, straight after |
|---|---|---|
| Run | `umg4dhfg98s9k5rmhpcc` | `ufsdrv6fsss2kp2xg5mr` |
| Result | all 5 chunks succeeded | all 5 chunks succeeded |
| Wall clock | about 16 min: node grant, 9 GB worker image pull, model load | **12 s**: each chunk 1 to 4 s on the node already up |

Then the demo shape: Qwen3-8B, **two replicas on two 8× MI355X nodes**, 1,000 notes in chunks of 50.

| | Cold pool, chunk 3 kills its replica | Warm pool, straight after |
|---|---|---|
| Run | `uxdc4s5gbppxf9wwlb98` | `unqtjprvbcqkf275qf6d` |
| Chunks | 20 / 20 succeeded; chunk 3 failed once ("Executor process terminated") and succeeded on its retry | 20 / 20 succeeded |
| Wall clock | 10 min 13 s: 19 chunks done 3.5 min in (model load included); chunk 3's retry ran 6 min | **46 s** |

The reports show who did the work. In the crash run one replica died on its first call (chunk 3), the other ran the
remaining 19 chunks, and the replacement (model load 129 s) ran only chunk 3's retry. In the warm run a single replica
served all 20 chunks: the other had been scaled down, because the per-replica `scaledown_ttl` was 5 minutes (now equal
to `IDLE_TTL`). Each 50-row chunk generates in about a second on one of the node's 8 GPUs, so the warm 46 s is mostly
per-call overhead; the demo wants bigger chunks and all 8 GPUs per node. Mean field score was 71.2% in both runs.

The retry of chunk 3 went to the replacement replica, which loaded the model again, though the other replica was free. Correct, but the wait grows with the model's load time: smaller chunks with replay of
finished sub-batches, or a spare replica, would shorten it.

A bigger model on the same pool: **gpt-oss-120b**, one replica on one 8× MI355X node, 200 notes in chunks of 50,
`reasoning_effort=low`, a 1,024-token budget (run `uk9qzr9k8pcqtvcppqnh`): all 4 chunks succeeded in 5 min 19 s, of
which 4 min 45 s was the node, the image pull, the weights (top-level files only, about 60 GB) and vLLM loading the
model; the chunks then finished 6 s apart. The worker keeps only gpt-oss's final channel, so its reasoning never
reaches the JSON parser.

And the demo shape on gpt-oss-120b: two replicas, 4,000 notes in 80 chunks, chunk 3 kills its replica (run
`u6m7tbkhj7vck8pb25rs`): 80 / 80 in 10 min 20 s. The survivor finished the other 79 chunks four minutes after the model
load, but the crash was only detected six minutes after it happened ("Reusable worker ... was unexpectedly
disconnected", against "Executor process terminated" for Qwen): vLLM's engine runs in a child process here, which
likely kept the container alive after the task's process exited, so Union noticed when the worker's heartbeat stopped.
The retry then took 6 s on the warm survivor. Killing the whole container (or deleting the worker pod) shows a faster
detection than `os._exit` in the task.

That pool's console and API do not return task outputs yet, so the scores for these two runs are not in the table.

## How it works

- **The pool** is `config.worker_env`: a `TaskEnvironment` with `ReusePolicy(replicas=REPLICAS, concurrency=1,
  idle_ttl=IDLE_TTL)` on the day-0 vLLM image plus the Flyte runtime and `unionai-reuse`. `concurrency=1` because there
  is one in-process engine per container and each call already gives vLLM a whole chunk.
- **The engine** is loaded once per container (`workflow._engine`, a module global): download the weights from
  `MODEL_URI` (a `flyte.prefetch.hf_model` output) or let vLLM pull `MODEL` from the Hub, then `vllm.LLM(...)` in a
  thread so the replica's heartbeat keeps running. Later calls on that replica reuse it.
- **The driver** (`batch`) is a plain CPU task: `prepare` (cached), one `generate` call per chunk, scoring with
  `extract.py`, a live report. The report shows a timeline of model loads and chunks by replica, which replica loaded
  the model in this run and which was already warm, and which chunks were retried.
- **Killing a replica**: `--crash_chunk N` makes chunk N write a marker to the run's object storage and call
  `os._exit(137)` (as on an OOM kill or a lost node). `generate` has `retries=3`; the retry finds the marker and runs
  normally on a live replica. You can also delete a worker pod by hand while a batch runs.

## Files

| File | What it does |
|---|---|
| `config.py` | `MODEL` picks the family; `GPU` the pod shape; `REPLICAS`, `IDLE_TTL`, `MODEL_URI`, `WORKER_IMAGE`; the worker and driver environments |
| `workflow.py` | `generate` (the warm worker), `prepare`, `batch` (the driver and its report) |
| `build.py` | Build the worker image once and print its URI, to pin with `WORKER_IMAGE` |
| `extract.py` | The task module: clinical notes, prompt, parse, per-field scores (same as the endpoint tutorial) |
| `engine.py` | Scoring and summaries (the same file as the endpoint tutorials) |
| `report_helpers.py` | Report CSS and SVG charts |

## Run

**Union demo cluster, Qwen3.5-4B on L40S** (two replicas):

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
export MODEL_URI=<the prefetched Qwen3.5-4B directory, from kimi-k3-serving's prefetch run>
flyte run workflow.py batch
```

**A whole-node AMD GPU pool (8× MI355X per node)**, a small model first:

```bash
export GPU=MI355Xx8 MODEL=Qwen/Qwen3-0.6B
python build.py                                  # once: prints the worker image URI
export WORKER_IMAGE=<that URI> MODEL_URI=<a prefetched Qwen3-0.6B directory>
QUEUE=<your GPU queue> python workflow.py --n 50 --chunk_size 10 --config <your cluster config>
```

**Kimi K3**: `MODEL=moonshotai/Kimi-K3 GPU=MI355Xx8`, a `MODEL_URI` from a K3 prefetch, and the same commands. The pool
keeps one replica (one whole node) warm for three hours by default, because loading 1.5 TB takes about an hour.

## Things to know

- **The vLLM image's entrypoint has to go.** The stock vLLM images set `ENTRYPOINT ["vllm", "serve"]`. Union starts a
  reusable worker by passing its launcher as the container args, so on those images vLLM reads the launcher as a
  model name and every replica exits with code 2 (`vllm: error: unrecognized arguments: --last-ack-grace-period-seconds
  ...`). Setting `command` in the pod template does not help: Flyte drops it. The NVIDIA worker therefore uses the vLLM
  plugin's image, built from Flyte's base with no entrypoint; a ROCm worker needs a Flyte-built ROCm vLLM image too.
- **The ROCm image's Python is externally managed.** `vllm/vllm-openai-rocm` installs vLLM into the system Python 3.12,
  which is marked PEP 668 externally managed, so the image builder's `uv pip install` refuses it. The worker image sets
  `UV_BREAK_SYSTEM_PACKAGES=1` next to `UV_PYTHON=/usr/bin/python3`, so the Flyte runtime lands next to the image's vLLM.
- **A whole-node GPU pool needs the GPUs asked for in a pod template.** Our pool admits only pods that request a multiple
  of 8 `amd.com/gpu`. The request goes on the container in the pod template, with no node selector or toleration (the
  queue adds them). `flyte.Resources(gpu=flyte.Device(device_class="AMD_GPU", ...))` makes Union add an AWS accelerator
  affinity these nodes never match, and the task waits for resources forever.
- **On a pool without an image registry, build the worker image elsewhere, then pin it.** An image is built by the cluster its environment
  runs on. The GPU pool's builder tags images `union-dataplane/<name>` with no registry host and cannot
  push; pinning `registry=` gets an anonymous-token 403; and the "already built?" check runs per cluster, so it cannot see
  an image the other builder made. `python build.py` builds on the default pool and prints the URI;
  `WORKER_IMAGE=<uri>` makes the environment use it as-is, with nothing to build.
- **Prefetch weights on the GPU's own queue.** A cluster pool is its own data plane: a node in the GPU pool cannot
  read the default pool's GCS bucket (it has no GCP credentials). `flyte.prefetch.hf_model` takes no queue, so
  `prefetch.py` runs Flyte's own prefetch task on `QUEUE` instead, in the public Flyte image with its three packages
  installed at task start. Give the copy its own `--artifact_name`: artifact versions are the Hub commit, so a second
  pool's copy of the same commit would not be recorded. Verified with Qwen3-0.6B: stored in the GPU pool's store
  in 23 s and read back from that queue at about 200 MB/s.
- **A reusable task cannot see its attempt number.** `FLYTE_ATTEMPT_NUMBER` is set when the container starts, not per
  task, so a crash triggered on "attempt 0" fires on every retry and the chunk fails for good. The crash switch uses an
  object-store marker instead.
- **The worker image must come from Flyte's base.** The plugin's `DEFAULT_VLLM_IMAGE` is the SDK's pattern for reusable
  vLLM workers; the stock vLLM images' `vllm serve` entrypoint kills every replica, and a pod-template `command` is dropped.
- **The driver pod builds the worker's spec.** Child task specs are serialized in the parent's pod, so the driver image
  carries the `kubernetes` client: without it the worker's pod template (the GPU request) would silently disappear.
- **Each grant is a whole node with a minimum billing period** on our GPU pool. Every GPU workload so far got
  its own grant, so a test costs at least two node-hours; keep the pool warm (`IDLE_TTL`) instead of relaunching it.
