# Fine-tune Qwen3.5-4B three ways, measure them one way, serve the winner

Full fine-tune, LoRA and QLoRA on the same support-ticket task, scored on the same held-out rows
by the same function, with the cost of each (VRAM, minutes, bytes) on the same row as the lift.
Then the adapter is served next to the base on the same vLLM endpoint the serving tutorial built.

```bash
flyte run workflow.py baseline                  # score the untouched model first
flyte run workflow.py pipeline --method lora    # BF16 base + LoRA adapter
flyte run workflow.py pipeline --method qlora   # 4-bit NF4 base + LoRA adapter
flyte run workflow.py pipeline --method full    # every weight, bf16, 8-bit AdamW
flyte run workflow.py compare                   # all three in parallel, one report
python serve.py                                 # the adapter, served as its own model name
```

This is leg 2 of the [model-drop template](../MODEL_DROP.md). The dataset comes from the
rotation in [`IDEAS.md`](../IDEAS.md); swapping it means replacing `support.py`.

<a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/qwen3.5-fine-tuning/qwen3.5-fine-tuning-tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Results

RESULTS_PENDING

## The task, and why it has a gap to close

[Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(CDLA-Sharing-1.0, 27k rows): a customer message, one of 27 intent labels, and a reply in a
consistent house voice that keeps `{{placeholders}}` like `{{Order Number}}` intact. The model
must answer `{"intent": ..., "reply": ...}`.

The intent set is the gap. The base model can write a decent support reply; it does not know that
`check_cancellation_fee` and `cancel_order` are different labels, or where the line between
`edit_account` and `switch_account` is, because nobody told it. That is exactly the kind of thing
fine-tuning teaches cheaply. Run `baseline` first and look at the per-category bars before
training anything: if the untouched model already scores high, pick a different dataset.

## How the measurement works

Every evaluation produces the same numbers, in order of how much to trust them:

1. **Intent exact match**, against the gold label. Recomputable, no judgement.
2. **Format kept**: valid JSON with both keys, a real intent, and every placeholder from the
   message present in the reply. Also recomputable.
3. **Pairwise judge** (optional, `JUDGE=claude-haiku-5-5` in `.env` with the `ANTHROPIC_API_KEY`
   secret): base reply vs fine-tuned reply for the same ticket, blind, judged twice with A and B
   swapped. A verdict that flips with the swap counts as a tie, so position bias shows up as
   disagreement rather than as a win. This is the only number here that is an opinion, and it is
   the last one on the row.

Base and fine-tuned are scored in the same task, on the same rows, with greedy decoding and
thinking off, and the base is loaded the way it was trained (4-bit for QLoRA), so the comparison
isolates what the training changed. `compare` runs the three methods in parallel on three GPUs
and puts them in one table.

**Reports.** `train` shows a live loss curve, the discovered LoRA targets, trainable fraction and
VRAM; `evaluate` shows the two bar charts (overall, and intent accuracy by category) and
side-by-side replies; `pipeline` and `compare` roll them up.

## The three methods

| Method | What trains | Optimizer state | What comes out | Fits |
|---|---|---|---|---|
| `full` | every weight (vision tower frozen) | 8-bit AdamW, 2 bytes/param | the whole model, ~9 GB | one 48 GB card, barely; this is what `adamw_bnb_8bit` buys |
| `lora` | rank-32 adapters on every linear layer | AdamW on the adapters only | an adapter, tens of MB | one 48 GB card with room |
| `qlora` | the same adapters over a 4-bit NF4 base | the same | an adapter, tens of MB | a 24 GB card |

**LoRA targets are discovered, not copied.** Qwen3.5 is a hybrid: three Gated DeltaNet blocks for
every attention block. The `q_proj, k_proj, v_proj, o_proj` list from other tutorials would skip
three quarters of the layers, and PEFT would not complain. `support.discover_lora_targets` walks
the module tree and adapts every `nn.Linear` outside the vision tower and the embeddings.

## Serving the result

`serve.py` deploys vLLM with `--enable-lora` and the pipeline's adapter mounted from the run's
output, so the fine-tuned model is `qwen3.5-4b-support` on the same OpenAI API that serves the
base. A `full` run is served as the base model from its run output instead. The agent tutorial
can use either by changing `BRAIN`.

## Files

| File | What it does |
|---|---|
| `config.py` | `MODEL`, `DATASET`, `GPU`, optional `JUDGE`; pod shapes; pip pins; the two task environments |
| `support.py` | The task: prompt, target, JSON parsing, the two scores, LoRA target discovery. Swap this for a new dataset |
| `workflow.py` | `prepare_data` → `train` → `evaluate`; `pipeline`, `baseline`, `compare` entrypoints |
| `serve.py` | vLLM with the adapter (or the full weights) from a pipeline run |
| `report_helpers.py` | Report CSS and SVG charts |
| `qwen3.5-fine-tuning-tutorial.ipynb` | The same steps as a notebook |

## Setup

**Attach an accelerator first.** The GPU tasks need one, and a 4B model on a CPU will not finish:

| Where | What to do | Which methods |
|---|---|---|
| Union cluster | nothing; `GPU=L40s` is the default (`A100` also defined) | `full`, `lora`, `qlora` |
| NVIDIA machine | `flyte start devbox --gpu` | depends on the card: 48 GB for `full`/`lora`, 24 GB for `qlora` |
| Apple Silicon Mac | `flyte start devbox --apple-silicon`, then `GPU=apple` in `.env` | `lora` (bitsandbytes and the 8-bit optimizer are CUDA-only) |
| Colab | Runtime → Change runtime type → a GPU, then the `--local` cells | `qlora` on a T4 |

Three ways to run it; pick one. The `flyte run` commands below are the same on a Union cluster
and on a devbox; `flyte run --local` runs a task in your own Python with no cluster at all.

| | Union cluster | Devbox (your machine) | No cluster (`--local`) |
|---|---|---|---|
| What runs the work | the cluster's GPUs | a Docker container on your machine, your GPU | your Python process, your GPU if torch finds one |
| Needs | Python 3.10+, `uv`, a browser sign-in | Python 3.10+, `uv`, Docker running | Python 3.10+, `uv` |
| GPU legs (serve, fine-tune, RL) | yes | yes, with an NVIDIA GPU (`--gpu`) | fine-tune and RL only, if you have a CUDA GPU |
| Reports, run graph, retries, cache | yes | yes | no: plain Python, printed output |

In every case, start here:

```bash
git clone https://github.com/unionai/workshops
cd workshops/tutorials/qwen3.5-fine-tuning
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
flyte --version          # 2.12 or newer
```

<details>
<summary><b>Union cluster</b></summary>

```bash
flyte create config --endpoint dns:///<your-union-host> --project flytesnacks --domain development --builder remote
flyte get project         # opens a browser sign-in the first time
```

Optional judge: `JUDGE=claude-haiku-5-5` in `.env`, with an `ANTHROPIC_API_KEY` secret on the cluster
(`flyte create secret ANTHROPIC_API_KEY --value sk-ant-...`), or point `JUDGE_SECRET` at a secret you already have.
</details>

<details>
<summary><b>Devbox on your machine</b></summary>

A devbox is a Flyte control plane in one container. Workloads that ask for a GPU use the
machine's own:

```bash
flyte start devbox --gpu               # an NVIDIA machine
flyte start devbox --apple-silicon     # an Apple Silicon Mac: GPU tasks run natively on the Mac, PyTorch on mps
flyte create config --devbox           # writes .flyte/config.yaml pointing at localhost:30080
echo "GPU=apple" >> .env               # Mac only; the task image is then built without the CUDA-only packages
flyte run workflow.py baseline --num_eval_examples 20
flyte run workflow.py pipeline --method lora --train_samples 200 --num_eval_examples 40
```

The first run builds the task image on your machine (several minutes: torch, transformers, trl).
On the Mac the Gated DeltaNet layers run on transformers' pure-PyTorch fallback (correct, slower than
the fused CUDA kernels), so keep `--train_samples` in the low hundreds. `qlora` and `full` need CUDA.
The judge works the same as on a cluster: `flyte create secret ANTHROPIC_API_KEY --value ...` against the devbox.
</details>

<details>
<summary><b>No cluster: <code>--local</code></b></summary>

The same pipeline in your Python, on whatever GPU torch finds. Install the training stack locally first:

```bash
uv pip install -r requirements-local.txt
flyte run --local workflow.py baseline --num_eval_examples 20
flyte run --local workflow.py pipeline --method lora --train_samples 100 --num_eval_examples 20
```

Verified paths are the cluster and the devbox; `--local` is for a CUDA laptop or Colab's GPU, where
`qlora` is the one that fits a 16 GB card. There are no reports without a Flyte deployment; the
numbers print.
</details>

## Run

```bash
flyte run workflow.py baseline                                   # ~10 min: image build + 100 held-out tickets
flyte run workflow.py pipeline --method lora                     # data -> train -> evaluate
flyte run workflow.py pipeline --method qlora --gpu L40s
flyte run workflow.py pipeline --method full
flyte run workflow.py compare                                    # three GPUs at once
flyte run workflow.py pipeline --train_samples 100 --num_eval_examples 20   # a smaller first run
```

Then serve what a run produced:

```bash
python serve.py                        # latest pipeline run, LoRA adapter
python serve.py --run <run> --method full
```

### Parameters

| Flag | Default | Notes |
|---|---|---|
| `--method` | `lora` | `full`, `lora`, `qlora` |
| `--gpu` | from `config.py` | `L40s` or `A100`; overrides the pod shape for the GPU tasks at call time |
| `--train_samples` / `--eval_samples` | 800 / 200 | shuffled with seed 42; eval rows are held out from every run |
| `--num_eval_examples` | 100 | how many held-out rows to score |
| `--judge_pairs` | 40 | reply pairs to send to the judge, if `JUDGE` is set |
| `--lr` | 0 → method default | 1e-5 for full, 1e-4 for the adapters |
| `--max_length` | 768 | tickets are short; this is generous |
| `--lora_r` / `--lora_alpha` | 32 / 64 | |

## Union or open-source Flyte

Everything here runs on open-source Flyte: the adapter moves between tasks as a run output and
`serve.py` mounts it from the run. On Union, `pipeline` also publishes it as a new version of the
`qwen3.5-4b-support` model artifact with the evaluation numbers on its card and as searchable
attributes, so the Versions tab is the history of your fine-tunes and the lineage view leads back
to the run and the data. `serve.py --artifact` then binds the app to the artifact by name. The
registry is detected at run time; `ARTIFACTS=0` forces the fallback.

## Things to know

- **Full fine-tuning on one 48 GB card works because of the optimizer, not the model.** bf16
  weights (9.3 GB) + bf16 grads (9.3 GB) + 8-bit AdamW (9.3 GB) leaves room for activations with
  gradient checkpointing. Plain AdamW would need 37 GB of state on its own.
- **`flash-linear-attention` is in the image for a reason.** Without its fused kernels
  transformers runs Qwen3.5's Gated DeltaNet on a pure-torch fallback that is correct and several
  times slower.
- **The base scores are not zero.** The base model writes fine JSON and gets the obvious intents.
  The lift is on the labels that need the dataset's definitions; the per-category chart shows
  where.
- **The judge is optional and last.** Both recomputable numbers are reported without it. When it
  runs, its win rate is next to its W/L/T so a pile of ties is visible.
