# BioNeMo Playground

NVIDIA's [generative virtual screening blueprint][blueprint] running on Union, with the
BioNeMo NIMs self-hosted as Union apps and the pipeline orchestrated by Flyte 2. Built
as a booth demo for AIDD Boston.

A visitor picks a target protein and a seed molecule, hits one button, and about half a
minute later sees ten generated molecules ranked by docking pose confidence, a 3D view
of the best pose, and their name on a leaderboard. Every click is a real Flyte run with
a link to its graph.

[blueprint]: https://github.com/NVIDIA-BioNeMo-blueprints/generative-virtual-screening

## How it differs from the upstream blueprint

The blueprint is a notebook that calls four NIMs you deploy with Helm. The science is
unchanged here; what changes is the packaging and the latency budget.

**We don't fold at all, and that is deliberate.** The blueprint's pipeline is
MSA-Search → OpenFold2 → GenMol → DiffDock. The first two stages exist to produce a
receptor structure. But the demo is *generative virtual screening* — the screening is
GenMol → DiffDock. Folding is input preparation, not the thing being shown.

Every demo target here is a heavily studied protein with solved crystal structures, so
folding one would be strictly worse on all three axes:

* **Slower.** Minutes per target instead of a single HTTP GET.
* **More infrastructure.** MSA-Search needs ~1.3 TB of sequence databases on fast local
  NVMe plus two extra GPUs, which is the longest lead-time item in the whole setup.
* **Worse science.** Docking is sensitive to binding-site geometry, and predicted models
  routinely place side chains in non-physiological rotamers. Docking into a predicted
  lysozyme when a 0.65 Å crystal structure exists is a downgrade, and this audience
  knows it.

So `scripts/fetch_structures.py` pulls experimental receptors from the RCSB PDB, and the
booth runs GenMol → DiffDock only: 20–40 s with both NIMs warm, on two GPUs.

The folding path still exists in `playground/screening.py` and
`scripts/precompute_structures.py`, and the UI still has a "Fold the protein in this run"
checkbox. It is simply not deployed for the booth. It earns its keep for the case where
folding genuinely belongs in a screening workflow: a target with no experimental
structure, such as a resistance mutant or an engineered variant. That is a good answer
to give at the booth, and a bad thing to put on the critical path.

**NIMs are Union apps, not Helm releases.** Each NIM is NVIDIA's own container from
`nvcr.io`, run as a `flyte.app.AppEnvironment`. GenMol and DiffDock hold a floor of one
replica so weights stay resident in GPU memory. MSA-Search and OpenFold2 scale to zero,
so they cost nothing between folding jobs.

## Layout

```
playground/        Flyte 2 pipeline and NIM app definitions
  config.py        Every cluster-specific knob: GPU types, replicas, timeouts
  nims.py          The four NIMs as Union AppEnvironments
  clients.py       Typed HTTP clients, payloads copied from the blueprint notebook
  screening.py     The tasks: fold_target, generate_molecules, dock_candidates, screen
  targets.py       Target and seed catalogue
api/
  app.py           FastAPI: launches runs, polls them, serves the catalogue
  leaderboard.py   SQLite-backed booth leaderboard
ui/                Vue 3 + Vite booth UI
scripts/
  fetch_targets.py          Resolve target sequences from UniProt (run once)
  precompute_structures.py  Fold every target and cache the PDBs (run before the booth)
  mock_api.py               Fake backend for UI work with no cluster
deploy.py          Deploys everything in dependency order
```

## Prerequisites

1. **NGC access.** The NIM images are gated. You need an NGC API key with access to
   `nvcr.io/nim/...`, and the cluster's service account needs an image pull secret for
   `nvcr.io`. Create the runtime secret with:

   ```bash
   flyte create secret ngc-api-key --value "$NGC_API_KEY"
   ```

2. **GPUs: three.** One for GenMol, two for DiffDock at peak (it floors at one and
   scales to two). The blueprint recommends L40S / A100 / H100; the default is
   `L40s:1`, overridable per NIM with `BIONEMO_GPU_GENMOL`, `BIONEMO_GPU_DIFFDOCK`,
   etc., or in `playground/config.py`.

   GenMol is ~2 s per request, so one replica covers roughly 30 runs/minute. DiffDock
   dominates the loop; its second replica is not about throughput but about the wait
   when four people are queued, which is exactly when the demo is being watched.

   You do **not** need GPUs for MSA-Search or OpenFold2, and you do **not** need the
   1.3 TB of MSA databases, because nothing folds. That is the main practical payoff
   of using experimental structures.

## Setup

```bash
uv sync

# 0. Edit config.yaml to point at your Union tenant (or regenerate it with `flyte create config`)

# 1. Resolve target sequences from UniProt (already committed; re-run if you add one)
python scripts/fetch_targets.py

# 2. Fetch experimental receptors from the RCSB PDB (also already committed)
python scripts/fetch_structures.py

# 3. Deploy the two booth NIMs, the API and the UI
python deploy.py
```

Both data steps are committed to the repo, so a fresh clone can go straight to
`deploy.py`. Nothing in the booth path depends on the conference wifi reaching UniProt
or RCSB.

### Structure selection, and what gets stripped

`fetch_structures.py` searches RCSB for X-ray entries matching each target's UniProt
accession, takes them best-resolution first, and **verifies every download against our
own sequence** before writing it. A wrong PDB ID fails loudly rather than quietly
poisoning the demo. Targets can pin an explicit `pdb_id` where the accession is
ambiguous; SARS-CoV-2 Mpro does, because its accession is the whole replicase
polyprotein.

Verification scores two ways and keeps the better, because PDB entries break sequence
comparison in two mutually exclusive ways. Residue numbering handles *gaps* (CDK2's
activation loop is unresolved, so a flat-string compare slides out of register).
Best-offset alignment handles *renumbering* (lysozyme's crystal numbers the mature
chain while UniProt counts an 18-residue signal peptide; bovine trypsin uses historical
chymotrypsinogen numbering). Insertion codes are carried through, which trypsin needs.

The receptor written to disk is **one protein chain, no solvent, no HETATM**. Dropping
the co-crystallised ligand is the most important step: leaving it in would hand DiffDock
a receptor whose pocket is already occupied.

> **Known chemical caveat.** Stripping HETATM also removes catalytic metals. Carbonic
> anhydrase II has a structural Zn²⁺ that its real inhibitors coordinate to, so docking
> against the metal-free receptor is chemically incomplete, and a crystallographer at
> the booth may well point that out. Keeping metals is a post-conference change, once
> there is time to confirm DiffDock handles them. The other four targets are unaffected.

### If you do want to fold

The folding path is intact, just undeployed:

```bash
python deploy.py --with-folding            # adds MSA-Search + OpenFold2 (2 more GPUs)
python scripts/precompute_structures.py    # minutes per target
```

This needs the ~1.3 TB of MSA databases. Worth it for a target with no experimental
structure; not worth it for anything in the current demo set.

## Running the UI locally

No cluster needed:

```bash
python scripts/mock_api.py        # fake backend on :8000
cd ui && npm install && npm run dev
```

The mock serves the same routes with synthetic scores and a simulated run, so the whole
interface — progress stages, candidate grid, pose viewer, leaderboard — can be worked on
from a hotel room.

## Targets

Protein sequences are **not** hardcoded. Each target is identified by UniProt accession
and `scripts/fetch_targets.py` pulls the real sequence and caches it to
`data/targets.json`. A single mistyped residue in a 300-residue sequence is invisible at
a booth and would quietly poison every fold downstream, so the sequences come from the
authoritative source or not at all.

The one exception is `sars-cov-2-mpro`, copied verbatim from the blueprint notebook so
our output can be diffed against NVIDIA's reference run.

Adding a target means adding a `Target(...)` to `playground/targets.py`, re-running
`fetch_targets.py`, and folding it with `precompute_structures.py --only <key>`.

## Reading the output, and saying it correctly

The leaderboard ranks on **DiffDock pose confidence**. This is the confidence model's
estimate that the predicted *geometry* is correct. It is **not** a binding affinity, and
DiffDock does not predict one. The UI says so, and so should you — this audience reads
docking output for a living, and claiming "best binder" is the fastest way to lose them.

Confidence is an unbounded logit, in practice about −7 to +3, coloured by the usual
convention: above 0 is high, −1.5 to 0 moderate, below −1.5 low.

QED, molecular weight, logP and ring counts are computed locally with RDKit and are
there to make the ranking chemically legible rather than a single opaque number.

## Operational notes for the booth

- **Warm up before the doors open.** Run one screen manually. The first request after a
  deploy pays model load time.
- **Watch the NIM pills** in the UI header. They poll `/v1/health/ready` every ten
  seconds. Two green means the fast path is live.
- **The leaderboard is SQLite on the API container.** It survives a server restart but
  not a pod replacement, and the API is pinned to one replica for that reason. Point
  `LEADERBOARD_PATH` at a mounted volume if you want it to survive anything.
- **NIM auth is off.** `requires_auth=False` on the NIM apps so task → NIM calls need no
  token plumbing. These are stock NVIDIA inference containers with none of your data in
  them, but turn it on before using any of this beyond a demo on a dev domain.

## Benchmark against NVIDIA's hosted demo

Measured 2026-10-27 against the hosted blueprint at
[build.nvidia.com](https://build.nvidia.com/nvidia/generative-virtual-screening-for-drug-discovery),
driven through a real browser. Wall clock from clicking "Generate Molecules" to the
output rendering, on NVIDIA's shared free tier.

| Run | Target | Length | Wall clock |
|---|---|---|---|
| 1 | SARS-CoV-2 Mpro (their default example) | 306 aa | **67.2 s** |
| 2 | CDK2 kinase domain (novel sequence) | 298 aa | **129.2 s** |
| 3 | SARS-CoV-2 Mpro, repeat | 306 aa | **64.2 s** |

Two things fall out of this.

**Their hosted demo is faster than the notebook implies.** The blueprint notebook warns
that MSA search takes "a few minutes". The hosted service does the whole pipeline — MSA,
fold, generate, dock — in about a minute for their example. Any claim that NVIDIA's
version is slow because folding is slow is wrong, and should not be made at the booth.

**But their default example is clearly privileged.** CDK2 is *eight residues shorter*
than Mpro and took roughly twice as long. Mpro reproduced at 67 s and 64 s across two
runs minutes apart, so that is not noise. The most likely explanation is that the MSA or
the folded structure for their showcase sequence is cached, which is the same
optimization this playground makes explicit. Treat ~130 s as the honest figure for a
protein their cache has never seen.

Caveats, because this matters if you quote it: n=1 for CDK2, shared free tier, so some
of the spread is queueing rather than compute. The per-stage split could not be
recovered — the page streams its pipeline calls and they do not appear in the network
log — so these are end-to-end numbers only.

**What this means for the booth.** The honest comparison is not "ours is fast, theirs is
slow". It is: *for a novel target, the full pipeline costs about two minutes, and most of
that is folding. We pay that once, offline, so the interactive loop is 20–40 s.* If
someone asks what happens with a protein we have not precomputed, the honest answer is
"about the same as NVIDIA's, because it is the same models" — and then you show them the
"Fold the protein in this run" checkbox.

Incidental confirmation from these runs: the DiffDock pose confidences came back as
−0.857, −1.645, −2.305, −2.742, −4.250 (Mpro) and −0.178, −0.399, −1.146, −2.633, −3.627
(CDK2). All negative, which matches the scoring convention the UI colours against. A
leaderboard that painted those green would have been wrong.

## GPUs, replica floors, and what they cost

Both NIMs hold a floor of one replica by default, so each keeps its node and a
visitor's click costs only inference time.

**The replica floor is what holds a GPU**, not the scaledown window. At `min=0` the pod
goes away, Karpenter reclaims the empty node after 10 minutes, and the next request
pays node provisioning plus a multi-gigabyte image pull plus NIM startup — about **ten
minutes**, measured. Fine overnight, fatal in front of a visitor.

| NIM | GPU | Node | Cost of holding it |
|---|---|---|---|
| GenMol | T4 | `g4dn.metal`, shared, exists anyway | **nothing extra** |
| DiffDock | L4 | `g6.2xlarge`, dedicated | **~$0.98/hour** (~$23/day) |

To scale down between demo days:

```bash
BIONEMO_DIFFDOCK_FLOOR=0 BIONEMO_GENMOL_FLOOR=0 python deploy.py
```

### Which GPU DiffDock can actually use

This is the sharpest edge in the whole project. **DiffDock fails silently on T4**: the
request returns HTTP 200 with `status: "failed"`, a `-1000.0` sentinel confidence and
empty poses, in about 0.3 s. Triton reports the inference as *successful*. Nothing
logs an error anywhere. The same request on an L40S returns 10 real poses in 2.5 s.
NVIDIA's hardware requirements list L40S/A100/H100 and omit T4 — that omission is
load-bearing, not advisory.

DiffDock is pinned to **L4** here: the same Ada generation as L40S, roughly half the
price, and usually easier to get. Check what your cluster can actually schedule before
choosing; an L40S node pool that is configured but contended is worse than an L4 pool
that provisions in 90 seconds.

### Will it lose the GPU mid-event?

If your cluster autoscales with Karpenter, check the GPU NodePool before the event:

* `capacity-type: on-demand`, not spot, so no interruption notices. This is the big one.
* `expireAfter` longer than the event, so nodes do not age out.
* A `Drifted` disruption budget of 0, so config drift does not trigger a rolling
  replacement.
* `consolidationPolicy: WhenEmptyOrUnderutilized` is **the one real risk.** Karpenter
  can reclaim a node it judges underutilized and reschedule the pod, and a swap costs
  the ten-minute cold-start cycle.

Cheapest mitigation: hit `/v1/health/ready` every few minutes from a cron so the pod
never looks idle. Definitive mitigation: a `karpenter.sh/do-not-disrupt: "true"`
annotation on the pod.

## Deployment runbook

Everything in this section was learned the hard way against a real cluster. None of
these failures reports its actual cause, so read this before debugging one of them.
If you have `kubectl` access to the dataplane, reading pod events directly is far faster
than inferring from Flyte's deployment conditions; set it up before the event, not
during it.

### Failure 1: unschedulable pod, reported as "Insufficient cpu"

```
Pod cannot be scheduled: Insufficient cpu; Insufficient memory;
Insufficient nvidia.com/gpu; node(s) didn't match Pod's node affinity/selector
```

Reads like GPU scarcity. Is not. Single-GPU instances advertise **7910m** allocatable,
not 8000m, because Kubernetes reserves ~90m for system daemons. A `cpu=8` request
overflows every single-GPU node by 1% and is silently unschedulable. `CPU_PER_NIM` is 6
for this reason; do not raise it above 7.

Pinning a GPU type the cluster has no node for produces the same message. When in
doubt, request `gpu=1` (type-agnostic) and see where it lands.

### Failure 2: pod evicted with exit 137, which is not an OOM

```
Pod ephemeral local storage usage exceeds the total limit of containers 1Gi
```

Exit 137 is SIGKILL and looks exactly like an out-of-memory kill. It isn't. NIMs unpack
their model store into `/opt/nim/workspace`, which is ephemeral container storage; with
no explicit `disk` request the pod inherits a 1 Gi limit and is evicted mid-startup.
`DISK_GENMOL` / `DISK_DIFFDOCK` exist to prevent this.

### Failure 3: a failed app ignores redeploys

A Union App in a failed state does **not** pick up a new image on redeploy. The spec
updates, but Knative never creates a new revision, so the app sits on the old revision
and keeps reporting the original error — which looks like your fix did nothing. Recreate
it instead:

```python
a = flyte.remote.App.get(name="nim-diffdock")
a.deactivate()
flyte.remote.App.delete(name="nim-diffdock")
flyte.deploy(diffdock_env)
```

`delete` fails with "must be in a stopped state" unless you `deactivate()` first. Worth
rehearsing before the booth; it is the recovery path for any wedged NIM.

### Failure 4: app CrashLoopBackOff on a missing data file

```
FileNotFoundError: No target cache at /home/flyte/data/targets.json
```

The code bundle ships **Python files only**. Anything else — `data/targets.json`, the
cached receptor PDBs — has to be named in the environment's `include`. The app imports
fine locally and then crash-loops on the cluster. Note `include` resolves relative
entries against the directory of the file that defines the environment, so `api/app.py`
passes absolute paths for things at the project root.

### Failure 5: UI container dies with no logs at all

Two distinct versions of this, both silent:

* `StartError exit 128`, empty log — the image was built with `install_flyte=False`.
  Because the env uses `args` rather than `command`, the container is launched through
  Flyte's `fserve` entrypoint, and that flag removes the binary. Keep Flyte installed
  even though no Python of ours runs in the UI container.
* `sh: 0: cannot open start.sh: No such file` — the bundle is rooted at the **project
  root**, so `ui/` files land under `ui/`, while the container's working directory is
  the bundle root. Launch it as `sh ui/start.sh`, and have the script `cd` to its own
  directory. Examples that keep the deploy script beside the UI files do not hit this.

### Why the images are mirrored

See `scripts/mirror_nims.sh`. Short version: Knative resolves an image tag to a digest
before any pod exists, so Union's `--type image_pull` secrets (which work via the pod
admission webhook) cannot help, and nvcr.io is not in the cluster's
`registries-skipping-tag-resolving` list. Deploying straight from nvcr.io fails with a
401 at digest resolution. The cluster's own ECR is in that list and its node role can
pull from it, so mirroring avoids the problem entirely with no dataplane change.

```bash
ECR_ACCOUNT=123456789012 AWS_PROFILE_NAME=my-profile ./scripts/mirror_nims.sh
export BIONEMO_ECR_HOST=123456789012.dkr.ecr.us-east-2.amazonaws.com
python deploy.py
```

With `BIONEMO_ECR_HOST` unset, the NIMs deploy straight from nvcr.io, which only works
on a dataplane already configured with nvcr.io credentials.

## Known gaps

- `screen` runs generation and docking in one task rather than fanning out. With one
  warm replica of each NIM there is nothing to gain from parallelism, and a flat run
  graph is easier to narrate to someone watching over your shoulder. If you get more
  GPUs, splitting the docking batch across replicas is the obvious next move.
- The nickname → run mapping lives in memory in the API process, so a restart
  mid-run loses the name and the entry lands as "anonymous".
