# Demand Forecasting to Vehicle Routing

Forecast tomorrow's demand across a city with a **time-series foundation model**, then solve the fleet routing problem it implies — on real data, entirely on CPU. Orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why This Matters

Most logistics decisions are two problems chained together, and teams usually treat them as separate systems: *what is demand going to do?* and *given that, where do the vehicles go?* The first is a forecasting problem, the second is a combinatorial optimization problem. They need completely different kinds of model, and the handoff between them is where plans go wrong.

This pipeline runs both in one workflow:

- **Chronos-Bolt**, a 205M-parameter time-series foundation model, applied **zero-shot**. No training, no per-series fitting. History goes in, a predictive distribution comes out.
- **OR-Tools**, a classical constraint solver, turning that forecast into a capacitated vehicle routing plan over real coordinates.

Two completely different modelling paradigms — a pretrained transformer and a branch-and-bound solver — in one typed Python pipeline, with no GPU anywhere.

## What This Pipeline Does

| Stage | What it does | Report visuals |
|-------|-------------|----------------|
| `load_demand` | Load real half-hourly pickup counts for ~2,400 Manhattan locations, keep the busiest N | Geographic demand bubble map, day-of-week × hour heatmap, sample series |
| `forecast_batch` | Zero-shot forecast 24 h ahead per zone (fans out over batches) | (feeds the summary) |
| `summarize_forecasts` | Merge batches, score against a seasonal-naive baseline | Forecast fan charts with 10–90% intervals, MAE comparison, forecast demand map |
| `build_routes` | Solve the capacitated VRP, compare to nearest-neighbour | **Animated route map**, side-by-side naive vs optimized, per-vehicle plan table |

## Results

On 40 zones with default settings (measured, not estimated):

| Metric | Chronos-Bolt (zero-shot) | Seasonal naive |
|--------|-------------------------|----------------|
| MAE (pickups / 30 min) | **7.47** | 13.80 |
| Zones where it wins | **40 / 40** | — |

**45.9% lower forecast error with zero training.** And on the routing side:

| Plan | Total fleet distance |
|------|---------------------|
| Nearest-neighbour | 35.44 km |
| OR-Tools optimized | **24.37 km** |

**31.2% less driving** for the same demand served, all 40 stops visited exactly once within capacity.

Forecasting all 40 zones takes **0.59 s on CPU**.

## The Data

**[autogluon/chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets)**, config `taxi_30min` — **Apache-2.0, ungated, plain parquet**.

- 2,428 Manhattan locations with genuine `lat`/`lng` (40.702–40.808 N, 74.017–73.950 W)
- Half-hourly pickup counts, 1,488 steps per series (31 days)
- Two subsets: `january_2015` and `january_2016`

Because it ships as parquet with no loading script, it is unaffected by `datasets>=4.0` removing script support — we read it with pyarrow and skip the `datasets` dependency entirely.

> **On dataset choice:** the obvious pick for a logistics demo is M5/Walmart, but that data is governed by [Kaggle competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules) rather than an open licence. `taxi_30min` is Apache-2.0 and — critically — carries real coordinates, which is what makes the routing half of this pipeline honest rather than a synthetic layout.

## The Models

**[amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base)** — 205M parameters, **Apache-2.0**. A T5-based encoder trained on a large corpus of time series. Runs comfortably on CPU and forecasts a batch of series in a single pass.

**[OR-Tools](https://developers.google.com/optimization) 9.15** — **Apache-2.0**. Google's constraint programming and routing solver, still the standard for VRP.

## Setup

```bash
cd tutorials/logistics-demand-routing

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Local

```bash
flyte run --local --tui workflow.py pipeline --n_zones 40 --vehicles 5
```

### Remote

```bash
flyte run workflow.py pipeline

# A bigger problem
flyte run workflow.py pipeline --n_zones 120 --vehicles 10 --solver_seconds 30

# Forecast a different month
flyte run workflow.py pipeline --subset january_2016
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_zones` | `60` | Busiest N zones to forecast and route |
| `subset` | `january_2015` | `january_2015` or `january_2016` |
| `vehicles` | `6` | Fleet size |
| `horizon` | `48` | Forecast steps (48 × 30 min = 24 h) |
| `batch_size` | `12` | Zones per forecast task — controls fan-out width |
| `solver_seconds` | `15` | OR-Tools time limit |
| `capacity_slack` | `1.25` | Vehicle capacity as a multiple of mean load |

## Key Concepts

**Zero-shot forecasting** — Chronos-Bolt was never fitted to this data. It was pretrained on a broad corpus of time series and generalizes to new ones at inference. This is the time-series analogue of prompting an LLM instead of training a classifier.

**Seasonal-naive baseline** — The forecast is scored against "the same 24 hours, one day earlier." For data this periodic that is a genuinely strong baseline; beating a flat mean would prove nothing. Reporting a weak baseline is the most common way forecasting results get oversold.

**Prediction intervals, not point forecasts** — The reports show the 10–90% band because capacity is a decision about the upper tail. A fleet sized to the median is short roughly half the time.

**Peak vs mean demand** — Routing uses each zone's forecast *peak* half-hour, not its average. Sizing to the average guarantees being under water exactly when it matters.

**Capacitated VRP** — Assign stops to vehicles and order them, minimizing total distance subject to per-vehicle capacity. NP-hard; OR-Tools uses a cheapest-arc construction followed by guided local search.

**Nearest-neighbour as the honest baseline** — The comparison is against what a dispatcher does without a solver (always drive to the closest unserved stop), not a random ordering. It is already decent, so the ~31% gap is a realistic estimate of what optimization buys rather than a strawman.

## How It Works

```
taxi_30min ──> busiest N zones ──> Chronos-Bolt ──> forecast + intervals
 (2,428 zones,      (real             (zero-shot,          │
  Apache-2.0)       lat/lng)          batched fan-out)     │
                                                            ▼
                        OR-Tools CVRP <──── peak demand per zone
                              │
                              ▼
                    routes vs nearest-neighbour
                    "how much less driving?"
```

## Architecture

- **config.py** — Flyte image (`.with_pip_packages`, per repo convention) + three CPU environments: loading/reporting, a **reusable** forecast pool, and the solver
- **report_helpers.py** — Self-contained SVG: Web-Mercator map projection, animated route polylines, forecast fan charts. No tile server, no CDN
- **workflow.py** — Tasks, distance matrix, VRP solve, and the pipeline orchestrator

The forecast environment uses a `ReusePolicy` so the ~800 MB Chronos weights deserialize once per warm container rather than once per batch. Route polylines animate via pure SVG `<animate>` — no JavaScript — so they draw themselves in when the report opens.

## Notes and Caveats

- **Distances are great-circle**, not road-network. A production system would call OSRM or Valhalla here. Both the naive and optimized plans pay the same metric, so the comparison holds; only the absolute kilometres would change.
- **`chronos-forecasting` v2 renamed** the `predict_quantiles` keyword from `context` to `inputs`. Versions are pinned exactly for this reason.
- The 2015/2016 subsets are separate location sets, not a train/test split of the same zones — treat `january_2016` as a second scenario rather than a holdout.

## References

- Ansari et al., "Chronos: Learning the Language of Time Series" ([arXiv:2403.07815](https://arxiv.org/abs/2403.07815))
- Chronos-Bolt: [github.com/amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- OR-Tools routing: [developers.google.com/optimization/routing](https://developers.google.com/optimization/routing)
