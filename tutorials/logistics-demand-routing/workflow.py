"""
Demand forecasting to vehicle routing, on real data, with no GPU.

The logistics question this answers end to end: *given what demand is about to do, where
should the fleet go tonight?*

  1. Load real half-hourly pickup demand for ~2,400 Manhattan locations (Jan 2015/2016).
  2. Forecast the next 24 hours per zone with Chronos-Bolt — a time-series **foundation
     model**, applied zero-shot. No training, no fitting per series.
  3. Turn forecast demand into a capacitated vehicle routing problem and solve it with
     OR-Tools, against real great-circle distances between real coordinates.
  4. Compare the optimized plan to the naive nearest-neighbour plan a dispatcher would
     otherwise fall back on, and report the difference in kilometres.

Every stage runs on CPU. The whole pipeline is minutes, not hours.

Usage:
    # Local (quick)
    flyte run --local --tui workflow.py pipeline --n_zones 40 --vehicles 5

    # Remote
    flyte run workflow.py pipeline

    # Bigger problem
    flyte run workflow.py pipeline --n_zones 120 --vehicles 10 --solver_seconds 30
"""

import asyncio
import json
import logging
import math
import os
import tempfile

import flyte
import flyte.io
import flyte.report

import report_helpers as rh
from config import cpu_env, forecast_env, solver_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DATASET_REPO = "autogluon/chronos_datasets"
DATASET_FILE = "taxi_30min/train-00000-of-00001.parquet"
MODEL_REPO = "amazon/chronos-bolt-base"

# Half-hourly data: 48 steps == 24 hours.
STEPS_PER_DAY = 48
HORIZON = 48

# Times Square — a plausible central depot for a Manhattan fleet.
DEPOT = {"lat": 40.7580, "lng": -73.9855, "name": "Midtown depot"}

PIPELINE_STEPS = ["Load Demand", "Forecast", "Build Routes", "Compare"]


# ------------------------------------------------------------------
# Geo helpers
# ------------------------------------------------------------------

def haversine_m(lat1, lng1, lat2, lng2) -> float:
    """Great-circle distance in metres."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def distance_matrix(points) -> list[list[int]]:
    """Integer metre distances. OR-Tools wants ints for its arc costs.

    Straight-line distance understates real driving distance on a street grid; a production
    system would call a routing engine (OSRM, Valhalla) here. The comparison between naive
    and optimized plans is unaffected, since both pay the same metric.
    """
    n = len(points)
    return [
        [int(haversine_m(points[i][0], points[i][1], points[j][0], points[j][1]))
         for j in range(n)]
        for i in range(n)
    ]


# ------------------------------------------------------------------
# Task 1: Load demand
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def load_demand(n_zones: int = 60, subset: str = "january_2015") -> flyte.io.Dir:
    """
    Load real half-hourly pickup counts per location, keep the busiest N zones.

    The dataset ships as plain parquet with no loading script, so it survives the
    `datasets>=4.0` removal of script support — we read it with pyarrow directly and skip
    the `datasets` dependency entirely.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    await flyte.report.replace.aio(rh.wrap_report(
        "<h2>Loading demand</h2><p>Fetching half-hourly pickup counts…</p>"
    ), do_flush=True)

    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    table = pq.read_table(path).to_pydict()

    rows = []
    for i, sub in enumerate(table["subset"]):
        if sub != subset:
            continue
        target = np.asarray(table["target"][i], dtype="float32")
        target = np.nan_to_num(target, nan=0.0)
        rows.append({
            "id": table["id"][i],
            "lat": float(table["lat"][i]),
            "lng": float(table["lng"][i]),
            "total": float(target.sum()),
            "series": target.tolist(),
            "timestamps": [str(t) for t in table["timestamp"][i][:1]],
        })

    if not rows:
        raise ValueError(f"No series found for subset '{subset}'.")

    rows.sort(key=lambda r: r["total"], reverse=True)
    zones = rows[:n_zones]
    log.info(f"Kept {len(zones)} of {len(rows)} zones for subset {subset}")

    out_dir = tempfile.mkdtemp(prefix="demand_")
    with open(os.path.join(out_dir, "zones.json"), "w") as f:
        json.dump({"subset": subset, "zones": zones, "depot": DEPOT}, f)

    # ---- report ----
    series = np.array([z["series"] for z in zones])
    total_trips = float(series.sum())
    steps = series.shape[1]

    # day-of-week x hour profile. 2015-01-01 was a Thursday.
    dow_hour = np.zeros((7, 24))
    counts = np.zeros((7, 24))
    for t in range(steps):
        day = (t // STEPS_PER_DAY + 3) % 7  # 3 == Thursday
        hour = (t % STEPS_PER_DAY) // 2
        dow_hour[day, hour] += series[:, t].sum()
        counts[day, hour] += 1
    dow_hour = np.divide(dow_hour, np.maximum(counts, 1))

    map_zones = [{"lat": z["lat"], "lng": z["lng"], "demand": z["total"]} for z in zones]
    busiest = zones[0]

    html = f"""
    <h2>Demand — Manhattan pickups, {subset.replace('_', ' ').title()}</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(zones)}</div><div class="label">Zones tracked</div></div>
      <div class="stat"><div class="value">{steps}</div><div class="label">Half-hour steps</div></div>
      <div class="stat"><div class="value">{steps//STEPS_PER_DAY}</div><div class="label">Days of history</div></div>
      <div class="stat"><div class="value">{total_trips:,.0f}</div><div class="label">Total pickups</div></div>
      <div class="stat"><div class="value">{total_trips/steps:,.0f}</div><div class="label">Mean per 30 min</div></div>
      <div class="stat"><div class="value">Apache-2.0</div><div class="label">License</div></div>
    </div>

    <div class="note">
      Real pickup counts at {len(rows):,} Manhattan locations, aggregated into 30-minute
      buckets. We keep the busiest {len(zones)} — these are the zones a dispatcher
      actually cares about. Coordinates are genuine, so every map below is real geography
      rather than a synthetic layout.
    </div>

    <div class="grid2">
      <div class="chart-container">
        {rh.demand_map(map_zones, title="Total pickups by zone", scale_label="Total pickups")}
      </div>
      <div>
        <div class="chart-container">
          {rh.heatmap_hour_dow(dow_hour.tolist(), title="Mean demand by day &amp; hour")}
        </div>
        <div class="chart-container">
          {rh.forecast_chart(zones[0]['series'][:STEPS_PER_DAY*3], [], [], [], [],
                             title=f"Busiest zone {busiest['id']} — first 3 days")}
        </div>
      </div>
    </div>

    <div class="note">
      The day/hour heatmap is the pattern any forecaster has to capture: a weekday evening
      peak, a late-night Friday and Saturday bulge, and a quiet early-morning trough. It is
      strongly seasonal at two frequencies at once (daily and weekly), which is exactly
      where a naive "same time last week" baseline starts to look reasonable — and is the
      baseline we hold the foundation model to.
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)
    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Forecast (fans out over zone batches)
# ------------------------------------------------------------------

# Cache the pipeline at module scope: under ReusePolicy the process is reused across
# batches, so the ~800 MB Chronos weights deserialize once per replica, not once per call.
_PIPE_CACHE: dict = {}


def _get_pipeline(model_repo: str):
    import torch
    from chronos import BaseChronosPipeline

    if model_repo not in _PIPE_CACHE:
        _PIPE_CACHE.clear()
        _PIPE_CACHE[model_repo] = BaseChronosPipeline.from_pretrained(
            model_repo, device_map="cpu", torch_dtype=torch.float32
        )
    return _PIPE_CACHE[model_repo]


@forecast_env.task(retries=2)
async def forecast_batch(
    zones_dir: flyte.io.Dir,
    zone_indices: list[int],
    horizon: int = HORIZON,
    model_repo: str = MODEL_REPO,
) -> str:
    """
    Zero-shot forecast a batch of zones.

    Chronos-Bolt is applied with no fitting whatsoever: the context window goes in, a
    distribution over the next `horizon` steps comes out. The last `horizon` steps are held
    out so every forecast can be scored against what actually happened.
    """
    import numpy as np
    import torch

    local = await zones_dir.download()
    with open(os.path.join(local, "zones.json")) as f:
        data = json.load(f)
    zones = data["zones"]

    pipe = _get_pipeline(model_repo)

    contexts, actuals = [], []
    for zi in zone_indices:
        y = np.asarray(zones[zi]["series"], dtype="float32")
        contexts.append(torch.tensor(y[:-horizon]))
        actuals.append(y[-horizon:])

    # NOTE: chronos-forecasting v2 renamed this kwarg from `context` to `inputs`.
    quantiles, mean = pipe.predict_quantiles(
        inputs=contexts, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
    )
    q = quantiles.numpy()
    mean = mean.numpy()

    out = []
    for k, zi in enumerate(zone_indices):
        y = np.asarray(zones[zi]["series"], dtype="float32")
        actual = actuals[k]
        median = q[k, :, 1]
        # Seasonal naive: the same 24 hours, one day earlier. The honest baseline for data
        # this periodic — beating a flat mean would prove nothing.
        naive = y[-horizon - STEPS_PER_DAY:-STEPS_PER_DAY]
        out.append({
            "zone": zi,
            "median": median.tolist(),
            "lo": q[k, :, 0].tolist(),
            "hi": q[k, :, 2].tolist(),
            "mean": mean[k].tolist(),
            "actual": actual.tolist(),
            "mae": float(np.mean(np.abs(median - actual))),
            "naive_mae": float(np.mean(np.abs(naive - actual))),
            "peak": float(np.max(median)),
            "sum": float(np.sum(median)),
        })

    return json.dumps(out)


@cpu_env.task(report=True)
async def summarize_forecasts(zones_dir: flyte.io.Dir, batches: list[str]) -> str:
    """Merge forecast batches and report accuracy against the seasonal-naive baseline."""
    import numpy as np

    local = await zones_dir.download()
    with open(os.path.join(local, "zones.json")) as f:
        data = json.load(f)
    zones = data["zones"]

    results = [r for b in batches for r in json.loads(b)]
    results.sort(key=lambda r: r["zone"])

    mae = float(np.mean([r["mae"] for r in results]))
    naive_mae = float(np.mean([r["naive_mae"] for r in results]))
    improvement = 100.0 * (1 - mae / naive_mae) if naive_mae else 0.0
    wins = sum(1 for r in results if r["mae"] < r["naive_mae"])

    # show the busiest few zones
    show = sorted(results, key=lambda r: r["sum"], reverse=True)[:3]
    charts = ""
    for r in show:
        z = zones[r["zone"]]
        history = z["series"][:-len(r["median"])]
        chart = rh.forecast_chart(
            history, r["actual"], r["median"], r["lo"], r["hi"],
            title=f"Zone {z['id']} — 24 h ahead",
        )
        charts += f'<div class="chart-container">{chart}</div>'

    map_zones = [
        {"lat": zones[r["zone"]]["lat"], "lng": zones[r["zone"]]["lng"], "demand": r["sum"]}
        for r in results
    ]

    html = f"""
    <h2>Forecast — 24 hours ahead, zero-shot</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(results)}</div><div class="label">Zones forecast</div></div>
      <div class="stat"><div class="value">{mae:.2f}</div><div class="label">MAE (Chronos)</div></div>
      <div class="stat"><div class="value">{naive_mae:.2f}</div><div class="label">MAE (seasonal naive)</div>
        </div>
      <div class="stat"><div class="value">{improvement:+.1f}%</div>
        <div class="label">Error reduction</div>
        <div class="delta {'up' if improvement > 0 else 'down'}">
          {'better' if improvement > 0 else 'worse'} than baseline</div></div>
      <div class="stat"><div class="value">{wins}/{len(results)}</div><div class="label">Zones beating baseline</div></div>
      <div class="stat"><div class="value">0</div><div class="label">Parameters trained</div></div>
    </div>

    <div class="note">
      <b>Nothing was trained.</b> Chronos-Bolt is a 205M-parameter time-series foundation
      model applied zero-shot — the history goes in, a predictive distribution comes out.
      The baseline is <b>seasonal naive</b> (the same 24 hours, one day earlier), which is a
      genuinely strong comparison for data this periodic; beating a flat mean would prove
      nothing.
    </div>

    <div class="chart-container">
      {rh.make_bar_chart(["Chronos-Bolt (zero-shot)", "Seasonal naive"], [mae, naive_mae],
                         colors=["#0284c7", "#94a3b8"],
                         title="Mean absolute error — lower is better",
                         y_label="MAE (pickups per 30 min)", lower_is_better=True)}
    </div>

    {charts}

    <div class="chart-container">
      {rh.demand_map(map_zones, title="Forecast demand, next 24 h", scale_label="Forecast pickups")}
    </div>

    <div class="note">
      The shaded band is the 10&ndash;90% prediction interval. It matters for logistics:
      capacity is a decision about the upper tail, not the mean. Sizing a fleet to the
      median guarantees you are short roughly half the time.
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)

    return json.dumps({
        "results": results,
        "mae": mae,
        "naive_mae": naive_mae,
        "improvement_pct": improvement,
        "wins": wins,
    })


# ------------------------------------------------------------------
# Task 3: Routing
# ------------------------------------------------------------------

def _solve_vrp(dist, demands, n_vehicles, capacity, seconds):
    """Capacitated VRP via OR-Tools. Returns (routes, total_metres) or (None, None)."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    manager = pywrapcp.RoutingIndexManager(len(dist), n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    transit = routing.RegisterTransitCallback(
        lambda i, j: dist[manager.IndexToNode(i)][manager.IndexToNode(j)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    demand_cb = routing.RegisterUnaryTransitCallback(
        lambda i: int(demands[manager.IndexToNode(i)])
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, [int(capacity)] * n_vehicles, True, "Capacity"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(int(seconds))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None, None

    routes, total = [], 0
    for v in range(n_vehicles):
        idx = routing.Start(v)
        path, load, dist_m = [], 0, 0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                path.append(node - 1)  # back to zone indexing (0 is the depot)
                load += demands[node]
            nxt = solution.Value(routing.NextVar(idx))
            dist_m += routing.GetArcCostForVehicle(idx, nxt, v)
            idx = nxt
        if path:
            routes.append({"vehicle": v, "path": path, "distance_m": dist_m, "load": int(load)})
            total += dist_m
    return routes, total


def _nearest_neighbour(dist, demands, n_vehicles, capacity):
    """The dispatcher's fallback: repeatedly drive to the closest unserved stop.

    This is the honest thing to compare against — not a random ordering. It is what a
    reasonable person does without a solver, and it is already fairly good.
    """
    n = len(dist)
    unvisited = set(range(1, n))
    routes, total = [], 0
    for v in range(n_vehicles):
        if not unvisited:
            break
        cur, load, path, dist_m = 0, 0, [], 0
        while True:
            candidates = [j for j in unvisited if load + demands[j] <= capacity]
            if not candidates:
                break
            nxt = min(candidates, key=lambda j: dist[cur][j])
            dist_m += dist[cur][nxt]
            load += demands[nxt]
            path.append(nxt - 1)
            unvisited.discard(nxt)
            cur = nxt
        if path:
            dist_m += dist[cur][0]
            routes.append({"vehicle": v, "path": path, "distance_m": dist_m, "load": int(load)})
            total += dist_m
    return routes, total, len(unvisited)


@solver_env.task(report=True)
async def build_routes(
    zones_dir: flyte.io.Dir,
    forecast_json: str,
    vehicles: int = 6,
    capacity_slack: float = 1.25,
    solver_seconds: int = 15,
) -> str:
    """
    Turn forecast demand into a routing plan, and compare it to the naive alternative.

    Demand per zone is the forecast **peak** half-hour over the horizon — a fleet sized to
    the average is under water exactly when it matters.
    """
    import numpy as np

    local = await zones_dir.download()
    with open(os.path.join(local, "zones.json")) as f:
        data = json.load(f)
    zones = data["zones"]
    depot = data["depot"]

    fc = json.loads(forecast_json)
    results = fc["results"]

    await flyte.report.replace.aio(rh.wrap_report(
        f"<h2>Routing</h2><p>Solving a {len(results)}-stop problem for {vehicles} vehicles…</p>"
    ), do_flush=True)

    served = [zones[r["zone"]] for r in results]
    demands = [0] + [max(1, int(round(r["peak"]))) for r in results]  # index 0 == depot
    points = [(depot["lat"], depot["lng"])] + [(z["lat"], z["lng"]) for z in served]
    dist = distance_matrix(points)

    total_demand = sum(demands)
    capacity = int(math.ceil(total_demand / vehicles * capacity_slack))

    opt_routes, opt_total = _solve_vrp(dist, demands, vehicles, capacity, solver_seconds)
    naive_routes, naive_total, unserved = _nearest_neighbour(dist, demands, vehicles, capacity)

    if opt_routes is None:
        raise RuntimeError(
            f"OR-Tools found no feasible solution for {len(results)} stops / {vehicles} "
            f"vehicles at capacity {capacity}. Try more vehicles or a larger capacity_slack."
        )

    saved_m = naive_total - opt_total
    saved_pct = 100.0 * saved_m / naive_total if naive_total else 0.0

    map_zones = [
        {"lat": z["lat"], "lng": z["lng"], "demand": demands[i + 1]}
        for i, z in enumerate(served)
    ]

    rows = "".join(
        f"<tr><td><span class='badge badge-info'>Vehicle {r['vehicle']}</span></td>"
        f"<td>{len(r['path'])}</td><td>{r['load']}</td>"
        f"<td>{r['distance_m']/1000:.2f} km</td>"
        f"<td>{100*r['load']/capacity:.0f}%</td></tr>"
        for r in opt_routes
    )

    html = f"""
    <h2>Routing plan — {len(results)} zones, {len(opt_routes)} vehicles</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{opt_total/1000:.1f} km</div><div class="label">Optimized total</div></div>
      <div class="stat"><div class="value">{naive_total/1000:.1f} km</div><div class="label">Nearest-neighbour</div></div>
      <div class="stat"><div class="value">{saved_pct:.1f}%</div><div class="label">Distance saved</div>
        <div class="delta {'up' if saved_m > 0 else 'down'}">{saved_m/1000:+.1f} km</div></div>
      <div class="stat"><div class="value">{len(opt_routes)}</div><div class="label">Vehicles used</div></div>
      <div class="stat"><div class="value">{capacity}</div><div class="label">Capacity each</div></div>
      <div class="stat"><div class="value">{total_demand}</div><div class="label">Total peak demand</div></div>
    </div>

    <div class="chart-container">
      {rh.route_map(map_zones, opt_routes, depot, title="Optimized routes (OR-Tools, guided local search)")}
      {rh.vehicle_legend(opt_routes)}
    </div>

    <h3>Optimized vs nearest-neighbour</h3>
    <div class="grid2">
      <div class="chart-container">
        {rh.route_map(map_zones, naive_routes, depot, title=f"Nearest-neighbour — {naive_total/1000:.1f} km", animate=False, height=440)}
      </div>
      <div class="chart-container">
        {rh.route_map(map_zones, opt_routes, depot, title=f"Optimized — {opt_total/1000:.1f} km", animate=False, height=440)}
      </div>
    </div>

    <div class="chart-container">
      {rh.make_bar_chart(["Nearest-neighbour", "OR-Tools optimized"],
                         [naive_total/1000, opt_total/1000],
                         colors=["#94a3b8", "#0284c7"],
                         title="Total fleet distance — lower is better",
                         value_format=".1f", y_label="km", lower_is_better=True)}
    </div>

    <h3>Per-vehicle plan</h3>
    <table>
      <tr><th>Vehicle</th><th>Stops</th><th>Load</th><th>Distance</th><th>Utilization</th></tr>
      {rows}
    </table>

    <div class="note">
      The baseline is <b>nearest-neighbour</b>, not a random route — it is what a dispatcher
      does without a solver, and it is already decent, so the {saved_pct:.1f}% gap is a
      realistic estimate of what optimization actually buys rather than a strawman.
      Distances are great-circle between real coordinates; a production system would
      substitute a road-network engine, which raises both numbers without changing the
      comparison.
      {f"<br><b>Note:</b> nearest-neighbour left {unserved} stops unserved within capacity." if unserved else ""}
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)

    return json.dumps({
        "optimized_km": opt_total / 1000,
        "naive_km": naive_total / 1000,
        "saved_km": saved_m / 1000,
        "saved_pct": saved_pct,
        "vehicles_used": len(opt_routes),
        "capacity": capacity,
        "stops": len(results),
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    n_zones: int = 60,
    subset: str = "january_2015",
    vehicles: int = 6,
    horizon: int = HORIZON,
    batch_size: int = 12,
    solver_seconds: int = 15,
    capacity_slack: float = 1.25,
) -> tuple[str, str]:
    """Forecast demand, then route a fleet against it. Returns (forecast JSON, routing JSON)."""

    async def step(n: int, note: str):
        await flyte.report.replace.aio(
            rh.wrap_report(f"<h2>Demand &rarr; Routing</h2>"
                           f"{rh.progress_html(PIPELINE_STEPS, n, note)}"),
            do_flush=True,
        )

    await step(1, "Loading real Manhattan pickup demand…")
    zones_dir = await load_demand(n_zones=n_zones, subset=subset)

    await step(2, f"Forecasting {n_zones} zones 24 h ahead, zero-shot…")
    batches = [list(range(i, min(i + batch_size, n_zones)))
               for i in range(0, n_zones, batch_size)]
    batch_results = await asyncio.gather(*[
        forecast_batch(zones_dir=zones_dir, zone_indices=b, horizon=horizon)
        for b in batches
    ])
    forecast_json = await summarize_forecasts(zones_dir=zones_dir, batches=list(batch_results))

    await step(3, "Solving the vehicle routing problem…")
    routing_json = await build_routes(
        zones_dir=zones_dir,
        forecast_json=forecast_json,
        vehicles=vehicles,
        capacity_slack=capacity_slack,
        solver_seconds=solver_seconds,
    )

    await step(4, "Summarizing…")
    fc = json.loads(forecast_json)
    rt = json.loads(routing_json)

    await flyte.report.replace.aio(rh.wrap_report(f"""
      <h2>Pipeline Complete</h2>
      <div class="stat-grid">
        <div class="stat"><div class="value">{rt['stops']}</div><div class="label">Zones</div></div>
        <div class="stat"><div class="value">{fc['improvement_pct']:+.1f}%</div><div class="label">Forecast error reduction</div></div>
        <div class="stat"><div class="value">{fc['wins']}/{rt['stops']}</div><div class="label">Zones beating baseline</div></div>
        <div class="stat"><div class="value">{rt['optimized_km']:.1f} km</div><div class="label">Optimized route</div></div>
        <div class="stat"><div class="value">{rt['saved_pct']:.1f}%</div><div class="label">Distance saved</div></div>
        <div class="stat"><div class="value">{rt['vehicles_used']}</div><div class="label">Vehicles</div></div>
      </div>
      <div class="card">
        <b>Forecast:</b> Chronos-Bolt zero-shot, 24 h horizon &nbsp;|&nbsp;
        <b>Routing:</b> OR-Tools CVRP, {rt['capacity']} capacity/vehicle &nbsp;|&nbsp;
        <b>Compute:</b> CPU only
      </div>
      <div class="note">
        Two very different kinds of model in one pipeline: a pretrained transformer that was
        never fitted to this data, feeding a classical constraint solver. Open the task
        reports for the demand heatmap, per-zone forecast intervals, and the animated
        route map.
      </div>
    """), do_flush=True)

    return forecast_json, routing_json
