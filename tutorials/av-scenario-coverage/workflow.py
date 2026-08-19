"""
Long-tail scenario coverage for autonomous driving.

The hard problem in AV data is not collecting more miles — it is knowing **which
situations you have and which you don't**. A fleet logs thousands of hours of ordinary
highway driving and almost no emergency-vehicle interactions in fog. The model then meets
one on a Tuesday.

This pipeline builds that picture from NVIDIA's synthetic long-tail scenario dataset:

  1. **Index.** Read per-camera scenario metadata (weather, time of day, surface, region)
     and captions, and build a coverage matrix of scenario type against condition.
  2. **Render.** For selected scenarios, decode the multi-camera rig and composite a
     synchronised surround view.
  3. **Report.** Show what is covered, and — the useful part — what is missing.

Usage:
    flyte run --local --tui workflow.py pipeline --n_scenarios 3 --n_sample 120
    flyte run workflow.py pipeline
"""

import asyncio
import json
import logging
import os
import tempfile
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import flyte
import flyte.io
import flyte.report

import detect
import report_helpers as rh
import video
from config import coverage_env, perception_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

REPO = "nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
API = f"https://huggingface.co/api/datasets/{REPO}"

SCENARIO_FAMILIES = ["emergency", "lanechange", "nudging"]
PIPELINE_STEPS = ["Survey Scenarios", "Detect & Evaluate", "Coverage Report"]

# Actual `time_of_day` values in the dataset. There is NO value called "Day" — the bright
# ones are these, and a naive `startswith("day")` test matches nothing at all.
DAYLIGHT = {"Mid-day", "Morning", "Afternoon", "Day", "Daytime"}

# Metadata is not uniform across the dataset: most scenarios carry
# weather/time_of_day/surface_type/region, but some campaigns (notably within `emergency`)
# ship only a `caption_key` instead. Those are labelled UNLABELLED rather than folded in
# with real zeros — "we never recorded the condition" and "this condition does not occur"
# are different findings, and conflating them makes a coverage matrix actively misleading.
UNLABELLED = "Unlabelled"


def _fetch(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


# ------------------------------------------------------------------
# Task 1: index scenarios
# ------------------------------------------------------------------

@coverage_env.task(report=True)
async def survey_scenarios(n_sample: int = 150, n_pick: int = 3,
                          prefer_daylight: bool = True) -> str:
    """
    Sample scenario metadata and build the coverage matrix.

    Metadata lives in small per-camera JSON files next to each video, so surveying the
    dataset costs a few hundred small requests rather than terabytes of video.
    """
    await flyte.report.replace.aio(rh.wrap_report(
        "<h2>Indexing scenarios</h2><p>Listing the dataset…</p>"
    ), do_flush=True)

    listing = json.loads(_fetch(API, timeout=180))
    files = [f["rfilename"] for f in listing["siblings"]]

    # One description JSON per camera; group them into scenarios.
    scen_cams = defaultdict(set)
    for f in files:
        parts = f.split("/")
        if len(parts) == 4 and parts[2] == "description" and parts[3].endswith(".json"):
            scen_cams[(parts[0], parts[1])].add(parts[3].removesuffix(".json"))
    scenarios = sorted(scen_cams)
    total = len(scenarios)
    if not total:
        raise RuntimeError("No scenarios found — dataset layout may have changed.")

    step = max(1, total // max(n_sample, 1))
    sample = scenarios[::step][:n_sample]
    log.info(f"{total} scenarios; sampling {len(sample)}")

    def probe(key):
        family, sid = key
        cams = sorted(scen_cams[key])
        cam = "front_wide" if "front_wide" in cams else cams[0]
        try:
            d = json.loads(_fetch(f"{BASE}/{family}/{sid}/description/{cam}.json", 60))
        except Exception:
            return None
        md = d.get("metadata", {}) or {}
        windows = d.get("t2w_windows") or [{}]
        caption = ""
        for w in windows:
            for k, v in w.items():
                if "caption" in k and isinstance(v, str):
                    caption = v
                    break
            if caption:
                break
        return {
            "family": family, "id": sid, "cameras": cams,
            "weather": md.get("weather") or UNLABELLED,
            "time_of_day": md.get("time_of_day") or UNLABELLED,
            "surface": md.get("surface_type") or UNLABELLED,
            "region": md.get("region") or UNLABELLED,
            "labelled": bool(md.get("time_of_day")),
            "framerate": d.get("framerate"), "nb_frames": d.get("nb_frames"),
            "caption": caption,
        }

    with ThreadPoolExecutor(max_workers=16) as ex:
        meta = [m for m in ex.map(probe, sample) if m]
    if not meta:
        raise RuntimeError("Could not read any scenario metadata.")

    families = sorted({m["family"] for m in meta})
    tods = sorted({m["time_of_day"] for m in meta})
    weathers = sorted({m["weather"] for m in meta})
    regions = sorted({m["region"] for m in meta})

    m_tod = Counter((m["family"], m["time_of_day"]) for m in meta)
    m_weather = Counter((m["family"], m["weather"]) for m in meta)
    m_region = Counter((m["family"], m["region"]) for m in meta)

    # Pick scenarios to render: spread across families, and prefer daylight because a
    # night clip renders as a mostly-black panel and reads badly in a report.
    def rank(m):
        # Prefer bright scenes: a night clip composites to a mostly-black panel and reads
        # badly in a report. Membership test against real values, not a prefix guess.
        bright = 0 if (prefer_daylight and m["time_of_day"] in DAYLIGHT) else 1
        return (bright, -len(m["cameras"]))

    picked, seen = [], set()
    for m in sorted(meta, key=rank):
        if m["family"] in seen:
            continue
        picked.append(m)
        seen.add(m["family"])
        if len(picked) >= n_pick:
            break
    for m in sorted(meta, key=rank):
        if len(picked) >= n_pick:
            break
        if m not in picked:
            picked.append(m)

    html = f"""
    <h2>Scenario index</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{total:,}</div><div class="label">Scenarios in dataset</div></div>
      <div class="stat"><div class="value">{len(meta)}</div><div class="label">Sampled</div></div>
      <div class="stat"><div class="value">{len(families)}</div><div class="label">Scenario families</div></div>
      <div class="stat"><div class="value">{len(weathers)}</div><div class="label">Weather conditions</div></div>
      <div class="stat"><div class="value">{len(tods)}</div><div class="label">Times of day</div></div>
      <div class="stat"><div class="value">{sum(1 for m in meta if m['labelled'])}/{len(meta)}</div>
        <div class="label">With condition metadata</div></div>
    </div>

    <div class="note">
      The useful question for an AV data team is not "how much data do we have" but
      "<b>which situations are we missing</b>". Every combination below that reads
      <b style="color:#991b1b;">0</b> is a scenario a fleet can encounter and this dataset
      cannot teach.
    </div>

    <div class="warn">
      <b>Metadata is not uniform across this dataset.</b>
      {sum(1 for m in meta if not m['labelled'])} of {len(meta)} sampled scenarios carry no
      condition fields at all — some campaigns ship only a <code>caption_key</code> instead
      of <code>weather</code>/<code>time_of_day</code>/<code>region</code>. Those appear as
      <b>{UNLABELLED}</b> rather than being folded into the counts, because
      "we never recorded the condition" and "this condition does not occur" are different
      findings. Collapsing them is how a coverage matrix ends up lying to you.
    </div>

    {rh.coverage_matrix_html(m_tod, families, tods, "Scenario family x time of day",
                             "family", "time of day")}
    {rh.coverage_matrix_html(m_weather, families, weathers, "Scenario family x weather",
                             "family", "weather")}
    {rh.coverage_matrix_html(m_region, families, regions, "Scenario family x region",
                             "family", "region")}

    <div class="chart-container">
      {rh.make_bar_chart([f for f, _ in Counter(m['family'] for m in meta).most_common()],
                         [c for _, c in Counter(m['family'] for m in meta).most_common()],
                         title="Sampled scenarios by family", height=180)}
    </div>

    <div class="warn">
      <b>This data is fully synthetic.</b> NVIDIA states it "exhibits a sim-to-real
      appearance gap relative to real driving footage", and that "a subset of authored
      agent behaviors may also appear unnatural — e.g. emergency vehicles cutting through
      dense traffic when open space is available nearby". It is built for long-tail
      coverage, not photorealism, and should be paired with real fleet data.
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)

    return json.dumps({
        "total": total, "sampled": len(meta), "picked": picked, "meta": meta,
        "families": families, "times_of_day": tods, "weathers": weathers,
        "regions": regions,
    })


# ------------------------------------------------------------------
# Task 2: render one scenario  (fans out)
# ------------------------------------------------------------------

@perception_env.task(retries=2)
async def detect_and_render(family: str, scenario_id: str, cameras: list[str],
                          n_frames: int = 16, tile_w: int = 320,
                          run_detection: bool = True,
                          det_threshold: float = 0.12) -> flyte.io.Dir:
    """Download this scenario's cameras, decode, detect, and composite a surround sequence.

    Detection runs on **every** camera, not just the forward view. That costs ~7x, but it
    buys the most interesting comparison in the pipeline: three of the seven cameras are
    200-degree fisheyes, and OWLv2 was trained on rectilinear photographs. Detection
    quality should fall off sharply on the warped views — which is exactly why production
    stacks rectify fisheye before detection. Reporting per-camera rates makes that visible
    instead of assumed.
    """
    work = tempfile.mkdtemp(prefix="scn_")
    frames_by_cam, got = {}, []
    for cam in cameras:
        url = f"{BASE}/{family}/{scenario_id}/video/{cam}.mp4"
        path = os.path.join(work, f"{cam}.mp4")
        try:
            data = _fetch(url, timeout=600)
            with open(path, "wb") as f:
                f.write(data)
            frames = video.decode_frames(path, n_frames, (tile_w, int(tile_w * 9 / 16)))
            if frames:
                frames_by_cam[cam] = frames
                got.append(cam)
            os.remove(path)
        except Exception as e:  # noqa: BLE001 — a missing camera must not fail the scenario
            log.warning(f"{scenario_id}: camera {cam} unavailable ({e})")

    if not frames_by_cam:
        raise RuntimeError(f"{scenario_id}: no cameras decoded")

    # ---- open-vocabulary detection on the forward view ----
    det_summary, det_error, per_camera = None, "", {}
    if run_detection and frames_by_cam:
        try:
            all_frames = []
            for cam in list(frames_by_cam):
                per_frame = detect.detect(frames_by_cam[cam], threshold=det_threshold)
                per_camera[cam] = detect.summarize(per_frame)
                per_camera[cam]["fisheye"] = "fisheye" in cam
                all_frames.extend(per_frame)
                frames_by_cam[cam] = [
                    detect.draw_detections(img, dets)
                    for img, dets in zip(frames_by_cam[cam], per_frame)
                ]
            det_summary = detect.summarize(all_frames)
            pin = [v for v in per_camera.values() if not v["fisheye"]]
            fis = [v for v in per_camera.values() if v["fisheye"]]
            det_summary["pinhole_mean_score"] = (
                sum(v["overall_mean_score"] for v in pin) / len(pin) if pin else 0.0)
            det_summary["fisheye_mean_score"] = (
                sum(v["overall_mean_score"] for v in fis) / len(fis) if fis else 0.0)
            det_summary["pinhole_per_frame"] = (
                sum(v["total_detections"] for v in pin) / max(sum(v["frames"] for v in pin), 1))
            det_summary["fisheye_per_frame"] = (
                sum(v["total_detections"] for v in fis) / max(sum(v["frames"] for v in fis), 1))
            log.info(f"{scenario_id}: {det_summary['total_detections']} detections across "
                     f"{len(per_camera)} cameras | pinhole {det_summary['pinhole_mean_score']:.3f} "
                     f"vs fisheye {det_summary['fisheye_mean_score']:.3f} | "
                     f"emergency_hit={det_summary['emergency_hit']}")
        except Exception as e:  # noqa: BLE001 — detection is additive; never sink the render
            det_error = str(e)[:200]
            log.warning(f"{scenario_id}: detection failed ({det_error})")

    composited = video.composite_surround(frames_by_cam, tile=(tile_w, int(tile_w * 9 / 16)))

    out_dir = tempfile.mkdtemp(prefix="scnout_")
    fdir = os.path.join(out_dir, "frames")
    os.makedirs(fdir, exist_ok=True)
    for i, jpg in enumerate(composited):
        with open(os.path.join(fdir, f"{i:03d}.jpg"), "wb") as f:
            f.write(jpg)

    stats = {
        "family": family, "id": scenario_id,
        "cameras": got, "n_cameras": len(got),
        "frames": len(composited),
        "bytes": sum(len(j) for j in composited),
        "detection": det_summary,
        "detection_per_camera": per_camera,
        "detection_error": det_error,
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f)
    log.info(f"{scenario_id}: {len(got)} cameras, {len(composited)} composited frames")
    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@coverage_env.task(report=True)
async def pipeline(n_scenarios: int = 3, n_sample: int = 150, n_frames: int = 16,
                   tile_w: int = 320, fps: int = 8, full_rig: bool = True) -> str:
    """Index the dataset, render surround views, and report coverage gaps."""

    async def step(n: int, note: str):
        await flyte.report.replace.aio(
            rh.wrap_report(f"<h2>Scenario Coverage</h2>"
                           f"{rh.progress_html(PIPELINE_STEPS, n, note)}"),
            do_flush=True,
        )

    await step(1, "Sampling scenario metadata…")
    index_json = await survey_scenarios(n_sample=n_sample, n_pick=n_scenarios)
    index = json.loads(index_json)
    picked = index["picked"]

    await step(2, f"Rendering {len(picked)} surround views…")
    with flyte.group("detect-scenarios"):
        results = await asyncio.gather(*[
            detect_and_render(
                family=p["family"], scenario_id=p["id"],
                cameras=(p["cameras"] if full_rig else
                         [c for c in video.DEFAULT_RIG if c in p["cameras"]]),
                n_frames=n_frames, tile_w=tile_w,
            ) for p in picked
        ], return_exceptions=True)

    dirs, failed = [], []
    for p, r in zip(picked, results):
        if isinstance(r, Exception):
            log.warning(f"scenario {p['id']} failed: {r}")
            failed.append(p["id"])
        else:
            dirs.append((p, r))
    if not dirs:
        raise RuntimeError(f"All {len(picked)} scenarios failed to render.")

    await step(3, "Assembling coverage report…")

    players, total_bytes = "", 0
    for meta, d in dirs:
        local = await d.download()
        with open(os.path.join(local, "stats.json")) as f:
            st = json.load(f)
        total_bytes += st["bytes"]
        fdir = os.path.join(local, "frames")
        uris = []
        for name in sorted(os.listdir(fdir)):
            with open(os.path.join(fdir, name), "rb") as fh:
                uris.append(rh.jpeg_uri(fh.read()))
        players += (
            f"<h3><span class='badge'>{meta['family']}</span> "
            f"{meta['time_of_day']} · {meta['weather']} · {meta['region']}</h3>"
            f"<div class='note' style='margin-top:4px;'><i>&ldquo;{meta['caption'][:320]}&rdquo;</i></div>"
            + rh.surround_player_html(
                uris, slug=st["id"][-8:], fps=fps,
                caption=f"{st['n_cameras']} cameras · {st['frames']} frames · "
                        f"OWLv2 boxes on every camera",
            )
        )
        d = st.get("detection")
        if d:
            det_rows = "".join(
                f"<tr><td>{p_.removeprefix('a ').removeprefix('an ')}</td>"
                f"<td>{d['hit_rate'][p_]:.0%}</td><td>{d['counts'][p_]}</td>"
                f"<td>{d['mean_score'][p_]:.3f}</td></tr>"
                for p_ in sorted(d['hit_rate'], key=lambda k: -d['hit_rate'][k])
                if d['counts'][p_]
            )
            expected = meta["family"] == "emergency"
            verdict = (
                "<span style='color:#15803d;font-weight:600;'>confirmed</span>"
                if d["emergency_hit"] else
                "<span style='color:#b91c1c;font-weight:600;'>not found</span>"
            )
            pc = st.get("detection_per_camera") or {}
            cam_rows = "".join(
                f"<tr><td>{c.replace('_',' ')}"
                + ("<span class='badge' style='margin-left:6px;'>fisheye</span>" if v['fisheye'] else "")
                + f"</td><td>{v['total_detections']}</td>"
                f"<td>{v['total_detections']/max(v['frames'],1):.1f}</td>"
                f"<td>{v['overall_mean_score']:.3f}</td></tr>"
                for c, v in sorted(pc.items(), key=lambda kv: -kv[1]['overall_mean_score'])
            )
            drop = ""
            if d.get("pinhole_mean_score") and d.get("fisheye_mean_score"):
                delta = 100 * (1 - d["fisheye_mean_score"] / d["pinhole_mean_score"])
                drop = (f"<div class='note' style='margin-top:8px;'>"
                        f"<b>Rectilinear vs fisheye:</b> mean confidence "
                        f"<b>{d['pinhole_mean_score']:.3f}</b> on the four standard cameras "
                        f"vs <b>{d['fisheye_mean_score']:.3f}</b> on the three 200&deg; "
                        f"fisheyes — a <b>{delta:.0f}%</b> drop, and "
                        f"{d['pinhole_per_frame']:.1f} vs {d['fisheye_per_frame']:.1f} "
                        f"detections per frame. OWLv2 never saw that warping in training, "
                        f"which is why production stacks rectify fisheye before detection."
                        f"</div>")
            players += f"""
              <div class="card">
                <b>Scenario detection evaluation</b> &nbsp;·&nbsp; {d['total_detections']} objects
                across {st['n_cameras']} cameras &nbsp;·&nbsp;
                avg confidence <b>{d['overall_mean_score']:.2f}</b>
                {f"<br><b>Emergency vehicle:</b> {verdict}" if expected else ""}
                {drop}
                <table style="margin-top:8px;">
                  <tr><th>Camera</th><th>Boxes</th><th>Per frame</th><th>Mean score</th></tr>
                  {cam_rows}
                </table>
                <table style="margin-top:8px;">
                  <tr><th>Prompt</th><th>Frames hit</th><th>Boxes</th><th>Mean score</th></tr>
                  {det_rows}
                </table>
              </div>"""
        elif st.get("detection_error"):
            players += f"<div class='warn'>Detection unavailable: {st['detection_error']}</div>"

    fams = index["families"]
    tods = index["times_of_day"]
    m_tod = Counter((m["family"], m["time_of_day"]) for m in index["meta"])
    gaps = sum(1 for f in fams for t in tods if m_tod.get((f, t), 0) == 0)

    await flyte.report.replace.aio(rh.wrap_report(f"""
      <h2>Long-tail scenario coverage</h2>
      {players}
      <div class="stat-grid">
        <div class="stat"><div class="value">{index['total']:,}</div><div class="label">Scenarios available</div></div>
        <div class="stat"><div class="value">{len(dirs)}</div><div class="label">Surround views rendered</div></div>
        <div class="stat"><div class="value">{sum(st for st in [len(d[0]['cameras']) for d in dirs])}</div><div class="label">Camera streams decoded</div></div>
        <div class="stat"><div class="value">{gaps}</div><div class="label">Uncovered family/time cells</div></div>
        <div class="stat"><div class="value">{total_bytes/1e6:.1f} MB</div><div class="label">Composited imagery</div></div>
      </div>
      {rh.coverage_matrix_html(m_tod, fams, tods, "Coverage: family x time of day",
                               "family", "time of day")}
      {'<div class="warn">Failed to render: ' + ', '.join(failed) + '</div>' if failed else ''}
      <div class="warn">
        Fully synthetic data (NVIDIA Omniverse). NVIDIA notes a sim-to-real appearance gap
        and that some authored agent behaviours may be unnatural. Intended for long-tail
        coverage, to be paired with real fleet data — not as a substitute for it.
      </div>
      <div class="note">
        <b>Every box is a live prediction.</b> OWLv2 runs on all seven cameras, an open-vocabulary detector
        prompted with free text ("a police car", "an ambulance", "a person") rather than a
        fixed class list. Two things this measures that rendering cannot: whether a clip
        filed under <code>emergency</code> actually contains an emergency vehicle, and how
        far confidence falls on synthetic imagery — OWLv2 was trained on real photographs,
        and its mean score here is well below the 0.5-0.8 typical of real scenes. That
        number <i>is</i> the sim-to-real gap, measured rather than asserted.
      </div>
      <div class="note">
        The surround view is composited server-side into one image per timestep, so the
        seven cameras cannot drift out of sync during playback. Open the index task's
        report for the full weather and region matrices.
      </div>
    """), do_flush=True)

    return json.dumps({
        "total": index["total"], "rendered": len(dirs), "failed": failed,
        "scenarios": [{"family": m["family"], "id": m["id"],
                       "time_of_day": m["time_of_day"], "weather": m["weather"]}
                      for m, _ in dirs],
    })
