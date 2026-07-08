"""Lance vs Parquet for training data — a standalone micro-benchmark.

Separate from the production pipeline (`workflow.py`) on purpose: this isn't part
of the ML workflow, it's a focused, honest comparison of *why* Lance beats
Parquet (and tar/WebDataset) for feeding a training loop.

The nuance most format demos miss: for **sequential** scans, Parquet is columnar
and streams just fine — Lance is comparable, not magic. The real difference is
what training actually needs: **shuffling / random access**.

  1. Sequential scan     — Parquet ≈ Lance (both fast). Lance streams as well.
  2. Shuffle an epoch     — Parquet must load the WHOLE dataset into memory to
                            shuffle; Lance holds one batch at a time via random
                            access (`Dataset.take`). Shown as working-set memory.
  3. Random-sample fetch  — grab N random rows: Lance reads only those; Parquet
                            has no point access, so it reads the whole file.

Run:
    uv run flyte run --local format_comparison.py compare_formats
    uv run flyte run format_comparison.py compare_formats
"""

from __future__ import annotations

import os
import random
import tempfile
import time

import flyte
import flyte.report
from flyte.io import Dir

import lance_dataset as lds
from config import cpu_env
from report_helpers import bar_chart, stat, wrap_report


# ----------------------------------------------------------------------------
# Build the same dataset in both formats
# ----------------------------------------------------------------------------
@cpu_env.task(report=True)
async def build_formats(num_images: int = 4000, img_size: int = 128, seed: int = 0) -> Dir:
    """Generate one synthetic image dataset and write it to both a Parquet file
    and a Lance dataset (identical rows: encoded PNG + a label)."""
    import lance
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    images, labels = [], []
    for _ in range(num_images):
        scene = lds.render_scene(rng, img_size)
        images.append(scene.image_png)
        labels.append(len(scene.bboxes))
    table = pa.table({"image": images, "num_objects": labels})

    root = tempfile.mkdtemp(prefix="formats_")
    pq_path = os.path.join(root, "data.parquet")
    lance_path = os.path.join(root, "data.lance")
    pq.write_table(table, pq_path)
    lance.write_dataset(table, lance_path)

    pq_mb = os.path.getsize(pq_path) / 1e6
    lance_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _d, fs in os.walk(lance_path)
        for f in fs
    ) / 1e6
    html = f"""
    <h2>Same dataset, two formats</h2>
    <p>{num_images:,} synthetic images (encoded PNG + label) written to a Parquet
    file and a Lance dataset — identical rows, so any read difference is the
    format, not the data.</p>
    <div class="stat-grid">
      {stat(f"{num_images:,}", "rows")}
      {stat(f"{pq_mb:.1f} MB", "Parquet on disk")}
      {stat(f"{lance_mb:.1f} MB", "Lance on disk")}
    </div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)

    return await Dir.from_local(root)


# ----------------------------------------------------------------------------
# Compare read patterns
# ----------------------------------------------------------------------------
@cpu_env.task(report=True)
async def compare(formats_dir: Dir, batch_size: int = 256, random_samples: int = 256) -> dict:
    """Read the same rows out of Parquet and Lance three ways and report the
    differences that matter for training."""
    import lance
    import pyarrow.parquet as pq

    root = await formats_dir.download()
    pq_path = os.path.join(root, "data.parquet")
    lance_path = os.path.join(root, "data.lance")
    ds = lance.dataset(lance_path)
    n = ds.count_rows()

    # --- 1. Sequential scan throughput (decode every image once, in order) ---
    t0 = time.perf_counter()
    c = 0
    for b in pq.ParquetFile(pq_path).iter_batches(batch_size=batch_size, columns=["image"]):
        for png in b.column("image").to_pylist():
            lds.decode_image(png)
            c += 1
    pq_seq = c / (time.perf_counter() - t0)

    t0 = time.perf_counter()
    c = 0
    for b in ds.scanner(columns=["image"], batch_size=batch_size).to_batches():
        for png in b.column("image").to_pylist():
            lds.decode_image(png)
            c += 1
    lance_seq = c / (time.perf_counter() - t0)

    # --- 2. Working set to shuffle one epoch (bytes held in memory) ---
    # Parquet has no lazy random access, so to shuffle you materialize the whole
    # table. Lance takes a random batch at a time and holds only that.
    pq_working_set = pq.read_table(pq_path, columns=["image"]).nbytes
    order = list(range(n))
    random.shuffle(order)
    lance_working_set = 0
    for i in range(0, n, batch_size):
        tbl = ds.take(order[i : i + batch_size], columns=["image"])
        lance_working_set = max(lance_working_set, tbl.nbytes)

    # --- 3. Random-sample fetch: grab `random_samples` rows at random indices ---
    idx = [random.randrange(n) for _ in range(random_samples)]
    t0 = time.perf_counter()
    ds.take(idx, columns=["image"])  # reads only those rows
    lance_rand = time.perf_counter() - t0
    t0 = time.perf_counter()
    pq.read_table(pq_path, columns=["image"]).take(idx)  # must read the whole file
    pq_rand = time.perf_counter() - t0

    result = {
        "rows": n,
        "seq_img_per_s": {"parquet": round(pq_seq, 1), "lance": round(lance_seq, 1)},
        "shuffle_working_set_mb": {
            "parquet": round(pq_working_set / 1e6, 1),
            "lance": round(lance_working_set / 1e6, 1),
        },
        "random_fetch_ms": {
            "parquet": round(pq_rand * 1e3, 1),
            "lance": round(lance_rand * 1e3, 1),
        },
    }

    seq_chart = bar_chart(
        ["Parquet", "Lance"], [pq_seq, lance_seq],
        title="1. Sequential scan — throughput (img/s)", unit=" img/s",
        colors=["#adb5bd", "#06d6a0"],
    )
    ws_chart = bar_chart(
        ["Parquet", "Lance"],
        [pq_working_set / 1e6, lance_working_set / 1e6],
        title="2. Shuffle an epoch — memory you must hold (MB)", unit=" MB",
        decimals=1, colors=["#adb5bd", "#06d6a0"],
    )
    rand_chart = bar_chart(
        ["Parquet", "Lance"], [pq_rand * 1e3, lance_rand * 1e3],
        title=f"3. Fetch {random_samples} random rows — time (ms, lower is better)",
        unit=" ms", decimals=1, colors=["#adb5bd", "#06d6a0"],
    )

    ws_ratio = pq_working_set / lance_working_set if lance_working_set else 0
    rand_ratio = pq_rand / lance_rand if lance_rand else 0
    html = f"""
    <h2>Lance vs Parquet for training data</h2>
    <p>Same {n:,} rows, read three ways. Sequential scanning is a near tie — both
    are columnar. The gap opens where training actually lives: <b>shuffling and
    random access</b>.</p>

    <h3>1. Sequential scan — a tie</h3>
    <p>Streaming the whole dataset in order, Parquet and Lance are comparable.
    Lance streams as well as Parquet; it doesn't give that up to get the rest.</p>
    <div class="chart-container">{seq_chart}</div>

    <h3>2. Shuffling an epoch — Lance holds a batch, Parquet holds everything</h3>
    <p>SGD needs shuffled batches. Parquet has no lazy random access, so you must
    load the <b>whole dataset</b> into memory to shuffle it. Lance takes a random
    batch at a time — its working set is <span class="win">~{ws_ratio:.0f}× smaller</span>,
    and it works on datasets far larger than RAM (and straight off object storage).</p>
    <div class="chart-container">{ws_chart}</div>

    <h3>3. Random access — Lance reads only what you ask for</h3>
    <p>Fetching {random_samples} random rows, Lance reads just those; Parquet has
    to scan the whole file. Lance is <span class="win">~{rand_ratio:.0f}× faster</span>
    here — the same property that makes shuffled streaming cheap.</p>
    <div class="chart-container">{rand_chart}</div>

    <div class="note">Sequential parity + cheap random access is exactly what a
    training loader wants, and what tar/WebDataset (stream-only) and Parquet
    (scan-only) can't both give you. This is why the main pipeline stores training
    data as Lance and shuffles it with <code>Dataset.take</code>.</div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)

    return result


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
@cpu_env.task(report=True)
async def compare_formats(
    num_images: int = 4000,
    img_size: int = 128,
    batch_size: int = 256,
    random_samples: int = 256,
) -> dict:
    """Build the dataset in both formats, then compare read patterns."""
    formats_dir = await build_formats(num_images=num_images, img_size=img_size)
    result = await compare(formats_dir, batch_size=batch_size, random_samples=random_samples)

    ws = result["shuffle_working_set_mb"]
    rnd = result["random_fetch_ms"]
    html = f"""
    <h2>Lance vs Parquet — summary</h2>
    <div class="stat-grid">
      {stat(f"{result['rows']:,}", "rows compared")}
      {stat(f"{ws['parquet']:.0f} → {ws['lance']:.0f} MB", "shuffle working set (pq→lance)")}
      {stat(f"{rnd['parquet']:.0f} → {rnd['lance']:.0f} ms", "random fetch (pq→lance)")}
    </div>
    <div class="note">Sequential: a tie. Shuffling & random access: Lance wins by
    a wide margin — see the <b>compare</b> report tab for the charts.</div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)

    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(compare_formats)
    print(run.url)
