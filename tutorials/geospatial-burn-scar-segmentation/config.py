import flyte

# NOTE: `.with_pip_packages()` rather than `.with_requirements()` — the latter stores a
# *relative* path and re-resolves it at runtime, which breaks anywhere the working
# directory isn't yours (most visibly when an app pod launches a task). Keep
# requirements.txt for the local venv and keep the two in sync by hand.
base_image = (
    flyte.Image.from_debian_base(name="geo-burn-scar-v2", python_version=(3, 12))
    # rasterio's manylinux wheel bundles GDAL and 43 of its shared libraries, but still
    # links against two it expects from the OS. The Debian base is slim enough not to
    # have libexpat, so `import rasterio` dies with
    # "libexpat.so.1: cannot open shared object file". Verified by parsing DT_NEEDED out
    # of the wheel's ELF headers: libexpat.so.1 and libz.so.1 are the only externals.
    .with_apt_packages("libexpat1", "zlib1g")
    .with_pip_packages(
        "flyte[tui]>=2.0",
        "torch==2.13.0",
        "timm==1.0.28",
        "einops==0.8.2",
        "numpy==2.5.1",
        "rasterio==1.5.0",
        "huggingface_hub==1.24.0",
        "pillow==12.3.0",
        # Required by any environment that sets a ReusePolicy. Only needed in the task
        # image, not the local venv.
        "unionai-reuse>=0.1.10",
    )
)

# Prithvi-EO-2.0-300M is a 300M-param ViT. Frozen-encoder training fits comfortably on a
# T4; full fine-tuning at 512x512 wants the L40S (48Gi).
gpu_env = flyte.TaskEnvironment(
    name="geo-burn-scar-gpu",
    image=base_image,
    # Note the exact literal: "L40s:1", lowercase s. A typed accelerator string that
    # doesn't match the enum fails at import time, not at run time.
    resources=flyte.Resources(cpu=8, memory="48Gi", gpu="L40s:1", shm="8Gi"),
)

# Tile fan-out: many small, independently-retryable tasks. This is the environment that
# makes the Flyte UI light up during a mosaic run.
#
# Sizing notes, all three learned by measuring rather than assuming:
#
#  * NO GPU. Tile inference is cheap: at 256x256 with a 16x16 patch the ViT sees only 256
#    tokens, so a 300M encoder forward measures ~0.15 s on 4 CPU threads. The whole 36-tile
#    grid is a few seconds of compute. A GPU here would spend longer waiting in the
#    scheduler queue than it saves, and CPU workers schedule instantly and cost far less.
#  * `reusable` is what actually matters. The docs put it plainly: a reusable environment
#    "preserves the Python execution environment across task executions, allowing you to
#    maintain state through global variables." That is exactly what the module-level model
#    cache in workflow.py relies on — each replica fetches the encoder and builds the model
#    once, and every later tile reuses the live process.
#    Total in-flight capacity is max_replicas * concurrency = 6 * 3 = 18.
#  * `concurrency` shares ONE Python process, so it shares memory. That is what makes the
#    cached model useful, but an unguarded cache lets every coroutine build its own 300M
#    model at once — an instant OOM. workflow.py guards it with an asyncio.Lock.
#    (concurrency > 1 requires async tasks, which these are.)
#  * Resources are billed as `replicas * resources`, so an over-generous per-replica ask is
#    multiplied by the pool. One cached model is ~1.2 GB, so 8Gi is comfortable; asking for
#    16Gi across 8 replicas would reserve 128Gi to do a few seconds of work.
#
# NOTE: container reuse requires a Union backend. Under `flyte run --local` the policy is
# simply inert — the pipeline still runs, just without a warm pool.
tile_env = flyte.TaskEnvironment(
    name="geo-burn-scar-tile",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    reusable=flyte.ReusePolicy(
        replicas=(2, 6),
        concurrency=3,
        idle_ttl=600,      # whole pool shuts down after 10 min idle
        scaledown_ttl=120,  # individual replicas linger, so a burst doesn't thrash
    ),
)

cpu_env = flyte.TaskEnvironment(
    name="geo-burn-scar-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi"),
    depends_on=[gpu_env, tile_env],
)
