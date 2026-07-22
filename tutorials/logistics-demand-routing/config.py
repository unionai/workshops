import flyte

# NOTE: `.with_pip_packages()` rather than `.with_requirements()` — the latter stores a
# *relative* path and re-resolves it at runtime, which breaks anywhere the working
# directory isn't yours. Keep requirements.txt for the local venv and keep them in sync.
#
# Every stage of this pipeline is CPU-only. Chronos-Bolt is a 205M-parameter encoder and
# forecasts a batch of zones in well under a second on CPU; OR-Tools is a CPU solver. No
# GPU is requested anywhere, which is a large part of why this one is cheap to run.
base_image = (
    flyte.Image.from_debian_base(name="logistics-demand-routing-v1", python_version=(3, 12))
    .with_pip_packages(
        "flyte[tui]>=2.0",
        "chronos-forecasting==2.3.1",
        "ortools==9.15.6755",
        "torch==2.13.0",
        "pyarrow==25.0.0",
        "numpy==2.5.1",
        "huggingface_hub==1.24.0",
        # Required by any environment that sets a ReusePolicy.
        "unionai-reuse>=0.1.10",
    )
)

# Forecasting fans out across zone batches. The Chronos weights are ~800 MB, so a warm
# pool means they deserialize once per replica rather than once per batch.
forecast_env = flyte.TaskEnvironment(
    name="logistics-forecast",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 6), concurrency=4, idle_ttl=300),
)

# The VRP solver is single-threaded per solve and benefits from headroom, not parallelism.
solver_env = flyte.TaskEnvironment(
    name="logistics-solver",
    image=base_image,
    resources=flyte.Resources(cpu=8, memory="16Gi"),
)

# `depends_on` runs CALLER -> CALLEE. `pipeline` lives in cpu_env and invokes tasks in
# BOTH other environments, so cpu_env declares them. Getting this backwards works locally
# and fails remotely with "Environment not found in image cache".
cpu_env = flyte.TaskEnvironment(
    name="logistics-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi"),
    depends_on=[forecast_env, solver_env],
)
