import flyte

# CPU-only. This pipeline reads annotations and renders geometry; there is no model
# forward pass, so a GPU would sit idle. See the README on why that is the honest framing
# rather than a limitation.
base_image = (
    flyte.Image.from_debian_base(name="av-perception-replay-v1", python_version=(3, 12))
    .with_pip_packages(
        "flyte[tui]>=2.0",
        "numpy==2.5.1",
        "pillow==12.3.0",
        "huggingface_hub==1.24.0",
        "unionai-reuse>=0.1.10",
    )
)

# Per-clip work fans out. Each clip's annotation bundle is ~9 MB, so these are small,
# fast, independently retryable tasks — the same shape as the geospatial tile fan-out.
clip_env = flyte.TaskEnvironment(
    name="av-clip",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 6), concurrency=3, idle_ttl=600,
                               scaledown_ttl=120),
)

# `depends_on` points from CALLER to CALLEE, and the direction is easy to get backwards.
# `pipeline` lives here in cpu_env and invokes `replay_clip`, which lives in clip_env — so
# cpu_env declares the dependency, not the other way round. Reversing it fails only on a
# remote run, with "Environment 'av-clip' not found in image cache"; locally there is no
# image cache and it works fine either way.
cpu_env = flyte.TaskEnvironment(
    name="av-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi"),
    depends_on=[clip_env],
)
