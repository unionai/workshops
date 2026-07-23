import flyte

# CPU-only. Depth Anything V2 Small is 24.8M params and runs a single forward per image in
# ~0.25 s on CPU, so a GPU would spend longer queueing than computing.
base_image = (
    flyte.Image.from_debian_base(name="depth-estimation-v1", python_version=(3, 12))
    .with_pip_packages(
        "flyte[tui]>=2.0",
        "torch==2.13.0",
        "transformers==5.14.1",
        "pillow==12.3.0",
        "numpy==2.5.1",
        "h5py==3.16.0",
        "huggingface_hub==1.24.0",
        "unionai-reuse>=0.1.10",
    )
)

# Per-image estimation fans out. The model weights are ~100 MB, so a warm reusable pool
# loads them once per replica rather than once per image.
estimate_env = flyte.TaskEnvironment(
    name="depth-estimate",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 6), concurrency=3, idle_ttl=600,
                               scaledown_ttl=120),
)

# `depends_on` runs CALLER -> CALLEE. `pipeline` lives here and invokes `estimate_depth`,
# so this env declares the dependency. Reversing it works locally and fails remotely with
# "Environment not found in image cache".
cpu_env = flyte.TaskEnvironment(
    name="depth-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi"),
    depends_on=[estimate_env],
)
