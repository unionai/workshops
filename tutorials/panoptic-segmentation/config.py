import flyte

# CPU-only. Mask2Former Swin-Base runs a single forward per image in ~0.3 s on CPU, so a
# GPU would queue longer than it computes.
base_image = (
    flyte.Image.from_debian_base(name="panoptic-segmentation-v1", python_version=(3, 12))
    .with_pip_packages(
        "flyte[tui]>=2.0",
        "torch==2.13.0",
        "transformers==5.14.1",
        # Mask2Former post-processing requires scipy; without it the failure is an opaque
        # requires_backends ImportError at first inference, not at import.
        "scipy==1.16.2",
        "pillow==12.3.0",
        "numpy==2.5.1",
        "pyarrow==21.0.0",
        "huggingface_hub==1.24.0",
        "unionai-reuse>=0.1.10",
    )
)

# Per-image segmentation fans out. The weights are ~430 MB, so a warm reusable pool loads
# them once per replica rather than once per image.
segment_env = flyte.TaskEnvironment(
    name="panoptic-segment",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 6), concurrency=3, idle_ttl=600,
                               scaledown_ttl=120),
)

# `depends_on` runs CALLER -> CALLEE. `pipeline` lives here and invokes `segment_image`, so
# this env declares the dependency. Reversing it works locally, fails remotely.
cpu_env = flyte.TaskEnvironment(
    name="panoptic-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi"),
    depends_on=[segment_env],
)
