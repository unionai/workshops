import flyte

# CPU-only: video decode and frame compositing. PyAV ships statically-linked FFmpeg wheels,
# so no system codec packages are needed in the image.
base_image = (
    flyte.Image.from_debian_base(name="av-scenario-coverage-v1", python_version=(3, 12))
    .with_pip_packages(
        "flyte[tui]>=2.0",
        "av==18.0.0",
        "pillow==12.3.0",
        "numpy==2.5.1",
        "torch==2.13.0",
        "transformers==5.14.1",
        # OWLv2's image processor requires scipy. Without it the failure is an opaque
        # `requires_backends` ImportError at first inference, not at import time.
        "scipy==1.18.0",
        "unionai-reuse>=0.1.10",
    )
)

# Per-scenario rendering fans out. A 7-camera scenario is ~118 MB of 4K video, so these
# want more memory and network than CPU.
scenario_env = flyte.TaskEnvironment(
    name="avcov-scenario",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 6), concurrency=2, idle_ttl=600,
                               scaledown_ttl=120),
)

# `depends_on` runs CALLER -> CALLEE. `pipeline` lives here and invokes `render_scenario`
# in scenario_env, so the dependency is declared here. Reversing it works locally and
# fails remotely with "Environment not found in image cache".
cpu_env = flyte.TaskEnvironment(
    name="avcov-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi"),
    depends_on=[scenario_env],
)
