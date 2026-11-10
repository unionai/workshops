"""Does a task land on a whole 8x MI355X node? The GPU request lives in the pod template, as whole nodes require.

    QUEUE=<your GPU queue> FLYTE_CONFIG=<your cluster config> python checks/amd_gpu_hello.py
"""

import glob
import os

import flyte

try:
    # Only needed to build the pod template at submit time; the task image lacks it.
    from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

    env = flyte.TaskEnvironment(
        name="amd_gpu_hello",
        resources=flyte.Resources(cpu="8", memory="32Gi"),
        pod_template=flyte.PodTemplate(
            primary_container_name="primary",
            pod_spec=V1PodSpec(
                containers=[
                    V1Container(
                        name="primary",
                        resources=V1ResourceRequirements(requests={"amd.com/gpu": "8"}, limits={"amd.com/gpu": "8"}),
                    )
                ],
            ),
        ),
    )
except ImportError:
    env = flyte.TaskEnvironment(name="amd_gpu_hello")


@env.task
def main() -> dict[str, str]:
    nodes = glob.glob("/sys/class/kfd/kfd/topology/nodes/*/name")
    gpus = [open(p).read().strip() for p in nodes if open(p).read().strip()]
    vram = [str(round(int(open(f).read()) / 2**30)) for f in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"))]
    info = {
        "dev_kfd": str(os.path.exists("/dev/kfd")),
        "dri": ",".join(sorted(os.listdir("/dev/dri"))) if os.path.isdir("/dev/dri") else "",
        "kfd_gpus": ",".join(gpus),
        "vram_gib_per_card": ",".join(vram),
        "cpus": str(os.cpu_count()),
    }
    print(info, flush=True)
    if not os.path.exists("/dev/kfd"):
        raise RuntimeError(f"no AMD GPU attached: {info}")
    return info


if __name__ == "__main__":
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))  # defaults to the usual config lookup
    run = flyte.with_runcontext(queue=os.environ["QUEUE"]).run(main)
    print(run.name, run.url)
