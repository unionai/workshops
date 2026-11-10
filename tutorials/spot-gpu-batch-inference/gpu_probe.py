# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
#    "kubernetes",
# ]
# main = "probe"
# params = ""
# ///

"""Check that a full MI355X node can be scheduled on the GPU queue.

Requests one whole node and reports the AMD GPUs the pod sees. If no node is
granted, the run waits in the queue instead of starting.

    uv run gpu_probe.py
"""

import glob
import os
import socket

import flyte

from common import QUEUE, full_node_pod_template, prebuild_and_reexec, with_registry

image_spec = with_registry(
    flyte.Image.from_debian_base(python_version=(3, 12)), "spot-batch-driver"
).with_pip_packages("kubernetes")

env = flyte.TaskEnvironment(
    name="spot-gpu-probe",
    image=os.getenv("DRIVER_IMAGE") or image_spec,
    resources=flyte.Resources(cpu="4", memory="16Gi"),
    pod_template=full_node_pod_template(),
    queue=QUEUE,
)


@env.task
def probe() -> dict[str, str]:
    names = []
    for path in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/name"):
        with open(path) as f:
            name = f.read().strip()
        if name:
            names.append(name)
    info = {
        "host": socket.gethostname(),
        "dev_kfd": str(os.path.exists("/dev/kfd")),
        "gpus": str(len(names)),
        "gpu_names": ",".join(sorted(set(names))),
    }
    print(info, flush=True)
    return info


if __name__ == "__main__":
    prebuild_and_reexec({"DRIVER_IMAGE": image_spec})
    flyte.init_from_config()
    run = flyte.with_runcontext(queue=QUEUE).run(probe)
    print(run.name, run.url)
