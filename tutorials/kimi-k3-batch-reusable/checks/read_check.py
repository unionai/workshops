"""Can a node on the GPU pool's queue read a prefetched model directory, and how fast?

    QUEUE=<your GPU queue> FLYTE_CONFIG=<your cluster config> [CPU_ONLY=1] python checks/read_check.py <prefetched dir uri>
"""

import os
import sys
import time

import flyte
import flyte.io

CPU_ONLY = os.getenv("CPU_ONLY") == "1"  # read the store from a CPU pod: same pool, no whole-node grant

try:
    if CPU_ONLY:
        raise ImportError
    from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

    env = flyte.TaskEnvironment(
        name="nc_read_check",
        resources=flyte.Resources(cpu="8", memory="32Gi", disk="100Gi"),
        pod_template=flyte.PodTemplate(
            primary_container_name="primary",
            pod_spec=V1PodSpec(containers=[V1Container(name="primary", resources=V1ResourceRequirements(
                requests={"amd.com/gpu": "8"}, limits={"amd.com/gpu": "8"}))]),
        ),
    )
except ImportError:
    env = flyte.TaskEnvironment(name="nc_read_check_cpu" if CPU_ONLY else "nc_read_check",
                                resources=flyte.Resources(cpu="2", memory="8Gi", disk="20Gi"))


@env.task
async def read_check(uri: str) -> dict[str, str]:
    info = {"uri": uri, "dev_kfd": str(os.path.exists("/dev/kfd"))}
    t0 = time.time()
    try:
        local = await flyte.io.Dir.from_existing_remote(uri).download()
        secs = time.time() - t0
        total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(local) for f in fs)
        info.update({"ok": "true", "bytes": str(total), "seconds": f"{secs:.1f}", "MB_per_s": f"{total / 1e6 / max(secs, 1e-3):.0f}",
                     "files": ",".join(sorted(os.listdir(local)))[:500]})
    except Exception as e:  # noqa: BLE001
        info.update({"ok": "false", "seconds": f"{time.time() - t0:.1f}", "error": f"{type(e).__name__}: {e}"[:1500]})
    print("READ_CHECK", info, flush=True)
    # Outputs and logs on this queue are not readable from the laptop, but an error message is: report through it.
    raise RuntimeError(("READ_OK " if info.get("ok") == "true" else "READ_FAILED ") + str(info))


if __name__ == "__main__":
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))  # defaults to the usual config lookup
    run = flyte.with_runcontext(queue=os.environ["QUEUE"]).run(read_check, uri=sys.argv[1])
    print(run.name, run.url)
