"""Build the worker and driver images once (no queue, so the default pool's builder) and print the URIs to pin.

    GPU=MI355Xx8 python build.py    # the ROCm worker, built by the default pool's builder
"""
import flyte

import config

if __name__ == "__main__":
    flyte.init_from_config()
    print("WORKER_IMAGE=" + flyte.build(config.worker_image).uri)
    print("DRIVER_IMAGE=" + flyte.build(config.driver_env.image).uri)
