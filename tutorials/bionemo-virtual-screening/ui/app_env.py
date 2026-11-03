"""
The Vue UI, served as a Union app.

This module lives inside ui/ on purpose: ``include`` paths are resolved relative to
the file that defines the environment, so anchoring here keeps them short and means
the bundle contains the UI files at the paths ``start.sh`` expects.

``VITE_API_BASE`` is filled in by deploy.py once the API app has an endpoint — see
``set_api_base`` below. The build runs at container start rather than image build
precisely so that value can be baked into the bundle.
"""

from __future__ import annotations

import flyte
import flyte.app

# Node for the Vue build, and Flyte must stay installed even though no Python of ours
# runs here: with `args` (rather than `command`) the container is launched through
# Flyte's own `fserve` entrypoint, which wraps `sh start.sh`. Building this with
# install_flyte=False removes that binary and the pod dies before logging anything,
# as StartError exit 128 with an empty log.
ui_image = flyte.Image.from_debian_base(name="gvs-ui").with_apt_packages("nodejs", "npm")

ui_env = flyte.app.AppEnvironment(
    name="bionemo-ui",
    description="BioNeMo playground: generative virtual screening on Union",
    image=ui_image,
    # Path is relative to the code bundle root, which is the project root (deploy.py
    # lives there), not this directory. start.sh cd's to its own location.
    args="sh ui/start.sh",
    port=8080,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    include=(
        "package.json",
        "vite.config.js",
        "index.html",
        "start.sh",
        "src",
    ),
    # Authenticated. Visitors reach the demo on a machine you have already signed in
    # on, rather than the link working for anyone who receives it.
    requires_auth=True,
    scaling=flyte.app.Scaling(replicas=1),
)


def set_api_base(endpoint: str) -> flyte.app.AppEnvironment:
    """
    Point the UI at the deployed API.

    Called by deploy.py after the API app is up, because the endpoint isn't knowable
    until then.
    """
    ui_env.env_vars = {**(ui_env.env_vars or {}), "VITE_API_BASE": endpoint.rstrip("/")}
    return ui_env
