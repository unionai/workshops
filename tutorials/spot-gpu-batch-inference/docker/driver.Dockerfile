# The CPU driver: Flyte's public image plus what the driver task imports.
FROM ghcr.io/flyteorg/flyte:py3.12-v2.13.1
# The image runs as user `flyte`, which cannot write to /opt/venv: install as root, then hand it back.
USER root
RUN uv pip install --python /opt/venv/bin/python --no-cache \
        python-dotenv "datasets>=3.0.0" "pandas>=2.0" "pyarrow>=15" kubernetes flyteplugins-vllm
USER flyte
