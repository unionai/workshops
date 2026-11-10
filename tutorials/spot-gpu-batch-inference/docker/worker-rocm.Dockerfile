# The warm-pool worker for AMD GPUs: the day-0 ROCm vLLM image, plus the Flyte runtime and Union's reuse bridge,
# the vLLM plugin (weight streaming from object storage), the Kubernetes client (pod templates are built at import)
# and the OpenAI client (for workers that run a vLLM server and call it).
# Built and pushed by hand (see build-public.sh) for pools that have no image registry of their own.
FROM vllm/vllm-openai-rocm:nightly-rocm100

# vLLM lives in the image's system Python 3.12, which is marked externally managed (PEP 668).
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "flyte==2.13.1" unionai-reuse async-lru python-dotenv \
        "flyteplugins-vllm>=2.13.1" kubernetes openai httpx

# The stock image starts `vllm serve`. Union starts a reusable worker by passing its launcher as the container args,
# so that entrypoint would read the launcher as a model name and exit. Clear it, as Flyte's own images have none.
ENTRYPOINT []
CMD []
WORKDIR /root
ENV PYTHONPATH=/root
