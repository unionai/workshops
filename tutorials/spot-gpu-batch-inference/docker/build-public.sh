#!/usr/bin/env bash
# Build the worker and driver images on this machine and push them to a public registry the GPU pool can pull from.
#   docker login -u <user>       # once
#   REGISTRY=docker.io/<user> ./docker/build-public.sh
# Then pin the printed URIs: WORKER_IMAGE=... DRIVER_IMAGE=...   (ONLY=worker or ONLY=driver builds one)
set -euo pipefail
cd "$(dirname "$0")"
REGISTRY="${REGISTRY:?set REGISTRY, e.g. docker.io/<your user>}"
WTAG="$(shasum < worker-rocm.Dockerfile | cut -c1-12)"
DTAG="$(shasum < driver.Dockerfile | cut -c1-12)"
[ "${ONLY:-}" = driver ] || docker buildx build --platform linux/amd64 -f worker-rocm.Dockerfile -t "$REGISTRY/batch-llm-vllm-rocm:$WTAG" --push .
[ "${ONLY:-}" = worker ] || docker buildx build --platform linux/amd64 -f driver.Dockerfile -t "$REGISTRY/batch-llm-driver:$DTAG" --push .
echo "WORKER_IMAGE=$REGISTRY/batch-llm-vllm-rocm:$WTAG"
echo "DRIVER_IMAGE=$REGISTRY/batch-llm-driver:$DTAG"
