#!/usr/bin/env bash
#
# Mirror the BioNeMo NIM images from nvcr.io into the dataplane's own ECR.
#
# WHY THIS EXISTS
#
# Union Apps run on Knative. Knative's Revision reconciler resolves an image tag to a
# digest *before any pod is created*, using only the Revision's imagePullSecrets, the
# namespace's default ServiceAccount, and the cluster's `registries-skipping-tag-resolving`
# list. Union's apps controller does not set imagePullSecrets on a Revision, and
# nvcr.io is not in the skip list, so pulling a gated NVIDIA image fails at digest
# resolution with a 401 long before the pod-mutating webhook (which is what Union's
# `--type image_pull` secrets rely on) ever gets a chance to run:
#
#   Unable to fetch image "nvcr.io/nim/nvidia/genmol:1.0":
#     failed to resolve image to digest: HEAD .../manifests/1.0: 401 Unauthorized
#
# Fixing that properly means a dataplane admin change. Mirroring into the cluster's own
# ECR sidesteps it entirely: that registry is already in `registries-skipping-tag-resolving`
# and the node IAM role can pull from it natively, so no secret and no admin change.
#
# LICENSING NOTE
#
# These are NVIDIA's gated containers. This mirrors them into a *private* registry in
# the same organisation that is already entitled to them. Do not push these to a public
# registry, and check your NGC entitlement covers internal mirroring.
#
# USAGE
#
#   ECR_ACCOUNT=123456789012 AWS_PROFILE_NAME=my-profile ./scripts/mirror_nims.sh
#
# Requires: docker, awscli, an NGC_API_KEY in .env, and SSO sessions for the AWS
# profile below. Safe to re-run; it re-pushes the same tags.

set -euo pipefail

AWS_PROFILE_NAME="${AWS_PROFILE_NAME:?set AWS_PROFILE_NAME to an AWS profile with ECR push access}"
AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-2}"
ECR_ACCOUNT="${ECR_ACCOUNT:?set ECR_ACCOUNT to the AWS account ID of your dataplane ECR}"
ECR_HOST="${ECR_ACCOUNT}.dkr.ecr.${AWS_REGION_NAME}.amazonaws.com"

# source image -> destination repository (tag is carried over)
MIRRORS=(
  "nvcr.io/nim/nvidia/genmol:1.0|bionemo-nim/genmol:1.0"
  "nvcr.io/nim/mit/diffdock:2.0|bionemo-nim/diffdock:2.0"
)

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Authenticating to nvcr.io"
key=$(grep -E '^NGC_API_KEY=' "$here/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' \r\n')
if [ -z "$key" ]; then
  echo "No NGC_API_KEY in $here/.env" >&2
  exit 1
fi
# '$oauthtoken' is a literal username required by NGC, not a shell variable.
printf '%s' "$key" | docker login nvcr.io -u '$oauthtoken' --password-stdin
unset key

echo "==> Authenticating to ${ECR_HOST}"
echo "    then point the deploy at it: export BIONEMO_ECR_HOST=${ECR_HOST}"
aws ecr get-login-password --profile "$AWS_PROFILE_NAME" --region "$AWS_REGION_NAME" \
  | docker login --username AWS --password-stdin "$ECR_HOST"

for entry in "${MIRRORS[@]}"; do
  src="${entry%%|*}"
  dst_repo="${entry##*|}"
  dst="${ECR_HOST}/${dst_repo}"

  echo
  echo "==> ${src}"
  echo "    -> ${dst}"

  # Create the repository if this is the first mirror. IMMUTABLE tags so a mirrored
  # tag can never silently change under a running demo.
  repo_name="${dst_repo%%:*}"
  aws ecr describe-repositories --profile "$AWS_PROFILE_NAME" --region "$AWS_REGION_NAME" \
    --repository-names "$repo_name" >/dev/null 2>&1 || \
    aws ecr create-repository --profile "$AWS_PROFILE_NAME" --region "$AWS_REGION_NAME" \
      --repository-name "$repo_name" --image-tag-mutability IMMUTABLE >/dev/null

  docker pull "$src"
  docker tag "$src" "$dst"
  docker push "$dst"
done

echo
echo "==> Done. Point playground/config.py at:"
echo "    ${ECR_HOST}/bionemo-nim/genmol:1.0"
echo "    ${ECR_HOST}/bionemo-nim/diffdock:2.0"
