#!/usr/bin/env bash
# Build (and optionally push) the hmr_world handler image.
#
# Usage:
#   scripts/build_image.sh              # build only, tags = latest + git SHA
#   PUSH=1 scripts/build_image.sh       # build + push to ECR
#
# Required env vars when PUSH=1:
#   AWS_REGION       (default: us-east-1)
#   AWS_ACCOUNT_ID   (auto-detected via aws sts get-caller-identity if unset)
#   ECR_REPO_NAME    (default: football-perspectives-hmr-world)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

REGION="${AWS_REGION:-us-east-1}"
REPO_NAME="${ECR_REPO_NAME:-football-perspectives-hmr-world}"
SHA="$(git rev-parse --short HEAD)"
IMAGE_LOCAL="${REPO_NAME}:${SHA}"

echo "[build_image] building ${IMAGE_LOCAL}"
docker build --platform linux/amd64 -t "${IMAGE_LOCAL}" -t "${REPO_NAME}:latest" .

if [[ "${PUSH:-0}" != "1" ]]; then
  echo "[build_image] PUSH=0 — skipping ECR push. Re-run with PUSH=1 to publish."
  exit 0
fi

if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
  AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
fi
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "[build_image] logging in to ECR ${ECR_URL}"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_URL}"

for tag in "${SHA}" latest; do
  remote="${ECR_URL}/${REPO_NAME}:${tag}"
  echo "[build_image] tagging + pushing ${remote}"
  docker tag "${IMAGE_LOCAL}" "${remote}"
  docker push "${remote}"
done

echo "[build_image] pushed. tag for terraform: ${SHA}"
echo "  terraform -chdir=infra/terraform apply -var image_tag=${SHA}"
