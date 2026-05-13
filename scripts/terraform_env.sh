#!/usr/bin/env bash
# Export Terraform outputs as shell env vars so config/default.yaml's
# ${HMR_WORLD_*} references resolve correctly.
#
# Usage:
#   source scripts/terraform_env.sh

set -euo pipefail

if ! command -v terraform >/dev/null 2>&1; then
  echo "terraform not found in PATH" >&2
  return 1 2>/dev/null || exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found in PATH" >&2
  return 1 2>/dev/null || exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
TF_DIR="${REPO_ROOT}/infra/terraform"
OUTPUTS_JSON="$(terraform -chdir="${TF_DIR}" output -json)"

export HMR_WORLD_S3_BUCKET="$(echo "${OUTPUTS_JSON}" | jq -r '.s3_bucket.value')"
export HMR_WORLD_BATCH_JOB_QUEUE="$(echo "${OUTPUTS_JSON}" | jq -r '.batch_job_queue.value')"
export HMR_WORLD_BATCH_JOB_DEFINITION="$(echo "${OUTPUTS_JSON}" | jq -r '.batch_job_definition.value')"
export AWS_REGION="$(echo "${OUTPUTS_JSON}" | jq -r '.region.value')"

echo "[terraform_env] HMR_WORLD_S3_BUCKET=${HMR_WORLD_S3_BUCKET}"
echo "[terraform_env] HMR_WORLD_BATCH_JOB_QUEUE=${HMR_WORLD_BATCH_JOB_QUEUE}"
echo "[terraform_env] HMR_WORLD_BATCH_JOB_DEFINITION=${HMR_WORLD_BATCH_JOB_DEFINITION}"
echo "[terraform_env] AWS_REGION=${AWS_REGION}"
