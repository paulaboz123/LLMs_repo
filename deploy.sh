#!/usr/bin/env bash
set -euo pipefail

# One-command deployment to an EXISTING Azure ML managed online endpoint.
#
# Required env:
#   ENDPOINT_NAME, DEPLOYMENT_NAME, OPENAI_API_KEY
#
# Optional env:
#   RESOURCE_GROUP, WORKSPACE, SUBSCRIPTION (if az defaults not set)
#   INSTANCE_TYPE (default Standard_DS3_v2), INSTANCE_COUNT (default 1)
#   OPENAI_MODEL (default gpt-4o-mini), OPENAI_EMBED_MODEL (default text-embedding-3-small)
#   SET_TRAFFIC (default 100)

: "${ENDPOINT_NAME:?Set ENDPOINT_NAME}"
: "${DEPLOYMENT_NAME:?Set DEPLOYMENT_NAME}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

INSTANCE_TYPE="${INSTANCE_TYPE:-Standard_DS3_v2}"
INSTANCE_COUNT="${INSTANCE_COUNT:-1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
OPENAI_EMBED_MODEL="${OPENAI_EMBED_MODEL:-text-embedding-3-small}"
SET_TRAFFIC="${SET_TRAFFIC:-100}"

az version >/dev/null
az extension add -n ml -y >/dev/null 2>&1 || true
az extension update -n ml >/dev/null 2>&1 || true

if [[ -n "${SUBSCRIPTION:-}" ]]; then
  az account set --subscription "${SUBSCRIPTION}"
fi
if [[ -n "${RESOURCE_GROUP:-}" && -n "${WORKSPACE:-}" ]]; then
  az configure --defaults group="${RESOURCE_GROUP}" workspace="${WORKSPACE}" >/dev/null
fi

echo "Creating/Updating environment..."
az ml environment create -f deploy/environment.yml

echo "Creating deployment..."
az ml online-deployment create \
  --name "${DEPLOYMENT_NAME}" \
  --endpoint-name "${ENDPOINT_NAME}" \
  --file deploy/deployment.yml \
  --set name="${DEPLOYMENT_NAME}" \
  --set endpoint_name="${ENDPOINT_NAME}" \
  --set instance_type="${INSTANCE_TYPE}" \
  --set instance_count="${INSTANCE_COUNT}" \
  --set environment_variables.OPENAI_API_KEY="${OPENAI_API_KEY}" \
  --set environment_variables.OPENAI_MODEL="${OPENAI_MODEL}" \
  --set environment_variables.OPENAI_EMBED_MODEL="${OPENAI_EMBED_MODEL}"

echo "Routing traffic (${SET_TRAFFIC}%) to ${DEPLOYMENT_NAME}..."
az ml online-endpoint update --name "${ENDPOINT_NAME}" --traffic "${DEPLOYMENT_NAME}=${SET_TRAFFIC}"

echo "Done."
