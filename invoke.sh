#!/usr/bin/env bash
set -euo pipefail
: "${ENDPOINT_NAME:?Set ENDPOINT_NAME}"
SCORING_URI="$(az ml online-endpoint show --name "${ENDPOINT_NAME}" --query scoring_uri -o tsv)"
KEY="$(az ml online-endpoint get-credentials --name "${ENDPOINT_NAME}" --query primaryKey -o tsv)"
curl -sS "${SCORING_URI}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${KEY}" \
  --data-binary @sample_request.json
