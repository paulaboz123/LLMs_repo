# OpenAI RAG Demand Labeler – Azure ML Online Endpoint (PoC)

This repo deploys a **drop-in replacement scoring script** to an **existing** Azure ML *Managed Online Endpoint*.
It preserves your application contract and only replaces transformer inference with **OpenAI + simple RAG**.

## What you must replace
### 1) Replace `demands.xlsx`
This repo includes a sample `demands.xlsx`. Replace it with your real file.

Required columns (exact):
- `demand_id`
- `demand`
- `demand_description`

If your real file uses different names, change the `rename(...)` mapping in `score.py` → `init()`.

### 2) Provide OpenAI API key
Deployment requires `OPENAI_API_KEY`.

Optional:
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_EMBED_MODEL` (default: `text-embedding-3-small`)

## One-line deployment to an EXISTING endpoint

Pre-req:
- `az login`
- (recommended) set defaults once:
  `az configure --defaults group="<RG>" workspace="<WS>"`

From repo root:
```bash
ENDPOINT_NAME="<existing-endpoint>" \
DEPLOYMENT_NAME="openai-rag-v1" \
OPENAI_API_KEY="<your-key>" \
./deploy/deploy.sh
```

If you do NOT have az defaults configured:
```bash
RESOURCE_GROUP="<rg>" WORKSPACE="<ws>" SUBSCRIPTION="<sub-id>" \
ENDPOINT_NAME="..." DEPLOYMENT_NAME="openai-rag-v1" OPENAI_API_KEY="..." \
./deploy/deploy.sh
```

Optional compute:
```bash
INSTANCE_TYPE="Standard_DS3_v2" INSTANCE_COUNT="1" \
ENDPOINT_NAME="..." DEPLOYMENT_NAME="openai-rag-v1" OPENAI_API_KEY="..." \
./deploy/deploy.sh
```

What the script does:
1) Creates AML environment `openai-rag-demand-labeler-env` (from `deploy/conda.yaml`)
2) Creates a deployment on your existing endpoint using `score.py`
3) Routes traffic to the new deployment (default 100%)

## Test the endpoint
```bash
ENDPOINT_NAME="<existing-endpoint>" ./deploy/invoke.sh
```

## Output contract (unchanged)
Input:
```json
{"document": {"contentDomain": {"byId": {"chunk_1": {"text":"..."}}}}, "num_preds": 3}
```

Output:
```json
{"predictions": <document>}
```

Each chunk is updated with:
- `relevantProba`
- `cdLogregPredictions` (empty list, kept for backward compatibility)
- `cdTransformerPredictions` = [{"label":"<demand_id>","proba":0.85}, ...]

Plus document-level:
- `documentDemandPredictions` (unique set across chunks)

## Notes
- This PoC is **fail-soft**: if OpenAI is unavailable/misconfigured, it returns empty predictions rather than crashing.
- For production: batch multiple chunks per LLM call to reduce latency/cost and rate-limit risk.

