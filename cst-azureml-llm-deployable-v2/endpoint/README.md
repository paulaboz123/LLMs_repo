# CST Demand Matching — AzureML `score.py` (LLM replacement for relevance + labelers)

## What this is
A **drop-in replacement** for your existing Azure ML online endpoint that currently uses:
- relevance model (highlight gating)
- logreg label classifier
- transformer label classifier

We replace them with **Azure OpenAI** using a simple, stable pipeline:

1) **Relevance** (`CST_RELEVANCE_V1`)  
   Produces `relevantProba` (0..1). If below `RELEVANCE_THRESHOLD`, we skip label calls (cost saver).
2) **Match** (`CST_MATCH_V1`)  
   Proposes up to `num_preds` demand labels from the closed catalog with probabilities.
3) **Verify** (`CST_VERIFY_V1`)  
   Confirms each proposed label and calibrates probability; rejects false positives.

## Compatibility (important)
We keep the **same request/response contract** as your current `score.py` pattern:

- Request JSON body contains:
  - `document` (already extracted and chunked by your app)
  - `num_preds` (how many labels per chunk to return)
- Response is:
  - `{"predictions": <updated document>}`

Per content (each paragraph/chunk in `document["contentDomain"]["byId"]`) we set:
- `relevantProba` (float)
- `cdTransformerPredictions` (list of `{label, proba}`)
- `cdLogregPredictions` (duplicated from transformer or empty; configurable)
And at document level:
- `document["documentDemandPredictions"]` (unique list of predicted labels)

## Demand catalog (Excel or CSV)
Excel columns (case-insensitive):
- `item` (pump or motor)
- `demand` (label name)
- `demand description`

Label id format:
- `label = item|demand` (lowercased)

Env vars:
- `DEMANDS_PATH` (default `./demands.xlsx`)
- `DEMANDS_SHEET` (optional for xlsx)

## Azure OpenAI env vars
Set:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (default: `2024-10-21`)
- `AZURE_OPENAI_CHAT_DEPLOYMENT`

## Tuning knobs
- `RELEVANCE_THRESHOLD` (default 0.40)
- `MIN_PROBABILITY` (default 0.60)
- `num_preds` (request)
- `TEMPERATURE` (default 0.0)

Files:
- `score.py`
- `app/prompts/relevance_v1.py`
- `app/prompts/match_v1.py`
- `app/prompts/verify_v1.py`
- `app/demands.py`
- `app/llm.py`
