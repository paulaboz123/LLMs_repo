# CST Demand Matching — One-command Azure ML SDK v2 deployment

This package includes:
- `endpoint/` — scoring code (`score.py` + `app/`) that replaces relevance + label models with Azure OpenAI
- `infra/` — Azure ML SDK v2 deployment scripts

## 1) Put your demand catalog (Excel)
Copy your Excel file to:
- `endpoint/demands.xlsx`

Required columns (case-insensitive):
- `item` (pump/motor)
- `demand`
- `demand description`

## 2) Export required environment variables
Workspace:
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_RESOURCE_GROUP`
- `AZUREML_WORKSPACE_NAME`

Azure OpenAI:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
Optional:
- `AZURE_OPENAI_API_VERSION` (default `2024-10-21`)

Optional deployment settings:
- `ENDPOINT_NAME` (default `cst-demand-llm-endpoint`)
- `DEPLOYMENT_NAME` (default `blue`)
- `INSTANCE_TYPE` (default `Standard_DS3_v2`)
- `INSTANCE_COUNT` (default `1`)

## 3) Deploy in one command
Linux/macOS:
```bash
bash infra/deploy.sh
```

Windows PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\infra\deploy.ps1
```

## 4) Call from the application
Your app keeps the same contract used by the previous `score.py`:

Request JSON body:
- `document`
- `num_preds`

Response:
- `{"predictions": <updated document>}`

## Inference environment packages (exact versions)
Azure ML inference environment is defined in:
- `infra/conda.yaml`

Local deployment dependencies are pinned in:
- `infra/requirements.deploy.txt`

Notes:
- The deployment uses a placeholder Model asset (required by managed online endpoints).
- Secrets should ideally be injected via Key Vault/managed identity later; env vars are fine for PoC.
