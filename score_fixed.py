# score.py
# Azure ML entrypoint (init/run) that preserves the existing app contract:
# Input JSON: {"document": <json>, "num_preds": <int>}
# Output JSON: {"predictions": <document_with_predictions>}
#
# Key requirements:
# - document["documentDemandPredictions"] MUST be a JSON ARRAY STRING of demandIds (strings)
#   so the C# side can do JsonArray.Parse(predictionsNode.ToJsonString()) safely.
# - Per-chunk updates live under document["contentDomain"]["byId"][<chunkId>]
#   and include: relevantProba, cdLogregPredictions, cdTransformerPredictions
#
# This version simplifies demand loading:
# - ONLY XLSX is supported
# - File is expected in the same directory as score.py (or DEMANDS_PATH can point to it)
# - Expected columns (case-insensitive, flexible):
#     demand_id | id
#     demand
#     demand_description | description | clarification

import json
import os
import time
import logging
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals initialized in init()
demands_rows: List[Dict[str, str]] = []
demands_context: str = ""

# Azure OpenAI config
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://xxxx.openai.azure.com
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # CHAT deployment name
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Behavior knobs
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.55"))
MAX_DEMANDS_PER_CHUNK = int(os.getenv("MAX_DEMANDS_PER_CHUNK", "3"))
OPENAI_TIMEOUT_S = float(os.getenv("OPENAI_TIMEOUT_S", "60"))

def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _load_demands_xlsx() -> Tuple[List[Dict[str, str]], str]:
    """
    Loads demands from a single XLSX file.

    Priority:
      1) env DEMANDS_PATH if set (absolute or relative to score.py folder)
      2) ./demands.xlsx (next to score.py)

    Expected columns (case-insensitive):
      - demand_id OR id
      - demand
      - demand_description OR description OR clarification

    Returns:
      rows: list of dicts with keys: id, demand, description
      context: formatted string that includes IDs (critical)
    """
    demands_path = os.getenv("DEMANDS_PATH", "").strip()
    if demands_path:
        if not os.path.isabs(demands_path):
            demands_path = os.path.join(_here(), demands_path)
    else:
        demands_path = os.path.join(_here(), "demands.xlsx")

    if not os.path.exists(demands_path):
        raise FileNotFoundError(
            f"Demands XLSX not found. Expected at: {demands_path}. "
            f"Set DEMANDS_PATH or place demands.xlsx next to score.py."
        )

    df = pd.read_excel(demands_path)
    if df is None or df.empty:
        raise ValueError(f"Demands XLSX is empty: {demands_path}")

    # Normalize columns
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    # Identify column names
    id_col = "demand_id" if "demand_id" in df.columns else ("id" if "id" in df.columns else None)
    demand_col = "demand" if "demand" in df.columns else None
    desc_col = None
    for c in ["demand_description", "description", "clarification"]:
        if c in df.columns:
            desc_col = c
            break

    missing = [name for name, col in [("demand_id/id", id_col), ("demand", demand_col), ("demand_description/description/clarification", desc_col)] if col is None]
    if missing:
        raise ValueError(
            f"Demands XLSX missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    rows: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        did = str(r.get(id_col, "")).strip()
        dname = str(r.get(demand_col, "")).strip()
        ddesc = str(r.get(desc_col, "")).strip()
        if not did:
            continue
        rows.append({"id": did, "demand": dname, "description": ddesc})

    if not rows:
        raise ValueError(f"No valid demand rows found in XLSX: {demands_path}")

    # Build context with explicit IDs — this is where LLM gets demandId from.
    lines = []
    for r in rows:
        # Keep it compact but unambiguous.
        lines.append(
            f"- id: {r['id']}\n  name: {r['demand']}\n  description: {r['description']}"
        )
    context = "\n".join(lines)

    logger.info("Loaded %d demands from %s", len(rows), demands_path)
    return rows, context

def _aoai_chat_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Calls Azure OpenAI chat completions, expecting JSON response content.
    Uses a robust 'extract-first-json-object' fallback.
    """
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
        raise ValueError(
            "Missing Azure OpenAI config. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT."
        )

    url = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    payload = {
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 800,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_S)
    if r.status_code >= 400:
        raise RuntimeError(f"Azure OpenAI error {r.status_code}: {r.text}")

    data = r.json()
    content = data["choices"][0]["message"].get("content", "")
    content = content.strip()

    # Try strict JSON first
    try:
        return json.loads(content)
    except Exception:
        pass

    # Fallback: extract first JSON object/array
    m = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
    if not m:
        raise ValueError(f"Model did not return JSON. Raw content: {content[:500]}")
    extracted = m.group(1)
    return json.loads(extracted)

def _prompt_for_chunk(demands_ctx: str, chunk_id: str, chunk_text: str, num_preds: int) -> List[Dict[str, Any]]:
    """
    Returns structured per-chunk predictions:
      [
        {"id": "...", "probability": 0.85},
        ...
      ]
    and also "relevantProbability" + explanation in the JSON payload.
    """
    # System: strict behavior
    system = (
        "You are a high-precision requirements classifier. "
        "You must ONLY use demand IDs from the provided list. "
        "If the text is not a requirement/specification, mark it as not relevant and return no demands."
    )

    user = f"""
Context:
You support bid engineers working in Alfa Laval in finding specific requirements in bid documentation.

Rules:
- First decide if this chunk contains a customer REQUIREMENT / SPECIFICATION that should be highlighted.
- If not relevant: return relevantProbability < {RELEVANCE_THRESHOLD} and empty demandIds.
- If relevant: pick up to {min(num_preds, MAX_DEMANDS_PER_CHUNK)} best-matching demands.
- You MUST return ONLY demand IDs exactly as listed under 'Demands' (do not invent IDs).
- Probability scale:
  - 1.0: direct explicit match
  - 0.7-0.9: strong semantic match
  - 0.5-0.7: moderate match
  - <0.3: exclude

Demands:
{demands_ctx}

Chunk:
chunkId: {chunk_id}
text: {chunk_text}

Respond with valid JSON ONLY:
{{
  "chunkId": "{chunk_id}",
  "relevantProbability": 0.0,
  "demandIds": [{{"id":"<demand_id>","probability":0.0}}],
  "explanation": "brief"
}}
""".strip()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    out = _aoai_chat_json(messages)

    # Normalize
    rel = float(out.get("relevantProbability", 0.0) or 0.0)
    demand_ids = out.get("demandIds", []) or []
    # ensure list[dict]
    cleaned: List[Dict[str, Any]] = []
    for d in demand_ids:
        if not isinstance(d, dict):
            continue
        did = str(d.get("id", "")).strip()
        if not did:
            continue
        prob = float(d.get("probability", 0.0) or 0.0)
        cleaned.append({"id": did, "probability": prob})

    # sort by prob desc, cap
    cleaned.sort(key=lambda x: x["probability"], reverse=True)
    cleaned = cleaned[: min(num_preds, MAX_DEMANDS_PER_CHUNK)]

    return rel, cleaned, str(out.get("explanation", "") or "")

def init():
    global demands_rows, demands_context
    logger.info("Initializing model (LLM-backed) ...")
    demands_rows, demands_context = _load_demands_xlsx()
    logger.info("Initialization done.")

def inference(document: Dict[str, Any], num_cd_predictions: int) -> Dict[str, Any]:
    """
    Updates the incoming document in-place to match legacy contract.
    """
    start = time.time()

    content_domain = document.get("contentDomain", {})
    by_id = content_domain.get("byId", {})
    if not isinstance(by_id, dict) or not by_id:
        logger.warning("document.contentDomain.byId missing or empty; returning unchanged document.")
        document["documentDemandPredictions"] = "[]"
        return document

    # Keep stable iteration order by key (chunkId)
    chunk_items: List[Tuple[str, Dict[str, Any]]] = sorted(by_id.items(), key=lambda kv: kv[0])

    document_demand_ids_set = set()

    for chunk_id, content in chunk_items:
        text = str(content.get("text", "") or "")
        # default updates
        content["relevantProba"] = 0.0
        content["cdLogregPredictions"] = []
        content["cdTransformerPredictions"] = []

        if len(text.strip()) < 3:
            continue

        try:
            rel, preds, expl = _prompt_for_chunk(demands_context, chunk_id, text, num_cd_predictions)
        except Exception as e:
            logger.exception("LLM call failed for chunk %s: %s", chunk_id, str(e))
            # leave defaults, continue
            continue

        # per-chunk relevance
        content["relevantProba"] = float(rel)

        # If not relevant => keep empty predictions
        if rel < RELEVANCE_THRESHOLD or not preds:
            content["cdTransformerPredictions"] = []
            content["cdLogregPredictions"] = []
            # optional debug fields (do not break app)
            content["llmExplanation"] = expl
            continue

        # Legacy fields:
        # - cdTransformerPredictions: list of {label, proba}
        # - cdLogregPredictions: same shape (kept for compatibility; we mirror)
        cd_transformer = [{"label": p["id"], "proba": float(p["probability"])} for p in preds]
        cd_logreg = [{"label": p["id"], "proba": float(p["probability"])} for p in preds]

        content["cdTransformerPredictions"] = cd_transformer
        content["cdLogregPredictions"] = cd_logreg
        content["llmExplanation"] = expl

        # Global set (documentDemandPredictions should be JSON array string of IDs)
        for p in preds:
            document_demand_ids_set.add(p["id"])

        # ensure update is applied back
        by_id[chunk_id] = content

    # Update document contentDomain
    content_domain["byId"] = by_id
    document["contentDomain"] = content_domain

    # CRITICAL: documentDemandPredictions must be a JSON array STRING of demandIds
    # Example: '["id1","id2"]'
    document_demand_ids = sorted(list(document_demand_ids_set))
    document["documentDemandPredictions"] = json.dumps(document_demand_ids)

    logger.info("Inference done. chunks=%d global_ids=%d time=%.2fs", len(chunk_items), len(document_demand_ids), time.time() - start)
    return document

def run(raw_data):
    """
    Azure ML scoring entrypoint.
    """
    try:
        logger.info("Received request with data length=%s", len(raw_data) if isinstance(raw_data, str) else "n/a")

        if not raw_data or (isinstance(raw_data, str) and raw_data.strip() == ""):
            raise ValueError("Bad Request: Request body cannot be empty!")

        # Parse the input data
        if isinstance(raw_data, (bytes, bytearray)):
            raw_data = raw_data.decode("utf-8")

        try:
            request_data = json.loads(raw_data)
        except json.JSONDecodeError:
            raise ValueError("Bad Request: Invalid JSON format!")

        if "document" not in request_data or ("num_preds" not in request_data and "num_pred" not in request_data):
            raise ValueError("Bad Request: Invalid input, expected 'document' and 'num_preds' in request body!")

        document = request_data["document"]
        num_pred = int(request_data.get("num_preds", request_data.get("num_pred", 3)))

        response = inference(document, num_pred)

        # Preserve previous wrapper behavior:
        # app expects response JSON with either predictions root or direct document.
        # Your C# code does: json?["predictions"]?["documentDemandPredictions"] ?? json?["documentDemandPredictions"]
        # so we return {"predictions": <document>}
        return {"predictions": response}

    except ValueError as ve:
        logger.warning("Bad request: %s", str(ve))
        raise
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise Exception("Internal server error")
