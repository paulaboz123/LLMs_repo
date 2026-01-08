import os
import json
import logging
from typing import Any

from dotenv import load_dotenv

from app.llm import make_client, call_json
from app.demands import load_demands, demands_block, demand_by_label
from app.prompts.relevance_v1 import SYSTEM_PROMPT as REL_SYSTEM, USER_TEMPLATE as REL_USER
from app.prompts.match_v1 import SYSTEM_PROMPT as MATCH_SYSTEM, USER_TEMPLATE as MATCH_USER
from app.prompts.verify_v1 import SYSTEM_PROMPT as VER_SYSTEM, USER_TEMPLATE as VER_USER

# Azure ML will import this file and call init() once, then run() per request.
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = None
deployment = None
demands_text_block = None
demands_map = None

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("TOP_P", "1.0"))

MIN_PROB = float(os.getenv("MIN_PROBABILITY", "0.60"))
REL_THRESH = float(os.getenv("RELEVANCE_THRESHOLD", "0.40"))
DUPLICATE = os.getenv("DUPLICATE_PREDICTIONS", "true").strip().lower() in ("1", "true", "yes", "y")

def init():
    global client, deployment, demands_text_block, demands_map

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    demands_path = os.getenv("DEMANDS_PATH", "./demands.xlsx")
    demands_sheet = os.getenv("DEMANDS_SHEET", "").strip() or None

    if not endpoint or not api_key or not deployment:
        raise RuntimeError("Missing Azure OpenAI env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT")

    client = make_client(endpoint=endpoint, api_key=api_key, api_version=api_version)

    demands = load_demands(demands_path, sheet=demands_sheet)
    demands_text_block = demands_block(demands)
    demands_map = demand_by_label(demands)

    logger.info("Initialized Azure OpenAI client and loaded %d demands from %s", len(demands), demands_path)

def _content_items(document: dict) -> list[dict]:
    cd = document.get("contentDomain") or {}
    by_id = cd.get("byId") or {}
    return list(by_id.values()) if isinstance(by_id, dict) else []

def _llm_relevance(text: str) -> float:
    try:
        r = call_json(client, deployment, REL_SYSTEM, REL_USER.format(text=text), TEMPERATURE, TOP_P)
    except Exception as e:
        logger.warning("Relevance LLM call failed: %s", e)
        return 0.0
    try:
        return float(r.get("relevantProba", 0.0) or 0.0) if isinstance(r, dict) else 0.0
    except Exception:
        return 0.0

def _llm_match(text: str, num_preds: int) -> list[dict]:
    try:
        m = call_json(
            client, deployment, MATCH_SYSTEM,
            MATCH_USER.format(text=text, num_preds=int(num_preds), min_prob=MIN_PROB, demands_block=demands_text_block),
            TEMPERATURE, TOP_P
        )
    except Exception as e:
        logger.warning("Match LLM call failed: %s", e)
        return []

    preds = m.get("predictions", []) if isinstance(m, dict) else []
    if not isinstance(preds, list):
        return []

    out = []
    for p in preds:
        if not isinstance(p, dict):
            continue
        label = str(p.get("label", "")).strip().lower()
        if not label or label not in demands_map:
            continue
        try:
            proba = float(p.get("proba", 0.0) or 0.0)
        except Exception:
            proba = 0.0
        if proba < MIN_PROB:
            continue
        out.append({
            "label": label,
            "proba": max(0.0, min(1.0, proba)),
            "explanation": str(p.get("explanation", "")).strip()
        })

    out.sort(key=lambda x: x["proba"], reverse=True)
    return out[:max(1, int(num_preds))]

def _llm_verify(text: str, cand: dict) -> dict | None:
    label = str(cand.get("label", "")).strip().lower()
    proba = float(cand.get("proba", 0.0) or 0.0)
    explanation = str(cand.get("explanation", "")).strip()

    d = demands_map.get(label)
    if not d:
        return None

    try:
        v = call_json(
            client, deployment, VER_SYSTEM,
            VER_USER.format(
                text=text,
                label=label,
                proba=proba,
                explanation=explanation,
                demand_label=d.label,
                item=d.item,
                demand=d.demand,
                description=d.description,
            ),
            TEMPERATURE, TOP_P
        )
    except Exception as e:
        logger.warning("Verify LLM call failed for %s: %s", label, e)
        return None

    if not isinstance(v, dict) or not v.get("accept", False):
        return None

    try:
        verify_proba = float(v.get("verify_proba", 0.0) or 0.0)
    except Exception:
        verify_proba = 0.0

    final_proba = min(max(0.0, min(1.0, proba)), max(0.0, min(1.0, verify_proba)))
    if final_proba < MIN_PROB:
        return None

    return {"label": label, "proba": round(final_proba, 3)}

def inference(document: dict, num_preds: int) -> dict:
    doc_labels = set()

    for content in _content_items(document):
        text = str(content.get("text", "") or "").strip()
        if not text:
            content.update({"relevantProba": 0.0, "cdTransformerPredictions": [], "cdLogregPredictions": []})
            continue

        # 1) relevance
        rel = max(0.0, min(1.0, _llm_relevance(text)))
        content["relevantProba"] = float(rel)

        if rel < REL_THRESH:
            content["cdTransformerPredictions"] = []
            content["cdLogregPredictions"] = []
            continue

        # 2) match + 3) verify
        verified = []
        for cand in _llm_match(text, num_preds=num_preds):
            v = _llm_verify(text, cand)
            if v:
                verified.append(v)

        verified.sort(key=lambda x: x["proba"], reverse=True)
        verified = verified[:max(1, int(num_preds))]

        preds = [{"label": p["label"], "proba": p["proba"]} for p in verified]
        content["cdTransformerPredictions"] = preds
        content["cdLogregPredictions"] = preds if DUPLICATE else []

        for p in verified:
            doc_labels.add(p["label"])

    document["documentDemandPredictions"] = list(doc_labels)
    return document

def run(raw_data: Any):
    try:
        logger.info("Received request with data: %s", str(raw_data)[:500])

        if not raw_data or (isinstance(raw_data, str) and raw_data.strip() == ""):
            raise ValueError("Bad Request: Request body cannot be empty!")

        try:
            req = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        except json.JSONDecodeError:
            raise ValueError("Bad Request: Invalid JSON format!")

        if not isinstance(req, dict) or "document" not in req or "num_preds" not in req:
            raise ValueError("Bad Request: Invalid input, expected 'document' and 'num_preds' in request body!")

        response = inference(req["document"], int(req["num_preds"]))
        logger.info("Inference completed. documentDemandPredictions=%s", response.get("documentDemandPredictions"))
        return {"predictions": response}

    except ValueError as ve:
        logger.warning("Bad request: %s", str(ve))
        raise ve
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise Exception("Internal server error")
