# score.py
# Azure ML online endpoint scoring script.
# Contract (must match existing application):
# - Input JSON: {"document": <JSON>, "num_preds": <int>, "model": <optional string>}
# - Output JSON: {"predictions": {"documentDemandPredictions": [<demand_id_str>, ...]}}
#
# The app reads json["predictions"]["documentDemandPredictions"] (or json["documentDemandPredictions"])
# and stores that array into document.predictedGlobalDemandIds.

import os
import json
import time
import logging
import re
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

try:
    from openai import AzureOpenAI
except Exception:  # pragma: no cover
    AzureOpenAI = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_client: Optional["AzureOpenAI"] = None
_chat_deployment: Optional[str] = None

_demands: List[Dict[str, str]] = []
_demands_context: str = ""

_min_prob: float = 0.30
_max_demands_per_chunk: int = 3
_max_chunks_per_call: int = 25


def _first_existing_path(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _load_demands_from_csv(path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(path)

    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("id", "demandid", "demand_id"):
            col_map["id"] = c
        elif lc in ("name", "demand", "label", "demand_name"):
            col_map["name"] = c
        elif lc in ("description", "demand_description", "clarification"):
            col_map["description"] = c

    missing = [k for k in ("id", "name", "description") if k not in col_map]
    if missing:
        raise ValueError(
            f"Demands CSV missing required columns {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Expected columns like: demand_id, demand, demand_description."
        )

    demands: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        did = str(row[col_map["id"]]).strip()
        if not did or did.lower() in ("nan", "none"):
            continue
        demands.append(
            {
                "id": did,
                "name": str(row[col_map["name"]]).strip(),
                "description": str(row[col_map["description"]]).strip(),
            }
        )
    return demands


def _load_demands_from_json(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["demands"] if isinstance(data, dict) and "demands" in data else data
    if not isinstance(items, list):
        raise ValueError(f"Unsupported demands.json shape at {path}")

    out: List[Dict[str, str]] = []
    for x in items:
        if not isinstance(x, dict):
            continue
        did = str(x.get("id") or x.get("demand_id") or x.get("demandId") or "").strip()
        if not did:
            continue
        name = str(x.get("demand") or x.get("name") or x.get("label") or "").strip()
        desc = str(x.get("demand_description") or x.get("description") or x.get("clarification") or "").strip()
        out.append({"id": did, "name": name, "description": desc})
    return out


def _build_demands_context(demands: List[Dict[str, str]]) -> str:
    # IMPORTANT: include DEMAND ID explicitly, because the model must output IDs.
    lines = []
    for d in demands:
        lines.append(
            f"- id: {d['id']}\n"
            f"  name: {d['name']}\n"
            f"  description: {d['description']}"
        )
    return "\n".join(lines)


def _system_prompt() -> str:
    return (
        "You are an expert requirements classifier for bid documentation. "
        "Match text chunks to predefined demands with high precision."
    )


def _user_prompt(demands_context: str, chunks: List[Dict[str, str]], max_demands_per_chunk: int) -> str:
    chunks_text = "\n\n".join([f"chunkId: {c['chunkId']}\ntext: {c['text']}" for c in chunks])

    return f"""**Context**
Bid engineers work with equipment specifications (motors, pumps, etc.). Demands below are CST-specific requirements.

**Rules & Constraints**
- You MUST use only the predefined demands from the list provided.
- The demand **id** is the ONLY identifier you can output. Never invent ids.
- Match ONLY when the chunk content aligns with the demand's **description** (clarification). Use name only as a hint.
- Find maximum {max_demands_per_chunk} demands per chunk.
- If no relevant demands found, return empty demandIds array for that chunk.
- Probabilities must be in [0.0, 1.0]. Use these guidelines:
  - 1.0: Direct, explicit match.
  - 0.7–0.9: Strong semantic match; clearly applies.
  - 0.5–0.7: Moderate; likely relevant.
  - 0.3–0.5: Weak; potentially applicable.
  - <0.3: Not relevant (exclude).

**Demands (predefined)**
{demands_context}

**Text Chunks to analyze**
{chunks_text}

Respond with valid JSON in this format EXACTLY:
{{
  "results": [
    {{
      "chunkId": "chunk_id",
      "demandIds": [{{"id":"<one_of_the_ids_from_demands_list>","probability":0.85}}],
      "explanation": "brief explanation"
    }}
  ]
}}
""".strip()


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        parts = text.splitlines()
        if parts and re.match(r"^[a-zA-Z]+$", parts[0].strip()):
            text = "\n".join(parts[1:]).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Model did not return JSON.")
    return json.loads(m.group(0))


def init():
    global _client, _chat_deployment, _demands, _demands_context, _min_prob, _max_demands_per_chunk, _max_chunks_per_call

    logger.info("Initializing scoring script (Azure OpenAI).")

    _min_prob = float(os.getenv("MIN_DEMAND_PROB", "0.30"))
    _max_demands_per_chunk = int(os.getenv("MAX_DEMANDS_PER_CHUNK", "3"))
    _max_chunks_per_call = int(os.getenv("MAX_CHUNKS_PER_CALL", "25"))

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    _chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    if not endpoint or not api_key or not _chat_deployment:
        raise EnvironmentError(
            "Missing Azure OpenAI configuration. Required env vars:\n"
            "- AZURE_OPENAI_ENDPOINT\n"
            "- AZURE_OPENAI_API_KEY\n"
            "- AZURE_OPENAI_DEPLOYMENT (chat deployment name)\n"
        )

    if AzureOpenAI is None:
        raise ImportError("openai package not available (AzureOpenAI). Ensure openai>=1.x is installed.")

    _client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )

    # Keep it simple: demands file next to score.py, with fallback to model dir.
    cwd = os.getcwd()
    model_dir = os.getenv("AZUREML_MODEL_DIR", "")

    candidates = [
        os.getenv("DEMANDS_PATH", ""),
        os.path.join(cwd, "demands.csv"),
        os.path.join(cwd, "demands.json"),
        os.path.join(model_dir, "albot", "demands.csv"),
        os.path.join(model_dir, "albot", "demands.json"),
        os.path.join(model_dir, "demands.csv"),
        os.path.join(model_dir, "demands.json"),
    ]
    demands_path = _first_existing_path([p for p in candidates if p])

    if not demands_path:
        raise FileNotFoundError("Could not find demands file. Put demands.csv next to score.py or set DEMANDS_PATH.")

    logger.info(f"Loading demands from: {demands_path}")
    if demands_path.lower().endswith(".csv"):
        _demands = _load_demands_from_csv(demands_path)
    elif demands_path.lower().endswith(".json"):
        _demands = _load_demands_from_json(demands_path)
    else:
        raise ValueError("Unsupported demands file type. Use .csv or .json")

    if not _demands:
        raise ValueError("Demands list is empty after loading.")

    _demands_context = _build_demands_context(_demands)
    logger.info(f"Loaded {len(_demands)} demands.")


def _call_llm_for_chunks(chunks: List[Dict[str, str]]) -> Dict[str, Any]:
    assert _client is not None and _chat_deployment is not None

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _user_prompt(_demands_context, chunks, _max_demands_per_chunk)},
    ]

    resp = _client.chat.completions.create(
        model=_chat_deployment,
        messages=messages,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1400")),
    )
    content = (resp.choices[0].message.content or "").strip()
    return _extract_json(content)


def inference(document: Dict[str, Any], num_preds: int) -> Dict[str, Any]:
    start = time.time()

    content_domain = document.get("contentDomain") or {}
    by_id = content_domain.get("byId") or {}
    if not isinstance(by_id, dict):
        raise ValueError("document.contentDomain.byId must be an object/dict")

    content_items: List[Tuple[str, Dict[str, Any]]] = []
    for cid, content in by_id.items():
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            txt = content.get("text", "").strip()
            if txt:
                content_items.append((cid, content))

    if not content_items:
        logger.info("No content items with text; returning empty predictions.")
        return {"documentDemandPredictions": []}

    chunks_all = [{"chunkId": cid, "text": content.get("text", "")} for cid, content in content_items]

    all_results_by_chunk: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(chunks_all), _max_chunks_per_call):
        batch = chunks_all[i : i + _max_chunks_per_call]
        try:
            out = _call_llm_for_chunks(batch)
        except Exception as e:
            logger.error(f"LLM call failed for batch starting at {i}: {e}", exc_info=True)
            continue

        results = out.get("results", [])
        if isinstance(results, list):
            for r in results:
                if not isinstance(r, dict):
                    continue
                chunk_id = str(r.get("chunkId") or "").strip()
                if chunk_id:
                    all_results_by_chunk[chunk_id] = r

    global_ids: List[str] = []
    global_set = set()

    for cid, content in content_items:
        r = all_results_by_chunk.get(cid)
        if not r:
            content["llmDemandPredictions"] = []
            continue

        demand_ids = r.get("demandIds", [])
        if not isinstance(demand_ids, list):
            demand_ids = []

        norm: List[Dict[str, Any]] = []
        for d in demand_ids:
            if not isinstance(d, dict):
                continue
            did = str(d.get("id") or "").strip()
            try:
                prob = float(d.get("probability"))
            except Exception:
                continue
            if not did:
                continue
            norm.append({"id": did, "probability": prob})

        norm.sort(key=lambda x: x["probability"], reverse=True)
        norm = norm[:_max_demands_per_chunk]

        content["llmDemandPredictions"] = norm
        content["llmDemandExplanation"] = str(r.get("explanation") or "")

        for d in norm:
            if d["probability"] >= _min_prob and d["id"] not in global_set:
                global_set.add(d["id"])
                global_ids.append(d["id"])

    if int(num_preds or 0) > 0:
        global_ids = global_ids[: int(num_preds)]

    latency = time.time() - start
    logger.info(f"Inference done. chunks={len(content_items)} global_ids={len(global_ids)} latency={latency:.2f}s")

    return {"documentDemandPredictions": global_ids}


def run(raw_data: str) -> Dict[str, Any]:
    try:
        if not raw_data or not str(raw_data).strip():
            raise ValueError("Bad Request: Request body cannot be empty")

        try:
            request_data = json.loads(raw_data)
        except json.JSONDecodeError:
            raise ValueError("Bad Request: Invalid JSON format")

        if "document" not in request_data or "num_preds" not in request_data:
            raise ValueError("Bad Request: expected 'document' and 'num_preds' in request body")

        document = request_data["document"]
        if not isinstance(document, dict):
            raise ValueError("Bad Request: 'document' must be a JSON object")

        num_pred = int(request_data.get("num_preds", 0))

        response = inference(document, num_pred)

        # Keep wrapper exactly as the app expects
        return {"predictions": response}

    except ValueError as ve:
        logger.warning(f"Bad request: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Internal error: {str(e)}", exc_info=True)
        raise Exception("Internal server error")
