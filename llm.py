import os
import json
import time
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Global state (init-loaded)
# -----------------------------
client: Optional[OpenAI] = None

DEMANDS: List[Dict[str, str]] = []         # [{"id","name","clarification"}]
DEMAND_EMB: Optional[np.ndarray] = None    # (n, dim), L2-normalized

OPENAI_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# -----------------------------
# Tunables (PoC defaults)
# -----------------------------
TOPK = int(os.getenv("RAG_TOPK", "8"))
MAX_LABELS_PER_CHUNK = 3
MIN_PROB = 0.30

# Token/size guards
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "4000"))
MAX_EXPLANATION_CHARS = 500

# Retries
EMBED_RETRIES = int(os.getenv("EMBED_RETRIES", "2"))
CHAT_RETRIES = int(os.getenv("CHAT_RETRIES", "2"))
RETRY_SLEEP_SEC = float(os.getenv("RETRY_SLEEP_SEC", "0.5"))

# If you need to cap per-request work
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "1000"))

# -----------------------------
# Safe helpers
# -----------------------------

def _now() -> float:
    return time.time()

def _sleep_backoff(attempt: int):
    time.sleep(RETRY_SLEEP_SEC * (2 ** attempt))

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _l2_normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12
    return m / n

def _safe_json_loads(s: str) -> Optional[dict]:
    """
    Tries to parse JSON even if model wrapped it with text.
    Strategy:
    1) direct json.loads
    2) extract first {...} block
    """
    if not s or not isinstance(s, str):
        return None

    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to extract first JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None

def _empty_chunk_predictions(content: Dict[str, Any]) -> None:
    """
    Always writes keys expected by app, so app doesn't crash.
    """
    content.update({
        "relevantProba": 0.0,
        "cdLogregPredictions": [],
        "cdTransformerPredictions": []
    })

def _safe_get_byid(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Always returns dict (possibly empty).
    """
    try:
        by_id = document.get("contentDomain", {}).get("byId", {})
        return by_id if isinstance(by_id, dict) else {}
    except Exception:
        return {}

def _safe_set_document_predictions(document: Dict[str, Any], preds: List[str]) -> None:
    """
    Make sure field exists and is JSON-serializable.
    """
    try:
        document["documentDemandPredictions"] = list(dict.fromkeys(preds))  # stable unique
    except Exception:
        document["documentDemandPredictions"] = []

# -----------------------------
# OpenAI calls (with retries)
# -----------------------------

def _embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """
    Returns normalized embeddings (n, dim) or None on failure.
    """
    if client is None:
        return None
    if not texts:
        return None

    for attempt in range(EMBED_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            emb = np.array([d.embedding for d in resp.data], dtype=np.float32)
            return _l2_normalize(emb)
        except Exception as e:
            logger.warning("Embedding failed (attempt %d/%d): %s", attempt+1, EMBED_RETRIES+1, str(e))
            if attempt < EMBED_RETRIES:
                _sleep_backoff(attempt)
            else:
                return None

def _chat_json(prompt: str) -> Optional[dict]:
    """
    Returns parsed JSON dict or None on failure.
    """
    if client is None:
        return None

    for attempt in range(CHAT_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Output must be valid JSON only. No markdown. No extra text."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            data = _safe_json_loads(content)
            if data is not None:
                return data

            logger.warning("Model returned non-JSON (attempt %d/%d).", attempt+1, CHAT_RETRIES+1)
            if attempt < CHAT_RETRIES:
                # Stronger nudge on retry
                prompt = prompt + "\n\nIMPORTANT: Return JSON only. Do not include any other text."
                _sleep_backoff(attempt)
            else:
                return None
        except Exception as e:
            logger.warning("Chat failed (attempt %d/%d): %s", attempt+1, CHAT_RETRIES+1, str(e))
            if attempt < CHAT_RETRIES:
                _sleep_backoff(attempt)
            else:
                return None

# -----------------------------
# RAG + scoring
# -----------------------------

def _topk_demands(chunk_text: str, k: int) -> List[Dict[str, str]]:
    """
    Vector retrieval. If embeddings are unavailable, return empty.
    """
    if DEMAND_EMB is None or not DEMANDS:
        return []

    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        return []

    q = _embed_texts([chunk_text])
    if q is None:
        return []

    sims = DEMAND_EMB @ q[0]  # cosine, both normalized
    idx = np.argsort(-sims)[:k]
    return [DEMANDS[i] for i in idx]

def _demands_context(demands_subset: List[Dict[str, str]]) -> str:
    lines = []
    for d in demands_subset:
        lines.append(
            f"- id: {d['id']}\n"
            f"  name: {d['name']}\n"
            f"  clarification: {d['clarification']}"
        )
    return "\n".join(lines)

def _score_one_chunk(chunk_id: str, chunk_text: str, demands_subset: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Returns normalized scoring output:
    {"chunkId":..., "demandIds":[{"id":..,"probability":..}], "explanation":...}
    Always returns valid dict (never raises).
    """
    chunk_text = (chunk_text or "").strip()
    if not chunk_text or not demands_subset:
        return {"chunkId": chunk_id, "demandIds": [], "explanation": "Empty text or no candidate demands."}

    # size guard
    if len(chunk_text) > MAX_CHUNK_CHARS:
        chunk_text = chunk_text[:MAX_CHUNK_CHARS]

    ctx = _demands_context(demands_subset)
    allowed = set(d["id"] for d in demands_subset)

    prompt = f"""
Rules & Constraints:
- Match ONLY when the chunk content aligns with the demand's clarification.
- You MUST use only the predefined demands from the list provided.
- Find maximum three demands per chunk.
- If no relevant demands found, return empty demandIds array.
- Exclude demands with probability below {MIN_PROB}.

Scoring Guidelines (0.0 to 1.0):
- 1.0: Direct, explicit match.
- 0.7-0.9: Strong semantic relationship.
- 0.5-0.7: Moderate relationship.
- 0.3-0.5: Weak relationship.
- Below {MIN_PROB}: Not relevant.

Demands:
{ctx}

Text chunk to analyze:
chunkId: {chunk_id}
text: {chunk_text}

Respond with valid JSON only:
{{
  "chunkId": "{chunk_id}",
  "demandIds": [{{"id":"d1","probability":0.85}}],
  "explanation": "brief explanation"
}}
""".strip()

    data = _chat_json(prompt)
    if data is None:
        return {"chunkId": chunk_id, "demandIds": [], "explanation": "Model call failed or invalid JSON."}

    # sanitize output
    out = {
        "chunkId": chunk_id,
        "demandIds": [],
        "explanation": str(data.get("explanation", ""))[:MAX_EXPLANATION_CHARS],
    }

    demand_ids = data.get("demandIds", [])
    if not isinstance(demand_ids, list):
        demand_ids = []

    cleaned = []
    for item in demand_ids:
        if not isinstance(item, dict):
            continue
        did = str(item.get("id", "")).strip()
        prob = _safe_float(item.get("probability", 0.0), 0.0)
        if did in allowed and prob >= MIN_PROB:
            cleaned.append({"id": did, "probability": float(prob)})

    cleaned = sorted(cleaned, key=lambda x: x["probability"], reverse=True)[:MAX_LABELS_PER_CHUNK]
    out["demandIds"] = cleaned
    return out

# -----------------------------
# Azure ML entrypoints
# -----------------------------

def init():
    """
    Fail-soft init:
    - If anything fails, we keep endpoint alive and return empty predictions at runtime.
    """
    global client, DEMANDS, DEMAND_EMB, OPENAI_MODEL, EMBED_MODEL

    OPENAI_MODEL = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
    EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", EMBED_MODEL)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            client = None
            logger.error("OpenAI client init failed: %s", str(e))
    else:
        client = None
        logger.error("Missing OPENAI_API_KEY. Endpoint will return empty predictions.")

    # Load Excel (fail-soft)
    try:
        base_dir = os.getenv("AZUREML_MODEL_DIR", os.getcwd())
        demands_path = os.getenv("DEMANDS_XLSX_PATH", os.path.join(base_dir, "demands.xlsx"))

        if not os.path.exists(demands_path):
            logger.error("Demands file not found: %s. Endpoint will return empty predictions.", demands_path)
            DEMANDS = []
            DEMAND_EMB = None
            return

        df = pd.read_excel(demands_path)

        # Your known column names (as you said: already known from your files)
        df = df.rename(columns={
            "demand_id": "id",
            "demand": "name",
            "demand_description": "clarification",
        })

        # Validate columns (fail-soft)
        for col in ("id", "name", "clarification"):
            if col not in df.columns:
                logger.error("Missing column '%s' in demands.xlsx. Endpoint will return empty predictions.", col)
                DEMANDS = []
                DEMAND_EMB = None
                return

        DEMANDS = []
        demand_texts = []
        for _, r in df.iterrows():
            did = str(r.get("id", "")).strip()
            name = str(r.get("name", "")).strip()
            clar = str(r.get("clarification", "")).strip()

            if not did or not name:
                continue

            DEMANDS.append({"id": did, "name": name, "clarification": clar})
            demand_texts.append(f"{name} | {clar}")

        if not DEMANDS:
            logger.error("No demands loaded from demands.xlsx. Endpoint will return empty predictions.")
            DEMAND_EMB = None
            return

        # Precompute embeddings (fail-soft)
        emb = _embed_texts(demand_texts)
        if emb is None:
            logger.error("Failed to precompute demand embeddings. Endpoint will return empty predictions.")
            DEMAND_EMB = None
            return

        DEMAND_EMB = emb
        logger.info("Init OK. demands=%d, emb_dim=%d", len(DEMANDS), DEMAND_EMB.shape[1])

    except Exception as e:
        logger.error("Init failed (fail-soft): %s", str(e), exc_info=True)
        DEMANDS = []
        DEMAND_EMB = None


def run(raw_data):
    """
    Fail-soft run:
    Always returns {"predictions": document} with required keys set.
    Never raises.
    """
    t0 = _now()

    # Default response if anything goes wrong
    def _default_response(doc: Dict[str, Any]) -> Dict[str, Any]:
        # ensure all chunks have required keys
        by_id = _safe_get_byid(doc)
        for _, content in by_id.items():
            if isinstance(content, dict):
                _empty_chunk_predictions(content)
        _safe_set_document_predictions(doc, [])
        return {"predictions": doc}

    try:
        # Parse input
        if raw_data is None or (isinstance(raw_data, str) and not raw_data.strip()):
            logger.warning("Empty request body.")
            return {"predictions": {"documentDemandPredictions": [], "contentDomain": {"byId": {}}}}

        request_data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        if not isinstance(request_data, dict):
            logger.warning("Request is not a dict.")
            return {"predictions": {"documentDemandPredictions": [], "contentDomain": {"byId": {}}}}

        document = request_data.get("document", {})
        if not isinstance(document, dict):
            logger.warning("'document' missing or invalid.")
            return {"predictions": {"documentDemandPredictions": [], "contentDomain": {"byId": {}}}}

        num_pred = int(request_data.get("num_preds", 3))

        by_id = _safe_get_byid(document)

        # hard cap chunks per doc
        items = list(by_id.items())[:MAX_CHUNKS_PER_DOC]

        document_demand_predictions: List[str] = []

        # If init failed -> return empty but consistent output
        if client is None or DEMAND_EMB is None or not DEMANDS:
            logger.warning("Model not ready (client/embeddings/demands missing). Returning empty predictions.")
            for _, content in items:
                if isinstance(content, dict):
                    _empty_chunk_predictions(content)
            _safe_set_document_predictions(document, [])
            return {"predictions": document}

        for chunk_id, content in items:
            if not isinstance(content, dict):
                continue

            text = (content.get("text") or "").strip()
            if len(text) > MAX_CHUNK_CHARS:
                text = text[:MAX_CHUNK_CHARS]

            # RAG retrieve
            candidates = _topk_demands(text, k=TOPK)

            # LLM score
            scored = _score_one_chunk(str(chunk_id), text, candidates)
            demand_ids = scored.get("demandIds", [])

            # Map to existing format
            cd_transformer_predictions = []
            for d in demand_ids:
                if isinstance(d, dict):
                    cd_transformer_predictions.append({
                        "label": str(d.get("id", "")).strip(),
                        "proba": float(_safe_float(d.get("probability", 0.0), 0.0))
                    })

            cd_transformer_predictions = sorted(cd_transformer_predictions, key=lambda x: x["proba"], reverse=True)[:max(0, num_pred)]
            relevant_proba = max([p["proba"] for p in cd_transformer_predictions], default=0.0)

            content.update({
                "relevantProba": float(relevant_proba),
                "cdLogregPredictions": [],  # keep contract stable
                "cdTransformerPredictions": cd_transformer_predictions
            })

            for p in cd_transformer_predictions:
                if p.get("label"):
                    document_demand_predictions.append(p["label"])

        _safe_set_document_predictions(document, document_demand_predictions)

        logger.info("run() done in %.3fs chunks=%d", _now() - t0, len(items))
        return {"predictions": document}

    except Exception as e:
        # Absolute last-resort fallback: never crash endpoint
        logger.error("run() failed (fail-soft): %s", str(e), exc_info=True)
        try:
            # attempt to recover document if present
            if isinstance(raw_data, str):
                rd = _safe_json_loads(raw_data) or {}
            else:
                rd = raw_data if isinstance(raw_data, dict) else {}
            doc = rd.get("document", {}) if isinstance(rd, dict) else {}
            if not isinstance(doc, dict):
                doc = {}
            return _default_response(doc)
        except Exception:
            return {"predictions": {"documentDemandPredictions": [], "contentDomain": {"byId": {}}}}
