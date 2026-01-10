
"""
Azure ML Online Endpoint scoring script (PoC)

Contract (unchanged):
Input:  {"document": {...}, "num_preds": 3}
Output: {"predictions": <document>}

Per chunk (document["contentDomain"]["byId"][chunkId]) we write:
- relevantProba: float
- cdLogregPredictions: []
- cdTransformerPredictions: [{"label": "<demand_id>", "proba": 0.85}, ...]

Fail-soft: never raises from run(); returns empty predictions on any failure.
"""

import os, json, time, logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client: Optional[OpenAI] = None
DEMANDS: List[Dict[str, str]] = []
DEMAND_EMB: Optional[np.ndarray] = None  # L2-normalized

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

TOPK = int(os.getenv("RAG_TOPK", "8"))
MAX_LABELS_PER_CHUNK = int(os.getenv("MAX_LABELS_PER_CHUNK", "3"))
MIN_PROB = float(os.getenv("MIN_PROB", "0.30"))

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "4000"))
MAX_EXPLANATION_CHARS = int(os.getenv("MAX_EXPLANATION_CHARS", "500"))

EMBED_RETRIES = int(os.getenv("EMBED_RETRIES", "2"))
CHAT_RETRIES = int(os.getenv("CHAT_RETRIES", "2"))
RETRY_SLEEP_SEC = float(os.getenv("RETRY_SLEEP_SEC", "0.5"))

MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "1000"))

def _sleep_backoff(attempt: int) -> None:
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
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(s[a:b+1])
        except Exception:
            return None
    return None

def _empty_chunk_predictions(content: Dict[str, Any]) -> None:
    content.update({"relevantProba": 0.0, "cdLogregPredictions": [], "cdTransformerPredictions": []})

def _safe_get_byid(document: Dict[str, Any]) -> Dict[str, Any]:
    try:
        by_id = document.get("contentDomain", {}).get("byId", {})
        return by_id if isinstance(by_id, dict) else {}
    except Exception:
        return {}

def _safe_set_document_predictions(document: Dict[str, Any], preds: List[str]) -> None:
    try:
        document["documentDemandPredictions"] = list(dict.fromkeys(preds))
    except Exception:
        document["documentDemandPredictions"] = []

def _embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    if client is None or not texts:
        return None
    for attempt in range(EMBED_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            emb = np.array([d.embedding for d in resp.data], dtype=np.float32)
            return _l2_normalize(emb)
        except Exception as e:
            logger.warning("Embedding failed (%d/%d): %s", attempt+1, EMBED_RETRIES+1, str(e))
            if attempt < EMBED_RETRIES:
                _sleep_backoff(attempt)
            else:
                return None

def _chat_json(prompt: str) -> Optional[dict]:
    if client is None:
        return None
    for attempt in range(CHAT_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Return valid JSON only. No markdown. No extra text."},
                    {"role": "user", "content": prompt},
                ],
            )
            data = _safe_json_loads((resp.choices[0].message.content or "").strip())
            if data is not None:
                return data
            logger.warning("Non-JSON from model (%d/%d).", attempt+1, CHAT_RETRIES+1)
            if attempt < CHAT_RETRIES:
                prompt += "\n\nIMPORTANT: JSON only."
                _sleep_backoff(attempt)
            else:
                return None
        except Exception as e:
            logger.warning("Chat failed (%d/%d): %s", attempt+1, CHAT_RETRIES+1, str(e))
            if attempt < CHAT_RETRIES:
                _sleep_backoff(attempt)
            else:
                return None

def _topk_demands(chunk_text: str, k: int) -> List[Dict[str, str]]:
    if DEMAND_EMB is None or not DEMANDS:
        return []
    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        return []
    q = _embed_texts([chunk_text])
    if q is None:
        return []
    sims = DEMAND_EMB @ q[0]
    idx = np.argsort(-sims)[:k]
    return [DEMANDS[i] for i in idx]

def _demands_context(demands_subset: List[Dict[str, str]]) -> str:
    return "\n".join(
        f"- id: {d['id']}\n  name: {d['name']}\n  clarification: {d['clarification']}"
        for d in demands_subset
    )

def _score_one_chunk(chunk_id: str, chunk_text: str, demands_subset: List[Dict[str, str]]) -> Dict[str, Any]:
    chunk_text = (chunk_text or "").strip()
    if not chunk_text or not demands_subset:
        return {"chunkId": chunk_id, "demandIds": [], "explanation": "Empty text or no candidates."}
    if len(chunk_text) > MAX_CHUNK_CHARS:
        chunk_text = chunk_text[:MAX_CHUNK_CHARS]

    allowed = {d["id"] for d in demands_subset}
    ctx = _demands_context(demands_subset)

    prompt = f"""
Rules:
- Match ONLY when the chunk aligns with the demand clarification.
- Use only the demands listed below.
- Return max {MAX_LABELS_PER_CHUNK} demands.
- Exclude probability below {MIN_PROB}.

Demands:
{ctx}

Chunk:
chunkId: {chunk_id}
text: {chunk_text}

Return JSON only:
{{
  "chunkId": "{chunk_id}",
  "demandIds": [{{"id":"d1","probability":0.85}}],
  "explanation": "brief explanation"
}}
""".strip()

    data = _chat_json(prompt)
    if data is None:
        return {"chunkId": chunk_id, "demandIds": [], "explanation": "Model call failed or invalid JSON."}

    out = {
        "chunkId": chunk_id,
        "demandIds": [],
        "explanation": str(data.get("explanation", ""))[:MAX_EXPLANATION_CHARS],
    }

    items = data.get("demandIds", [])
    if not isinstance(items, list):
        items = []

    cleaned = []
    for it in items:
        if not isinstance(it, dict):
            continue
        did = str(it.get("id", "")).strip()
        prob = _safe_float(it.get("probability", 0.0), 0.0)
        if did in allowed and prob >= MIN_PROB:
            cleaned.append({"id": did, "probability": float(prob)})

    cleaned = sorted(cleaned, key=lambda x: x["probability"], reverse=True)[:MAX_LABELS_PER_CHUNK]
    out["demandIds"] = cleaned
    return out

def init():
    """
    Fail-soft init:
    - If key/Excel/embeddings fail, endpoint stays up and returns empty predictions.
    """
    global client, DEMANDS, DEMAND_EMB

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            client = None
            logger.error("OpenAI client init failed: %s", str(e))
    else:
        client = None
        logger.error("Missing OPENAI_API_KEY. Will return empty predictions.")

    try:
        code_dir = os.path.dirname(os.path.abspath(__file__))
        demands_path = os.getenv("DEMANDS_XLSX_PATH", os.path.join(code_dir, "demands.xlsx"))
        if not os.path.exists(demands_path):
            logger.error("Demands file not found: %s", demands_path)
            DEMANDS, DEMAND_EMB = [], None
            return

        df = pd.read_excel(demands_path).rename(columns={
            "demand_id": "id",
            "demand": "name",
            "demand_description": "clarification",
        })
        for col in ("id", "name", "clarification"):
            if col not in df.columns:
                logger.error("Missing column '%s' in demands.xlsx", col)
                DEMANDS, DEMAND_EMB = [], None
                return

        DEMANDS, texts = [], []
        for _, r in df.iterrows():
            did = str(r.get("id", "")).strip()
            name = str(r.get("name", "")).strip()
            clar = str(r.get("clarification", "")).strip()
            if not did or not name:
                continue
            DEMANDS.append({"id": did, "name": name, "clarification": clar})
            texts.append(f"{name} | {clar}")

        if not DEMANDS:
            logger.error("No demands loaded.")
            DEMAND_EMB = None
            return

        emb = _embed_texts(texts)
        if emb is None:
            logger.error("Failed to compute demand embeddings.")
            DEMAND_EMB = None
            return

        DEMAND_EMB = emb
        logger.info("Init OK. demands=%d dim=%d", len(DEMANDS), DEMAND_EMB.shape[1])

    except Exception as e:
        logger.error("Init failed: %s", str(e), exc_info=True)
        DEMANDS, DEMAND_EMB = [], None

def run(raw_data):
    """
    Fail-soft run: never raises; always returns {"predictions": document}
    """
    def _default_doc():
        return {"documentDemandPredictions": [], "contentDomain": {"byId": {}}}

    def _default_response(doc: Dict[str, Any]):
        by_id = _safe_get_byid(doc)
        for _, c in by_id.items():
            if isinstance(c, dict):
                _empty_chunk_predictions(c)
        _safe_set_document_predictions(doc, [])
        return {"predictions": doc}

    try:
        if raw_data is None or (isinstance(raw_data, str) and not raw_data.strip()):
            return {"predictions": _default_doc()}

        request = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        if not isinstance(request, dict):
            return {"predictions": _default_doc()}

        document = request.get("document", {})
        if not isinstance(document, dict):
            return {"predictions": _default_doc()}

        num_pred = int(request.get("num_preds", 3))
        by_id = _safe_get_byid(document)
        items = list(by_id.items())[:MAX_CHUNKS_PER_DOC]

        if client is None or DEMAND_EMB is None or not DEMANDS:
            for _, c in items:
                if isinstance(c, dict):
                    _empty_chunk_predictions(c)
            _safe_set_document_predictions(document, [])
            return {"predictions": document}

        doc_preds: List[str] = []

        for chunk_id, content in items:
            if not isinstance(content, dict):
                continue
            text = (content.get("text") or "").strip()
            if len(text) > MAX_CHUNK_CHARS:
                text = text[:MAX_CHUNK_CHARS]

            candidates = _topk_demands(text, TOPK)
            scored = _score_one_chunk(str(chunk_id), text, candidates)

            preds = []
            for d in scored.get("demandIds", []):
                if isinstance(d, dict):
                    preds.append({"label": str(d.get("id", "")).strip(),
                                  "proba": float(_safe_float(d.get("probability", 0.0), 0.0))})

            preds = sorted(preds, key=lambda x: x["proba"], reverse=True)[:max(0, num_pred)]
            relevant = max([p["proba"] for p in preds], default=0.0)

            content.update({"relevantProba": float(relevant),
                            "cdLogregPredictions": [],
                            "cdTransformerPredictions": preds})

            for p in preds:
                if p.get("label"):
                    doc_preds.append(p["label"])

        _safe_set_document_predictions(document, doc_preds)
        return {"predictions": document}

    except Exception as e:
        logger.error("run() failed: %s", str(e), exc_info=True)
        try:
            rd = _safe_json_loads(raw_data) if isinstance(raw_data, str) else (raw_data if isinstance(raw_data, dict) else {})
            doc = rd.get("document", {}) if isinstance(rd, dict) else {}
            if not isinstance(doc, dict):
                doc = _default_doc()
            return _default_response(doc)
        except Exception:
            return {"predictions": _default_doc()}
