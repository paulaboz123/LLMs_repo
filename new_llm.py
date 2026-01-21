import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("score")

# -------------------------
# Tunables (env)
# -------------------------
USE_MOCK = os.getenv("USE_MOCK", "0").strip() == "1"

MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "10"))

MAX_LABELS_PER_CHUNK = int(os.getenv("MAX_LABELS_PER_CHUNK", "3"))
CANDIDATES_TOPK = int(os.getenv("CANDIDATES_TOPK", "10"))

LABEL_MIN_PROB = float(os.getenv("LABEL_MIN_PROB", "0.75"))
HIGHLIGHT_MIN_PROB = float(os.getenv("HIGHLIGHT_MIN_PROB", "0.80"))
RELEVANCE_MIN = float(os.getenv("RELEVANCE_MIN", "0.60"))

# Sentence/line processing
MAX_LINES_PER_CHUNK = int(os.getenv("MAX_LINES_PER_CHUNK", "18"))
MAX_LINES_TO_LLM = int(os.getenv("MAX_LINES_TO_LLM", "6"))

# Hard filters to prevent nonsense labels
OVERLAP_MIN = int(os.getenv("OVERLAP_MIN", "1"))  # now used only as "backup", not primary
CRITERIA_HITS_MIN = int(os.getenv("CRITERIA_HITS_MIN", "1"))

# Embedding retrieval thresholds
EMB_SIM_MIN = float(os.getenv("EMB_SIM_MIN", "0.35"))          # below this: do not include as candidate
EMB_SIM_STRONG = float(os.getenv("EMB_SIM_STRONG", "0.45"))    # above this: allow even if overlap is low
EMB_BATCH = int(os.getenv("EMB_BATCH", "64"))

MAX_DEBUG_CHUNKS = int(os.getenv("MAX_DEBUG_CHUNKS", "0"))

client = None
CHAT_DEPLOYMENT = None
EMB_DEPLOYMENT = None

DEMANDS: List[Dict[str, str]] = []               # [{id, demand, description}]
DEMAND_TEXT: Dict[str, str] = {}                 # id -> "name | description_short"
DEMAND_TOKENSETS: Dict[str, set] = {}            # id -> tokenset (for fallback / sanity)
DEMAND_EMB: Dict[str, List[float]] = {}          # id -> embedding vector

# Light heuristics to prioritize lines
REQ_MODAL_RE = re.compile(
    r"\b(shall|must|required|requirement|should|need to|has to|may not|must not|shall not|prohibit|forbidden)\b",
    re.IGNORECASE,
)
REQ_PL_MODAL_RE = re.compile(
    r"\b(musi|należy|wymaga|wymagane|powinien|powinna|zakazuje|zabrania|nie wolno|dopuszcza się|wymóg)\b",
    re.IGNORECASE,
)
REQ_NUM_RE = re.compile(r"\b\d+([.,]\d+)?\b")
UNIT_RE = re.compile(r"\b(mm|cm|m|kg|w|kw|mw|v|kv|a|ma|hz|rpm|bar|pa|°c|c|ip\d{2})\b", re.IGNORECASE)

# Optional: very small stopword list to reduce overlap noise (English + PL)
STOPWORDS = {
    "shall","must","should","required","requirement","provide","provided","system","equipment","contractor","vendor",
    "the","and","for","with","this","that","are","is","to","of","in","on","by","as","be","will",
    "należy","musi","powinien","powinna","wymaga","wymagane","zapewnić","systemu","urządzenie","urządzenia",
    "oraz","dla","z","na","do","w","i","że","jest","są"
}

# -------------------------
# IO helpers
# -------------------------
def _json_load_maybe(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="replace")
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return x
    return x


def _find_content_byid(document: Dict[str, Any]) -> Dict[str, Any]:
    cd = document.get("contentDomain", {})
    by_id = cd.get("byId", {})
    return by_id if isinstance(by_id, dict) else {}


# -------------------------
# Demands loading + index
# -------------------------
def _load_demands_xlsx() -> List[Dict[str, str]]:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "demands.xlsx"),
        os.path.join(here, "assets", "demands.xlsx"),
    ]
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if model_dir:
        candidates += [
            os.path.join(model_dir, "demands.xlsx"),
            os.path.join(model_dir, "assets", "demands.xlsx"),
        ]

    for p in candidates:
        if os.path.exists(p):
            df = pd.read_excel(p)
            df.columns = [c.strip().lower() for c in df.columns]
            if "demand_id" not in df.columns and "id" in df.columns:
                df = df.rename(columns={"id": "demand_id"})
            if "description" not in df.columns and "demand_description" in df.columns:
                df = df.rename(columns={"demand_description": "description"})

            required = {"demand_id", "demand", "description"}
            missing = required - set(df.columns)
            if missing:
                raise RuntimeError(f"demands.xlsx missing columns: {sorted(list(missing))}. Found: {list(df.columns)}")

            out: List[Dict[str, str]] = []
            for _, r in df.fillna("").iterrows():
                did = str(r["demand_id"]).strip()
                if not did or did.lower() == "nan":
                    continue
                out.append(
                    {
                        "id": did,
                        "demand": str(r["demand"]).strip(),
                        "description": str(r["description"]).strip(),
                    }
                )
            logger.info("Loaded %d demands from %s", len(out), p)
            return out

    raise RuntimeError(f"demands.xlsx not found. Tried: {candidates}")


def _tokenize(s: str) -> set:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9ąćęłńóśźż]+", " ", s)
    toks = [t for t in s.split() if len(t) >= 3 and t not in STOPWORDS]
    return set(toks)


def _build_demand_indexes():
    global DEMAND_TOKENSETS, DEMAND_TEXT
    DEMAND_TOKENSETS = {}
    DEMAND_TEXT = {}
    for d in DEMANDS:
        did = d["id"]
        name = d.get("demand", "") or ""
        desc = d.get("description", "") or ""
        DEMAND_TOKENSETS[did] = _tokenize(name + " " + desc)
        desc_short = desc if len(desc) <= 240 else (desc[:240] + "…")
        DEMAND_TEXT[did] = f"{name} | {desc_short}"


# -------------------------
# Azure OpenAI init + embeddings
# -------------------------
def _init_openai():
    global client, CHAT_DEPLOYMENT, EMB_DEPLOYMENT

    if AzureOpenAI is None:
        raise RuntimeError("openai package not available")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip() or "2024-06-01"

    CHAT_DEPLOYMENT = (
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    )
    EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()

    if not (endpoint and api_key and CHAT_DEPLOYMENT):
        raise RuntimeError(
            "Missing AOAI env vars. Need: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_CHAT_DEPLOYMENT (or AZURE_OPENAI_DEPLOYMENT)."
        )

    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    if not EMB_DEPLOYMENT:
        logger.warning("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not set -> falling back to lexical retrieval.")
    else:
        logger.info("Using embedding retrieval with deployment: %s", EMB_DEPLOYMENT)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    AzureOpenAI embeddings call. Returns list of vectors.
    """
    if not EMB_DEPLOYMENT:
        return []

    # The SDK supports batching; keep moderate batch size
    vectors: List[List[float]] = []
    for i in range(0, len(texts), EMB_BATCH):
        batch = texts[i : i + EMB_BATCH]
        resp = client.embeddings.create(model=EMB_DEPLOYMENT, input=batch)
        # resp.data preserves order
        for item in resp.data:
            vectors.append(item.embedding)
    return vectors


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(a: List[float]) -> float:
    return float(sum(x * x for x in a) ** 0.5)


def _cosine(a: List[float], b: List[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _build_demand_embeddings():
    """
    Precompute embeddings for demands at init (id -> vector).
    """
    global DEMAND_EMB
    if not EMB_DEPLOYMENT:
        DEMAND_EMB = {}
        return

    ids = [d["id"] for d in DEMANDS]
    texts = []
    for d in DEMANDS:
        # Embedding on demand + description (best signal)
        name = d.get("demand", "") or ""
        desc = d.get("description", "") or ""
        texts.append(f"{name}\n{desc}".strip())

    vecs = _embed_texts(texts)
    if len(vecs) != len(ids):
        logger.warning("Embedding count mismatch: ids=%d vecs=%d -> fallback to lexical", len(ids), len(vecs))
        DEMAND_EMB = {}
        return

    DEMAND_EMB = {did: v for did, v in zip(ids, vecs)}
    logger.info("Built demand embeddings: %d vectors", len(DEMAND_EMB))


# -------------------------
# Chunk -> lines
# -------------------------
def _split_lines_or_sentences(chunk_text: str) -> List[str]:
    t = (chunk_text or "").strip()
    if not t:
        return []
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    tableish = (len(lines) >= 6) or any(("|" in ln or "\t" in ln) for ln in lines)
    if tableish:
        out = [ln for ln in lines if len(ln) >= MIN_CHUNK_LEN]
        return out[:MAX_LINES_PER_CHUNK]
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts[:MAX_LINES_PER_CHUNK]


def _line_priority_score(line: str) -> float:
    t = (line or "").strip()
    if len(t) < MIN_CHUNK_LEN:
        return 0.0
    has_modal = bool(REQ_MODAL_RE.search(t) or REQ_PL_MODAL_RE.search(t))
    has_num = bool(REQ_NUM_RE.search(t))
    has_unit = bool(UNIT_RE.search(t))
    if has_modal and (has_num or has_unit):
        return 0.90
    if has_modal:
        return 0.75
    if has_num and has_unit:
        return 0.70
    if has_num:
        return 0.60
    return 0.20


# -------------------------
# Candidate retrieval (NEW: embeddings first)
# -------------------------
def _candidate_retrieval(sentence: str, k: int) -> List[str]:
    """
    Prefer embedding similarity retrieval.
    Fallback to lexical overlap if embeddings are unavailable.
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return []

    if EMB_DEPLOYMENT and DEMAND_EMB:
        sent_vecs = _embed_texts([sentence])
        if sent_vecs:
            sv = sent_vecs[0]
            scored: List[Tuple[str, float]] = []
            for did, dv in DEMAND_EMB.items():
                sim = _cosine(sv, dv)
                if sim >= EMB_SIM_MIN:
                    scored.append((did, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [did for did, _ in scored[: max(1, k)]]

    # fallback lexical
    toks = _tokenize(sentence)
    if not toks:
        return []
    scored2: List[Tuple[str, int]] = []
    for did, dtoks in DEMAND_TOKENSETS.items():
        overlap = len(toks & dtoks)
        if overlap <= 0:
            continue
        scored2.append((did, overlap))
    scored2.sort(key=lambda x: x[1], reverse=True)
    return [did for did, _ in scored2[: max(1, k)]]


def _build_candidates_block(candidate_ids: List[str]) -> str:
    return "\n".join([f"- {did}: {DEMAND_TEXT.get(did,'')}" for did in candidate_ids])


# -------------------------
# LLM (single prompt)
# -------------------------
def _llm_predict(sentence: str, candidate_ids: List[str], top_k: int) -> Dict[str, Any]:
    if client is None or not CHAT_DEPLOYMENT:
        return {"isRequirement": False, "relevance": 0.0, "labels": [], "notes": "no_client"}

    candidates_block = _build_candidates_block(candidate_ids)
    if not candidates_block:
        return {"isRequirement": False, "relevance": 0.0, "labels": [], "notes": "no_candidates"}

    system = (
        "You are a strict requirements classifier for bid/spec documents. "
        "Return STRICT JSON only. No markdown. No prose. "
        "Never invent demand IDs. Only label when the DEMAND DESCRIPTION conditions are satisfied."
    )

    user = f"""
You have ONE sentence/line.

STEP 1 — REQUIREMENT GATE:
Decide if this sentence is a concrete, verifiable requirement/specification to highlight.
If not a requirement: isRequirement=false, relevance<{RELEVANCE_MIN}, labels=[].

STEP 2 — LABELING (only if isRequirement=true):
Pick up to {min(int(top_k), MAX_LABELS_PER_CHUNK)} IDs from CANDIDATES only.

For EACH label:
- probability >= {LABEL_MIN_PROB}
- evidence: exact quote copied from the sentence (5–25 words), MUST be a substring
- criteriaMatched: 1–3 short phrases reflecting DESCRIPTION conditions (the "WHEN")
- If you cannot find criteria from the description that are met here, DO NOT label.

CANDIDATES (id: demand | description):
{candidates_block}

SENTENCE:
{sentence}

Return JSON ONLY:
{{
  "isRequirement": true,
  "relevance": 0.0,
  "labels": [
    {{
      "id": "<candidate_id>",
      "probability": 0.0,
      "evidence": "<exact substring from sentence>",
      "criteriaMatched": ["<phrase1>", "<phrase2>"]
    }}
  ],
  "notes": "short"
}}
""".strip()

    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=900,
    )

    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# -------------------------
# Post-filters (tight)
# -------------------------
def _criteria_hits(sentence: str, criteria_list: List[str]) -> int:
    txt = (sentence or "").lower()
    hits = 0
    for c in criteria_list:
        c = (c or "").lower().strip()
        if not c:
            continue
        for tok in re.split(r"\s+", c):
            tok = tok.strip()
            if len(tok) >= 4 and tok in txt:
                hits += 1
                break
    return hits


def _lexical_overlap(sentence: str, demand_id: str) -> int:
    sent_toks = _tokenize(sentence)
    dem_toks = DEMAND_TOKENSETS.get(demand_id, set())
    return len(sent_toks & dem_toks)


def _sanitize(sentence: str, candidate_ids: List[str], data: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    is_req = bool(data.get("isRequirement", False))
    try:
        rel = float(data.get("relevance", 0.0) or 0.0)
    except Exception:
        rel = 0.0

    labels_raw = data.get("labels", []) or []
    if (not is_req) or (rel < RELEVANCE_MIN) or (not isinstance(labels_raw, list)):
        return {"isRequirement": False, "relevance": float(rel), "labels": []}

    labels: List[Dict[str, Any]] = []
    for it in labels_raw[: min(int(top_k), MAX_LABELS_PER_CHUNK)]:
        if not isinstance(it, dict):
            continue

        did = str(it.get("id", "")).strip()
        if did not in candidate_ids:
            continue

        try:
            prob = float(it.get("probability", 0.0) or 0.0)
        except Exception:
            continue
        if prob < LABEL_MIN_PROB:
            continue

        evidence = str(it.get("evidence", "") or "").strip()
        if not evidence or evidence.lower() not in sentence.lower():
            continue

        crit = it.get("criteriaMatched", [])
        if not isinstance(crit, list):
            crit = []
        crit = [str(c).strip() for c in crit if str(c).strip()]
        if not crit:
            continue

        if _criteria_hits(sentence, crit) < CRITERIA_HITS_MIN:
            continue

        # FINAL SAFETY:
        # If embeddings are strong, we accept; otherwise require minimal lexical overlap.
        # This blocks "spare parts" -> "earthing" even if LLM tries.
        if EMB_DEPLOYMENT and DEMAND_EMB:
            # If embeddings exist, candidates already filtered by sim; still keep a mild overlap fallback
            if _lexical_overlap(sentence, did) < OVERLAP_MIN:
                # allow only if demand tokens are sparse; keep conservative:
                continue
        else:
            if _lexical_overlap(sentence, did) < max(1, OVERLAP_MIN):
                continue

        labels.append({"id": did, "probability": max(0.0, min(1.0, prob))})

    labels.sort(key=lambda x: x["probability"], reverse=True)
    labels = labels[:MAX_LABELS_PER_CHUNK]

    if not labels:
        return {"isRequirement": False, "relevance": float(rel), "labels": []}

    rel = max(float(rel), float(labels[0]["probability"]))
    return {"isRequirement": True, "relevance": float(rel), "labels": labels}


# -------------------------
# AML entrypoints
# -------------------------
def init():
    global DEMANDS, client

    DEMANDS = _load_demands_xlsx()
    _build_demand_indexes()

    if not USE_MOCK:
        _init_openai()
        _build_demand_embeddings()

    logger.info(
        "init(): demands=%d embeddings=%s CAND_TOPK=%d LABEL_MIN_PROB=%.2f HIGHLIGHT_MIN_PROB=%.2f",
        len(DEMANDS),
        "on" if (EMB_DEPLOYMENT and DEMAND_EMB) else "off",
        CANDIDATES_TOPK,
        LABEL_MIN_PROB,
        HIGHLIGHT_MIN_PROB,
    )


def run(raw_data):
    """
    Contract (do not break):
      { "predictions": <document> }

    document MUST include:
      documentDemandPredictions: ["id1","id2", ...]  (ARRAY OF STRINGS)

    per chunk:
      relevantProba: float
      cdTransformerPredictions: [{label, proba}]
      cdLogregPredictions: SAME SHAPE (mirror)
    """
    try:
        req = _json_load_maybe(raw_data)
        if not isinstance(req, dict):
            return {"error": "Bad request: body must be JSON object", "predictions": {"documentDemandPredictions": []}}

        if "document" not in req:
            return {"error": "Bad request: missing 'document'", "predictions": {"documentDemandPredictions": []}}

        document = _json_load_maybe(req["document"])
        if not isinstance(document, dict):
            return {"error": "Bad request: 'document' must be JSON object", "predictions": {"documentDemandPredictions": []}}

        num_preds = int(req.get("num_preds", MAX_LABELS_PER_CHUNK))
        num_preds = max(1, min(10, num_preds))
        num_preds = min(num_preds, MAX_LABELS_PER_CHUNK)

        by_id = _find_content_byid(document)
        best_doc_probs: Dict[str, float] = {}

        debug_left = MAX_DEBUG_CHUNKS

        for cid, content in by_id.items():
            if not isinstance(content, dict):
                continue

            chunk_text = str(content.get("text", "") or "").strip()

            # Anti-crash defaults
            content["relevantProba"] = 0.0
            content["cdTransformerPredictions"] = []
            content["cdLogregPredictions"] = []
            content["highlightText"] = ""

            if debug_left > 0:
                logger.info("chunk=%s sample=%r", cid, chunk_text[:200])
                debug_left -= 1

            if len(chunk_text) < MIN_CHUNK_LEN:
                continue

            lines = _split_lines_or_sentences(chunk_text)
            if not lines:
                continue

            ranked = sorted([( _line_priority_score(ln), ln) for ln in lines], key=lambda x: x[0], reverse=True)
            candidate_lines = [ln for _, ln in ranked[:MAX_LINES_TO_LLM] if ln.strip()]

            best_sentence = ""
            best_top1 = 0.0
            best_rel = 0.0
            best_labels: List[Dict[str, Any]] = []

            for sentence in candidate_lines:
                cand_ids = _candidate_retrieval(sentence, CANDIDATES_TOPK)
                if not cand_ids:
                    continue

                if USE_MOCK:
                    # mock: pick top candidate deterministically
                    data = {
                        "isRequirement": True,
                        "relevance": 0.8,
                        "labels": [{"id": cand_ids[0], "probability": 0.85, "evidence": sentence[:80], "criteriaMatched": ["mock"]}],
                    }
                else:
                    data = _llm_predict(sentence, cand_ids, num_preds)

                sanitized = _sanitize(sentence, cand_ids, data, num_preds)
                if not sanitized.get("isRequirement", False):
                    continue

                rel = float(sanitized.get("relevance", 0.0) or 0.0)
                labels = sanitized.get("labels", []) or []
                if not labels:
                    continue

                top1 = float(labels[0]["probability"])
                if top1 > best_top1:
                    best_top1 = top1
                    best_rel = rel
                    best_sentence = sentence
                    best_labels = labels

            if not best_labels or best_top1 < HIGHLIGHT_MIN_PROB:
                continue

            ui_preds = [{"label": str(x["id"]), "proba": float(x["probability"])} for x in best_labels[:num_preds]]
            ui_preds.sort(key=lambda x: x["proba"], reverse=True)

            content["relevantProba"] = float(max(best_rel, best_top1))
            content["cdTransformerPredictions"] = ui_preds
            content["cdLogregPredictions"] = list(ui_preds)
            content["highlightText"] = best_sentence

            for p in ui_preds:
                did = p["label"]
                pr = float(p["proba"])
                if did not in best_doc_probs or pr > best_doc_probs[did]:
                    best_doc_probs[did] = pr

        ids_sorted = [k for k, v in sorted(best_doc_probs.items(), key=lambda kv: kv[1], reverse=True)]
        document["documentDemandPredictions"] = ids_sorted[:num_preds]

        return {"predictions": document}

    except Exception as e:
        logger.exception("run() failed")
        return {"error": str(e), "predictions": {"documentDemandPredictions": []}}
