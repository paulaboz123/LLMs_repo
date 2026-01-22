import json
import logging
import os
import re
import math
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("score")

# ============================================================
# Configuration (ENV)
# ============================================================
USE_MOCK = os.getenv("USE_MOCK", "0").strip() == "1"  # if 1 -> never call LLM

# Sentence/chunk handling
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "10"))
MAX_SENTENCES_PER_CHUNK = int(os.getenv("MAX_SENTENCES_PER_CHUNK", "12"))
MAX_SENTENCES_TO_EVAL = int(os.getenv("MAX_SENTENCES_TO_EVAL", "10"))

# Output constraints / contract
MAX_LABELS_PER_CHUNK = int(os.getenv("MAX_LABELS_PER_CHUNK", "3"))
NUM_PREDS_CAP = 10

# Thresholds (LLM probabilities are trusted more than heuristic)
LABEL_MIN_PROB = float(os.getenv("LABEL_MIN_PROB", "0.70"))
HIGHLIGHT_MIN_PROB = float(os.getenv("HIGHLIGHT_MIN_PROB", "0.78"))

# Candidate retrieval
CANDIDATES_TOPK = int(os.getenv("CANDIDATES_TOPK", "20"))
LLM_CANDIDATES = int(os.getenv("LLM_CANDIDATES", "8"))  # how many candidates sent to LLM per sentence
FORCE_TOPIC_GATING = os.getenv("FORCE_TOPIC_GATING", "1").strip() == "1"

# Minimal artifact rejection ONLY (avoid removing useful content)
REJECT_TOC = os.getenv("REJECT_TOC", "1").strip() == "1"
REJECT_PAGE_ARTIFACTS = os.getenv("REJECT_PAGE_ARTIFACTS", "1").strip() == "1"

# Multi-label
ALLOW_MULTI_LABEL = os.getenv("ALLOW_MULTI_LABEL", "1").strip() == "1"
MAX_LLM_LABELS = int(os.getenv("MAX_LLM_LABELS", "3"))

# Fallback deterministic scoring weights (when LLM off/fails)
W_TOPIC_HIT = float(os.getenv("W_TOPIC_HIT", "10.0"))
W_TOPIC_IDF = float(os.getenv("W_TOPIC_IDF", "3.0"))
W_DESC_IDF = float(os.getenv("W_DESC_IDF", "2.0"))
W_ASSET_HIT = float(os.getenv("W_ASSET_HIT", "0.6"))

# Probability mapping fallback
PROB_CAP = float(os.getenv("PROB_CAP", "0.95"))
PROB_SCALE = float(os.getenv("PROB_SCALE", "12.0"))
PROB_BIAS = float(os.getenv("PROB_BIAS", "0.02"))

# LLM settings
USE_LLM = os.getenv("USE_LLM", "1").strip() == "1" and not USE_MOCK  # master switch
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))

MAX_DEBUG_CHUNKS = int(os.getenv("MAX_DEBUG_CHUNKS", "0"))

# ============================================================
# Global state
# ============================================================
client = None
CHAT_DEPLOYMENT = None

DEMANDS: List[Dict[str, str]] = []
DEMAND_NAME_RAW: Dict[str, str] = {}
DEMAND_DESC_RAW: Dict[str, str] = {}

DEMAND_ASSET: Dict[str, str] = {}
DEMAND_TOPIC: Dict[str, str] = {}

DEMAND_TOPIC_TOKS: Dict[str, set] = {}
DEMAND_DESC_TOKS: Dict[str, set] = {}

TOPIC_TO_IDS: Dict[str, List[str]] = {}

TOKEN_DF: Dict[str, int] = {}
TOKEN_IDF: Dict[str, float] = {}

# DemandCard cache sent to LLM (built per demand id)
DEMAND_CARD: Dict[str, str] = {}

# ============================================================
# NLP helpers
# ============================================================
STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "on", "for", "with", "a", "an", "as", "at", "by",
    "from", "into", "over", "under", "than", "then", "that", "this", "these", "those",
    "be", "is", "are", "was", "were", "been", "being",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "can", "could", "may", "might", "will", "would",
    "not", "no",
    # DO NOT include: test, routine, type, measurement(s), etc.
}

TOC_DOTS_RE = re.compile(r"^\s*(\d+(\.\d+)*\s+)?[A-Za-z].{2,}\.{5,}\s*\d+\s*$")
TOC_SECTION_PAGE_RE = re.compile(r"^\s*\d+(\.\d+)+\s+[A-Za-z].{2,}\s+\d+\s*$")
PAGE_ONLY_RE = re.compile(r"^\s*\d+\s*$")
PAGE_FRACTION_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
PAGE_WORD_RE = re.compile(r"^\s*page\s*\d+\s*$", re.IGNORECASE)

_DASH_SPLIT_RE = re.compile(r"\s*[-–—]\s*")

# light requirement heuristic (not a hard gate)
REQ_MODAL_RE = re.compile(r"\b(shall|must|required|requirement|should|need to|has to|may not|must not|shall not)\b", re.IGNORECASE)
REQ_IMP_RE = re.compile(r"^\s*(use|provide|ensure|verify|test|install|maintain|replace|include|apply|set|adjust|calibrate|limit)\b", re.IGNORECASE)
REQ_NUM_RE = re.compile(r"\b\d+([.,]\d+)?\b")
UNIT_RE = re.compile(r"\b(mm|cm|m|kg|w|kw|v|kv|a|hz|rpm|bar|pa|°c|ip\d{2})\b", re.IGNORECASE)


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
                out.append({"id": did, "demand": str(r["demand"]).strip(), "description": str(r["description"]).strip()})
            logger.info("Loaded %d demands from %s", len(out), p)
            return out

    raise RuntimeError(f"demands.xlsx not found. Tried: {candidates}")


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = []
    for t in s.split():
        if len(t) < 2:
            continue
        if t in STOPWORDS:
            continue
        toks.append(t)
    return toks


def _tokset(s: str) -> set:
    return set(_tokenize(s))


def _word_count(s: str) -> int:
    return len([w for w in re.findall(r"[A-Za-z0-9]+", s or "") if w.strip()])


def _is_toc_line(s: str) -> bool:
    if not REJECT_TOC:
        return False
    t = (s or "").strip()
    if len(t) < 6:
        return False
    return bool(TOC_DOTS_RE.match(t) or TOC_SECTION_PAGE_RE.match(t))


def _is_page_artifact(s: str) -> bool:
    if not REJECT_PAGE_ARTIFACTS:
        return False
    t = (s or "").strip()
    if not t:
        return True
    return bool(PAGE_ONLY_RE.match(t) or PAGE_FRACTION_RE.match(t) or PAGE_WORD_RE.match(t))


def _should_reject_sentence(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True
    if len(t) < MIN_CHUNK_LEN:
        return True
    if _word_count(t) < 3:
        return True
    if _is_page_artifact(t):
        return True
    if _is_toc_line(t):
        return True
    return False


def _split_into_sentences_or_lines(chunk_text: str) -> List[str]:
    t = (chunk_text or "").strip()
    if not t:
        return []

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    tableish = (len(lines) >= 6) or any(("|" in ln or "\t" in ln) for ln in lines)
    if tableish:
        out = [ln for ln in lines if len(ln) >= MIN_CHUNK_LEN]
        return out[:MAX_SENTENCES_PER_CHUNK]

    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts[:MAX_SENTENCES_PER_CHUNK]


def _is_requirement_fast(text: str) -> Tuple[bool, float]:
    t = (text or "").strip()
    if len(t) < MIN_CHUNK_LEN:
        return False, 0.0
    has_modal = bool(REQ_MODAL_RE.search(t))
    has_imp = bool(REQ_IMP_RE.search(t))
    has_num = bool(REQ_NUM_RE.search(t))
    has_unit = bool(UNIT_RE.search(t))
    if (has_modal or has_imp) and (has_num or has_unit):
        return True, 0.86
    if has_modal or has_imp:
        return True, 0.72
    if has_num and has_unit:
        return True, 0.66
    return False, 0.25


# ============================================================
# Demand parsing + IDF
# ============================================================
def _parse_demand_parts(demand_name: str) -> Tuple[str, str]:
    """
    "CST - Motor - painting" => asset="motor", topic="painting"
    CST ignored.
    """
    raw = (demand_name or "").strip()
    parts = [p.strip() for p in _DASH_SPLIT_RE.split(raw) if p.strip()]
    parts_l = [p.lower() for p in parts]

    asset = ""
    if len(parts_l) >= 2:
        if "motor" in parts_l[1]:
            asset = "motor"
        elif "pump" in parts_l[1]:
            asset = "pump"

    topic = parts_l[-1].strip() if parts_l else ""
    topic = re.sub(r"\s+", " ", topic).strip()
    return asset, topic


def _topic_variants(topic: str) -> List[str]:
    """
    Robust matching without synonyms list:
    - singular/plural
    - hyphen/space variants
    - and: if topic is multiword, include single important tokens (>=4 chars)
      (this is safe because topic gating is only used when *some* topic token exists)
    """
    t = (topic or "").strip().lower()
    if not t:
        return []
    out = {t, t.replace("-", " "), t.replace(" ", "-")}

    if t.endswith("s") and len(t) > 3:
        out.add(t[:-1])
    else:
        out.add(t + "s")

    # Multiword -> include tokens as weak variants
    toks = [x for x in re.split(r"[^a-z0-9]+", t) if x]
    for x in toks:
        if len(x) >= 4:
            out.add(x)
        if len(x) >= 4 and x.endswith("s"):
            out.add(x[:-1])

    return sorted({x.strip() for x in out if x.strip()})


def _topic_hit(sentence_lower: str, topic: str) -> bool:
    for v in _topic_variants(topic):
        if re.search(rf"\b{re.escape(v)}\b", sentence_lower):
            return True
    return False


def _build_idf():
    global TOKEN_DF, TOKEN_IDF
    TOKEN_DF = {}
    TOKEN_IDF = {}
    N = max(1, len(DEMANDS))

    for d in DEMANDS:
        did = d["id"]
        toks = set()
        toks |= DEMAND_TOPIC_TOKS.get(did, set())
        toks |= DEMAND_DESC_TOKS.get(did, set())
        for tok in toks:
            TOKEN_DF[tok] = TOKEN_DF.get(tok, 0) + 1

    for tok, df in TOKEN_DF.items():
        TOKEN_IDF[tok] = math.log((N + 1.0) / (df + 1.0)) + 1.0


def _build_demand_cards():
    """
    Minimal DemandCard: topic + short description.
    No synonyms / no examples (per your constraint).
    """
    global DEMAND_CARD
    DEMAND_CARD = {}
    for d in DEMANDS:
        did = d["id"]
        name = DEMAND_NAME_RAW.get(did, "") or ""
        desc = DEMAND_DESC_RAW.get(did, "") or ""
        asset = DEMAND_ASSET.get(did, "") or ""
        topic = DEMAND_TOPIC.get(did, "") or ""
        # Keep it short and consistent for LLM
        DEMAND_CARD[did] = (
            f"ID: {did}\n"
            f"ASSET: {asset or 'n/a'}\n"
            f"TOPIC: {topic or 'n/a'}\n"
            f"DEFINITION: {desc.strip()}\n"
        )


def _build_demand_indexes():
    global DEMAND_NAME_RAW, DEMAND_DESC_RAW, DEMAND_ASSET, DEMAND_TOPIC, DEMAND_TOPIC_TOKS, DEMAND_DESC_TOKS, TOPIC_TO_IDS

    DEMAND_NAME_RAW = {}
    DEMAND_DESC_RAW = {}
    DEMAND_ASSET = {}
    DEMAND_TOPIC = {}
    DEMAND_TOPIC_TOKS = {}
    DEMAND_DESC_TOKS = {}
    TOPIC_TO_IDS = {}

    for d in DEMANDS:
        did = d["id"]
        name = (d.get("demand", "") or "").strip()
        desc = (d.get("description", "") or "").strip()

        DEMAND_NAME_RAW[did] = name
        DEMAND_DESC_RAW[did] = desc

        asset, topic = _parse_demand_parts(name)
        DEMAND_ASSET[did] = asset
        DEMAND_TOPIC[did] = topic

        DEMAND_TOPIC_TOKS[did] = _tokset(topic)
        DEMAND_DESC_TOKS[did] = _tokset(desc)

        if topic:
            TOPIC_TO_IDS.setdefault(topic, []).append(did)

    _build_idf()
    _build_demand_cards()


def _idf_overlap(sentence_toks: set, demand_toks: set) -> float:
    if not sentence_toks or not demand_toks:
        return 0.0
    ov = sentence_toks & demand_toks
    if not ov:
        return 0.0
    return float(sum(TOKEN_IDF.get(t, 1.0) for t in ov))


def _topics_present_in_sentence(sentence: str) -> List[str]:
    s_lower = (sentence or "").lower()
    present = []
    for topic in TOPIC_TO_IDS.keys():
        if _topic_hit(s_lower, topic):
            present.append(topic)
    return present


def _candidate_ids(sentence: str) -> List[str]:
    """
    Candidate retrieval:
      - If topic(s) explicitly present: ONLY those topics' demands (strict gating).
      - Else: IDF overlap shortlist.
    """
    present_topics = _topics_present_in_sentence(sentence)
    if present_topics and FORCE_TOPIC_GATING:
        out = []
        seen = set()
        for t in present_topics:
            for did in TOPIC_TO_IDS.get(t, []):
                if did not in seen:
                    seen.add(did)
                    out.append(did)
        return out[:max(1, CANDIDATES_TOPK)]

    stoks = _tokset(sentence)
    scored = []
    for d in DEMANDS:
        did = d["id"]
        sc = _idf_overlap(stoks, DEMAND_TOPIC_TOKS.get(did, set())) + _idf_overlap(stoks, DEMAND_DESC_TOKS.get(did, set()))
        if sc > 0.0:
            scored.append((did, sc))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [did for did, _ in scored[:max(1, CANDIDATES_TOPK)]]


# ============================================================
# Fallback deterministic scoring (used if LLM off/fails)
# ============================================================
def _score_fallback(sentence: str, did: str) -> Tuple[float, Dict[str, Any]]:
    s = sentence or ""
    s_lower = s.lower()
    stoks = _tokset(s)

    topic = DEMAND_TOPIC.get(did, "")
    asset = DEMAND_ASSET.get(did, "")

    topic_hit = bool(topic and _topic_hit(s_lower, topic))
    topic_idf = _idf_overlap(stoks, DEMAND_TOPIC_TOKS.get(did, set()))
    desc_idf = _idf_overlap(stoks, DEMAND_DESC_TOKS.get(did, set()))
    asset_hit = bool(asset and re.search(rf"\b{re.escape(asset)}\b", s_lower))

    score = 0.0
    if topic_hit:
        score += W_TOPIC_HIT
    score += W_TOPIC_IDF * topic_idf
    score += W_DESC_IDF * desc_idf
    if asset_hit:
        score += W_ASSET_HIT

    meta = {
        "topic": topic,
        "asset": asset,
        "topic_hit": topic_hit,
        "topic_idf": topic_idf,
        "desc_idf": desc_idf,
        "asset_hit": asset_hit,
    }
    return float(score), meta


def _score_to_prob_fallback(score: float) -> float:
    s = max(0.0, float(score))
    p = 1.0 - math.exp(-s / max(1e-6, PROB_SCALE))
    p = min(PROB_CAP, max(0.0, p + PROB_BIAS))
    return float(p)


# ============================================================
# LLM Verification (RAG over DemandCards)
# ============================================================
def _llm_verify(sentence: str, candidate_ids: List[str]) -> Optional[Dict[str, Any]]:
    """
    One call does both:
      - requirement/caption judgement
      - select matching demand ids (from candidate list only)
    Returns:
      { isRequirement: bool, relevance: float, labels: [{id, probability}], notes: str }
    """
    if client is None or not CHAT_DEPLOYMENT:
        return None
    if not candidate_ids:
        return None

    cards = []
    for did in candidate_ids[:LLM_CANDIDATES]:
        cards.append(DEMAND_CARD.get(did, f"ID: {did}\nDEFINITION: {DEMAND_DESC_RAW.get(did,'')}\n"))
    cards_text = "\n---\n".join(cards)

    # The critical instruction: no hallucinated ids, and do not match unrelated topics.
    system = "You are a strict information extraction and classification engine. Return STRICT JSON only."
    user = f"""
Task:
1) Decide if the SENTENCE should be HIGHLIGHTED as a concrete requirement/specification (not a TOC line, not a section/table title, not a generic heading).
2) From the provided DEMAND CARDS, select the demand IDs that the sentence truly satisfies.

Rules:
- You MUST choose IDs only from the provided Demand Cards.
- If the sentence contains explicit topic words (e.g., 'painting', 'coating', 'vendor'), do NOT assign an unrelated topic ID.
- Prefer precision: return fewer labels rather than guessing.
- If it's a heading/caption/TOC (e.g., "Table 2.1 ...", "1.1 Spare parts .... 12"), set isRequirement=false and return empty labels.
- The probabilities are 0..1. Use 0.85+ only when it is clearly a match.

Return JSON only in this schema:
{{
  "isRequirement": true|false,
  "relevance": 0.0,
  "labels": [{{"id":"<demand_id>","probability":0.0}}],
  "notes": "short reason"
}}

SENTENCE:
{sentence}

DEMAND CARDS:
{cards_text}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)

        is_req = bool(data.get("isRequirement", False))
        rel = float(data.get("relevance", 0.0) or 0.0)
        labels = data.get("labels", []) or []
        if not isinstance(labels, list):
            labels = []

        # sanitize: keep only candidate ids, clamp probabilities
        allowed = set(candidate_ids[:LLM_CANDIDATES])
        out_labels = []
        for it in labels:
            if not isinstance(it, dict):
                continue
            did = str(it.get("id", "")).strip()
            if did not in allowed:
                continue
            p = float(it.get("probability", 0.0) or 0.0)
            p = max(0.0, min(0.95, p))
            out_labels.append({"id": did, "probability": p})

        out_labels.sort(key=lambda x: x["probability"], reverse=True)
        out_labels = out_labels[:max(1, min(MAX_LLM_LABELS, MAX_LABELS_PER_CHUNK))]

        # If LLM returns 0 labels, still return structure (caller decides highlight)
        return {
            "isRequirement": is_req,
            "relevance": rel,
            "labels": out_labels,
            "notes": str(data.get("notes", "") or "")[:200],
        }
    except Exception:
        return None


# ============================================================
# Unified sentence prediction
# ============================================================
def _predict_sentence(sentence: str, top_k: int) -> Dict[str, Any]:
    if _should_reject_sentence(sentence):
        return {"isRequirement": False, "relevance": 0.0, "labels": [], "meta": {}, "notes": "rejected"}

    is_req_fast, rel_fast = _is_requirement_fast(sentence)
    cand_ids = _candidate_ids(sentence)

    # LLM verification (preferred)
    if USE_LLM:
        llm = _llm_verify(sentence, cand_ids)
        if llm is not None:
            labels = llm.get("labels", []) or []
            # If LLM returned labels, trust them (and keep multi-label)
            if labels:
                labels_sorted = sorted(labels, key=lambda x: float(x.get("probability", 0.0)), reverse=True)
                labels_sorted = labels_sorted[:top_k]
                rel = max(float(llm.get("relevance", 0.0) or 0.0), float(labels_sorted[0]["probability"]))
                return {
                    "isRequirement": bool(llm.get("isRequirement", False)),
                    "relevance": float(rel),
                    "labels": labels_sorted,
                    "meta": {"source": "llm", "candidates": cand_ids[:LLM_CANDIDATES]},
                    "notes": f"llm:{llm.get('notes','')}",
                }
            # If LLM says not requirement, keep it as not highlight (but allow fallback for label suggestion if desired)
            if bool(llm.get("isRequirement", False)) is False:
                return {
                    "isRequirement": False,
                    "relevance": float(llm.get("relevance", rel_fast) or rel_fast),
                    "labels": [],
                    "meta": {"source": "llm", "candidates": cand_ids[:LLM_CANDIDATES]},
                    "notes": f"llm_no_labels:{llm.get('notes','')}",
                }
        # If LLM failed, fall through to deterministic fallback.

    # Deterministic fallback
    scored: List[Tuple[str, float, Dict[str, Any]]] = []
    meta_all: Dict[str, Any] = {"source": "fallback", "candidates": cand_ids[:CANDIDATES_TOPK]}

    for did in (cand_ids[:CANDIDATES_TOPK] if cand_ids else [d["id"] for d in DEMANDS][:CANDIDATES_TOPK]):
        sc, m = _score_fallback(sentence, did)
        scored.append((did, sc, m))

    scored.sort(key=lambda x: x[1], reverse=True)

    labels = []
    for did, sc, _m in scored[: max(10, top_k * 5)]:
        p = _score_to_prob_fallback(sc)
        if p >= LABEL_MIN_PROB:
            labels.append({"id": did, "probability": float(p)})

    labels.sort(key=lambda x: x["probability"], reverse=True)
    labels = labels[:top_k]

    rel = max(float(rel_fast), float(labels[0]["probability"]) if labels else 0.0)
    return {
        "isRequirement": bool(is_req_fast),
        "relevance": float(rel),
        "labels": labels,
        "meta": meta_all,
        "notes": "fallback",
    }


# ============================================================
# Azure OpenAI init
# ============================================================
def _init_llm_client():
    global client, CHAT_DEPLOYMENT

    if not USE_LLM:
        client = None
        CHAT_DEPLOYMENT = None
        logger.info("LLM disabled (USE_LLM=%s USE_MOCK=%s)", str(USE_LLM), str(USE_MOCK))
        return

    if AzureOpenAI is None:
        logger.warning("openai package not available; running without LLM")
        client = None
        CHAT_DEPLOYMENT = None
        return

    aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    aoai_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    aoai_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip() or "2024-06-01"

    CHAT_DEPLOYMENT = (
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    )

    if not (aoai_endpoint and aoai_key and CHAT_DEPLOYMENT):
        logger.warning("Missing AOAI env vars; running without LLM verification")
        client = None
        CHAT_DEPLOYMENT = None
        return

    client = AzureOpenAI(
        azure_endpoint=aoai_endpoint,
        api_key=aoai_key,
        api_version=aoai_version,
    )
    logger.info("LLM enabled: deployment=%s", CHAT_DEPLOYMENT)


# ============================================================
# AML entrypoints
# ============================================================
def init():
    global DEMANDS, USE_MOCK
    USE_MOCK = os.getenv("USE_MOCK", "0").strip() == "1"

    DEMANDS = _load_demands_xlsx()
    _build_demand_indexes()
    _init_llm_client()

    logger.info(
        "init(): demands=%d USE_LLM=%s USE_MOCK=%s LABEL_MIN_PROB=%.2f HIGHLIGHT_MIN_PROB=%.2f "
        "CANDIDATES_TOPK=%d LLM_CANDIDATES=%d FORCE_TOPIC_GATING=%s",
        len(DEMANDS), str(USE_LLM), str(USE_MOCK), LABEL_MIN_PROB, HIGHLIGHT_MIN_PROB,
        CANDIDATES_TOPK, LLM_CANDIDATES, str(FORCE_TOPIC_GATING)
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
      highlightText: string
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
        num_preds = max(1, min(NUM_PREDS_CAP, num_preds))
        num_preds = min(num_preds, MAX_LABELS_PER_CHUNK)

        by_id = _find_content_byid(document)
        best_doc_probs: Dict[str, float] = {}

        debug_left = MAX_DEBUG_CHUNKS

        for cid, content in by_id.items():
            if not isinstance(content, dict):
                continue

            chunk_text = (str(content.get("text", "") or "")).strip()

            # Always set expected keys/types
            content["relevantProba"] = 0.0
            content["cdTransformerPredictions"] = []
            content["cdLogregPredictions"] = []
            content["highlightText"] = ""
            content["labelExplanations"] = []

            if debug_left > 0:
                logger.info("chunk=%s sample=%r", cid, chunk_text[:180])
                debug_left -= 1

            sentences_raw = _split_into_sentences_or_lines(chunk_text)
            if not sentences_raw:
                continue

            sentences = [s for s in sentences_raw if not _should_reject_sentence(s)]
            if not sentences:
                continue

            # rank sentences for evaluation: prefer those with any topic hits + requirement-ish cues
            ranked: List[Tuple[float, str]] = []
            for s in sentences:
                is_req, rel_fast = _is_requirement_fast(s)
                topics = _topics_present_in_sentence(s)
                topic_bonus = 0.15 * float(len(topics))  # encourages evaluating the right lines
                ranked.append((0.30 * rel_fast + topic_bonus + (0.02 if is_req else 0.0), s))
            ranked.sort(key=lambda x: x[0], reverse=True)
            to_eval = [s for _, s in ranked[:MAX_SENTENCES_TO_EVAL]]

            best_metric = -1.0
            best_sentence = ""
            best_pred: Optional[Dict[str, Any]] = None

            for s in to_eval:
                pred = _predict_sentence(s, top_k=num_preds)
                labels = pred.get("labels", []) or []
                if not labels:
                    continue

                top1 = float(labels[0].get("probability", 0.0) or 0.0)
                rel = float(pred.get("relevance", 0.0) or 0.0)
                is_req = bool(pred.get("isRequirement", False))

                # metric: prioritize correct label confidence + requirement-ness
                metric = top1 * (1.0 + 0.20 * rel) * (1.08 if is_req else 1.0)
                if metric > best_metric:
                    best_metric = metric
                    best_sentence = s
                    best_pred = pred

            if best_pred is None:
                continue

            labels = best_pred.get("labels", []) or []
            if not labels:
                continue

            labels_sorted = sorted(labels, key=lambda x: float(x.get("probability", 0.0) or 0.0), reverse=True)
            labels_sorted = labels_sorted[:num_preds]

            # enforce multi-label only if allowed (LLM already tends to be conservative)
            if not ALLOW_MULTI_LABEL and labels_sorted:
                labels_sorted = labels_sorted[:1]

            final_labels = [{"label": str(lb["id"]), "proba": float(lb["probability"])} for lb in labels_sorted]
            top1 = float(final_labels[0]["proba"]) if final_labels else 0.0
            is_req = bool(best_pred.get("isRequirement", False))
            rel = float(best_pred.get("relevance", 0.0) or 0.0)

            # Highlight rule:
            # - If LLM says requirement and top prob passes threshold -> highlight
            # - If LLM is off, fallback requirement heuristic is used (same behavior)
            should_highlight = bool((top1 >= HIGHLIGHT_MIN_PROB) and is_req)

            if not should_highlight:
                continue

            content["relevantProba"] = float(max(rel, top1))
            content["cdTransformerPredictions"] = final_labels
            content["cdLogregPredictions"] = list(final_labels)
            content["highlightText"] = best_sentence

            # optional explanations (safe, short)
            expls = []
            for p in final_labels:
                did = p["label"]
                desc = (DEMAND_DESC_RAW.get(did, "") or "").strip()
                expls.append({"label": did, "explanation": desc[:200]})
            content["labelExplanations"] = expls

            for p in final_labels:
                did = p["label"]
                prob = float(p["proba"])
                if did not in best_doc_probs or prob > best_doc_probs[did]:
                    best_doc_probs[did] = prob

        ids_sorted = [k for k, v in sorted(best_doc_probs.items(), key=lambda kv: kv[1], reverse=True)]
        document["documentDemandPredictions"] = ids_sorted[:num_preds]
        return {"predictions": document}

    except Exception as e:
        logger.exception("run() failed")
        return {"error": str(e), "predictions": {"documentDemandPredictions": []}}
