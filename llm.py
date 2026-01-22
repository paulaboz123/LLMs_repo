import json
import logging
import math
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

# =============================================================================
# Tunables (env)
# =============================================================================
USE_MOCK = os.getenv("USE_MOCK", "0").strip() == "1"  # if 1, never call LLM/embeddings

MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "10"))
MAX_LABELS_PER_CHUNK = int(os.getenv("MAX_LABELS_PER_CHUNK", "3"))
CANDIDATES_TOPK = int(os.getenv("CANDIDATES_TOPK", "12"))

LABEL_MIN_PROB = float(os.getenv("LABEL_MIN_PROB", "0.75"))
HIGHLIGHT_MIN_PROB = float(os.getenv("HIGHLIGHT_MIN_PROB", "0.80"))
RELEVANCE_MIN = float(os.getenv("RELEVANCE_MIN", "0.60"))

MAX_SENTENCES_PER_CHUNK = int(os.getenv("MAX_SENTENCES_PER_CHUNK", "12"))
MAX_SENTENCES_TO_MODEL = int(os.getenv("MAX_SENTENCES_TO_MODEL", "6"))

# Content filters
MIN_WORDS_PER_SENTENCE = int(os.getenv("MIN_WORDS_PER_SENTENCE", "4"))  # reject <=3 words
REJECT_TOC = os.getenv("REJECT_TOC", "1").strip() == "1"
REJECT_PAGE_ARTIFACTS = os.getenv("REJECT_PAGE_ARTIFACTS", "1").strip() == "1"

# A) category-first
CATEGORY_CONF_MIN = float(os.getenv("CATEGORY_CONF_MIN", "0.20"))

# B) embeddings for candidate retrieval/scoring (candidates only)
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "1").strip() == "1" and not USE_MOCK
EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "").strip()
EMBEDDINGS_BATCH = int(os.getenv("EMBEDDINGS_BATCH", "64"))
EMBEDDING_WEIGHT = float(os.getenv("EMBEDDING_WEIGHT", "2.0"))

# D) checklist constraints from description
CHECKLIST_ENFORCE = os.getenv("CHECKLIST_ENFORCE", "1").strip() == "1"
CHECKLIST_SIZE = int(os.getenv("CHECKLIST_SIZE", "6"))
CHECKLIST_MIN_HITS = int(os.getenv("CHECKLIST_MIN_HITS", "1"))
CHECKLIST_NGRAM_MAX = int(os.getenv("CHECKLIST_NGRAM_MAX", "2"))

# E) single-call LLM verify+explain per sentence (top K)
USE_LLM_VERIFY_EXPLAIN = os.getenv("USE_LLM_VERIFY_EXPLAIN", "1").strip() == "1" and not USE_MOCK
LLM_VERIFY_MAX_CANDS = int(os.getenv("LLM_VERIFY_MAX_CANDS", "3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "450"))

# Optional LLM requirement gate (OFF by default)
USE_LLM_GATE = os.getenv("USE_LLM_GATE", "0").strip() == "1" and not USE_MOCK

# Deterministic guards (keep recall safe)
MIN_CANDIDATE_TOKEN_OVERLAP = int(os.getenv("MIN_CANDIDATE_TOKEN_OVERLAP", "1"))
MIN_LABEL_TOKEN_OVERLAP = int(os.getenv("MIN_LABEL_TOKEN_OVERLAP", "2"))
MIN_DESC_OVERLAP_FOR_LABEL = int(os.getenv("MIN_DESC_OVERLAP_FOR_LABEL", "1"))
ZERO_LEXICAL_REJECT = os.getenv("ZERO_LEXICAL_REJECT", "1").strip() == "1"

# F) calibrated confidence (margin-based)
PROB_CAP = float(os.getenv("PROB_CAP", "0.95"))
KEEP_RELATIVE_TO_TOP = float(os.getenv("KEEP_RELATIVE_TO_TOP", "0.88"))
KEEP_ABSOLUTE_FLOOR = float(os.getenv("KEEP_ABSOLUTE_FLOOR", "0.70"))

SIG_A = float(os.getenv("SIG_A", "1.6"))
SIG_B = float(os.getenv("SIG_B", "2.0"))
MARGIN_A = float(os.getenv("MARGIN_A", "6.0"))
MARGIN_B = float(os.getenv("MARGIN_B", "0.08"))

# Highlight fallback: allow strong semantic match even if requirement gate is weak
ALLOW_STRONG_MATCH_HIGHLIGHT = os.getenv("ALLOW_STRONG_MATCH_HIGHLIGHT", "1").strip() == "1"
STRONG_MATCH_PROB = float(os.getenv("STRONG_MATCH_PROB", "0.90"))
STRONG_MATCH_DESC_OVERLAP = int(os.getenv("STRONG_MATCH_DESC_OVERLAP", "2"))

# G) aliases/synonyms
ENABLE_ALIAS_SYN = os.getenv("ENABLE_ALIAS_SYN", "1").strip() == "1"

MAX_DEBUG_CHUNKS = int(os.getenv("MAX_DEBUG_CHUNKS", "0"))

# =============================================================================
# Stopwords (English only, minimal). Keep modals and requirement verbs.
# =============================================================================
STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "on", "for", "with", "a", "an", "as", "at", "by",
    "from", "into", "over", "under", "than", "then", "that", "this", "these", "those",
    "be", "is", "are", "was", "were", "been", "being",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "can", "could", "may", "might", "will", "would",
    "not", "no", "yes",
}

# =============================================================================
# Requirement heuristics (EN) — broadened to avoid "0 highlights"
# =============================================================================
REQ_MODAL_RE = re.compile(
    r"\b(shall|must|required|requirement|should|need to|has to|may not|must not|shall not|prohibit|forbidden|recommended|recommend)\b",
    re.IGNORECASE,
)

# Imperative/spec verbs often used as requirements without shall/must:
REQ_IMPERATIVE_RE = re.compile(
    r"^\s*(use|provide|ensure|verify|test|install|maintain|replace|protect|equip|include|fit|apply|set|adjust|calibrate)\b",
    re.IGNORECASE,
)

REQ_CONDITION_RE = re.compile(r"\b(if|when|unless|in case)\b", re.IGNORECASE)
REQ_NUM_RE = re.compile(r"\b\d+([.,]\d+)?\b")
UNIT_RE = re.compile(r"\b(mm|cm|m|kg|w|kw|mw|v|kv|a|ma|hz|rpm|bar|pa|degc|°c|c|ip\d{2}\b)", re.IGNORECASE)

# =============================================================================
# TOC / page artifacts
# =============================================================================
TOC_DOTS_RE = re.compile(r"^\s*(\d+(\.\d+)*\s+)?[A-Za-z][A-Za-z0-9 \-_/(),]{2,}\s*\.{5,}\s*\d+\s*$")
TOC_SECTION_PAGE_RE = re.compile(r"^\s*\d+(\.\d+)+\s+[A-Za-z].{2,}\s+\d+\s*$")

PAGE_ONLY_RE = re.compile(r"^\s*\d+\s*$")
PAGE_DASH_RE = re.compile(r"^\s*[-–—]+\s*\d+\s*[-–—]+\s*$")
PAGE_WORD_RE = re.compile(r"^\s*(page)\s*\d+\s*$", re.IGNORECASE)
PAGE_FRACTION_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")

# =============================================================================
# Clients
# =============================================================================
client = None
CHAT_DEPLOYMENT = None

# =============================================================================
# Demand store
# =============================================================================
DEMANDS: List[Dict[str, str]] = []
DEMAND_BY_ID: Dict[str, Dict[str, Any]] = {}

# Split demand (A) while preserving demand_id
DEMAND_CATEGORY: Dict[str, str] = {}   # id -> motor/pump/unknown
DEMAND_SUBLABEL: Dict[str, str] = {}   # id -> earthing, spare parts, etc.
DEMAND_UI_LABEL: Dict[str, str] = {}   # id -> original demand string

DEMAND_NAME_RAW: Dict[str, str] = {}
DEMAND_DESC_RAW: Dict[str, str] = {}
DEMAND_ALIAS_LIST: Dict[str, List[str]] = {}

DEMAND_NAME_TOKS: Dict[str, set] = {}
DEMAND_DESC_TOKS: Dict[str, set] = {}
DEMAND_ALL_TOKS: Dict[str, set] = {}

# D) checklist constraints
DEMAND_CHECKLIST: Dict[str, List[str]] = {}

# B) embeddings
DEMAND_EMB: Dict[str, List[float]] = {}
DEMAND_CAT_EMB: Dict[str, List[float]] = {}

# =============================================================================
# G) deterministic synonym map (small)
# =============================================================================
SYN_MAP = {
    "earthing": ["grounding", "ground", "protective earth", "pe"],
    "grounding": ["earthing", "protective earth", "pe"],
    "voltage": ["volt", "kv", "vdc", "vac"],
    "coating": ["paint", "plating", "lining", "corrosion protection"],
    "insulation": ["dielectric", "isolation"],
    "test": ["testing", "verification", "acceptance test", "qualification"],
    "spare parts": ["spares", "replacement parts", "spare part"],
    "lubrication": ["grease", "oil", "lubricant"],
}

# =============================================================================
# Helpers
# =============================================================================
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
                row = {
                    "id": did,
                    "demand": str(r["demand"]).strip(),
                    "description": str(r["description"]).strip(),
                }
                if "aliases" in df.columns:
                    row["aliases"] = str(r.get("aliases", "") or "").strip()
                out.append(row)

            logger.info("Loaded %d demands from %s", len(out), p)
            return out

    raise RuntimeError(f"demands.xlsx not found. Tried: {candidates}")


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = []
    for t in s.split():
        if len(t) < 3:
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
    if TOC_DOTS_RE.match(t):
        return True
    if TOC_SECTION_PAGE_RE.match(t):
        return True
    if re.search(r"\.{5,}", t) and re.search(r"\d+\s*$", t):
        return True
    return False


def _is_page_artifact(s: str) -> bool:
    if not REJECT_PAGE_ARTIFACTS:
        return False
    t = (s or "").strip()
    if not t:
        return True
    if PAGE_ONLY_RE.match(t):
        return True
    if PAGE_DASH_RE.match(t):
        return True
    if PAGE_WORD_RE.match(t):
        return True
    if PAGE_FRACTION_RE.match(t):
        return True
    return False


def _should_reject_sentence(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True
    if _word_count(t) < MIN_WORDS_PER_SENTENCE:
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
    """
    More permissive heuristic to avoid "0 highlights":
    - modal words
    - imperative verbs at sentence start
    - numbers/units
    - conditional requirement framing
    """
    t = (text or "").strip()
    if len(t) < MIN_CHUNK_LEN:
        return False, 0.0

    has_modal = bool(REQ_MODAL_RE.search(t))
    has_imp = bool(REQ_IMPERATIVE_RE.search(t))
    has_num = bool(REQ_NUM_RE.search(t))
    has_unit = bool(UNIT_RE.search(t))
    has_cond = bool(REQ_CONDITION_RE.search(t))

    # Strongest signals
    if (has_modal or has_imp) and (has_num or has_unit):
        return True, 0.86
    if has_modal or has_imp:
        return True, 0.72
    if has_num and has_unit:
        return True, 0.66
    if has_cond and (has_num or has_unit):
        return True, 0.62

    return False, 0.25


# =============================================================================
# A) Parse hierarchy "motor - earthing" while preserving demand_id
# =============================================================================
def _parse_category_sublabel(demand_str: str) -> Tuple[str, str]:
    s = (demand_str or "").strip()
    if not s:
        return "unknown", ""
    parts = [p.strip() for p in re.split(r"\s*-\s*", s, maxsplit=1)]
    if len(parts) == 2:
        cat = parts[0].lower()
        sub = parts[1].lower()
        if cat.startswith("motor"):
            return "motor", sub
        if cat.startswith("pump"):
            return "pump", sub
        return "unknown", sub
    low = s.lower()
    if low.startswith("motor"):
        return "motor", low.replace("motor", "", 1).strip(" -")
    if low.startswith("pump"):
        return "pump", low.replace("pump", "", 1).strip(" -")
    return "unknown", low


# =============================================================================
# G) Aliases
# =============================================================================
def _parse_aliases_cell(cell: str) -> List[str]:
    if not cell:
        return []
    items = [x.strip().lower() for x in re.split(r"[;,]+", cell) if x and x.strip()]
    return [x for x in items if len(x) >= 3]


def _augment_aliases_from_synonyms(base: str) -> List[str]:
    if not ENABLE_ALIAS_SYN:
        return []
    out: List[str] = []
    b = (base or "").strip().lower()
    if not b:
        return out
    if b in SYN_MAP:
        out += SYN_MAP[b]
    # multiword phrase
    if " " in b and b in SYN_MAP:
        out += SYN_MAP[b]
    toks = re.split(r"\s+", b)
    if toks:
        last = toks[-1]
        if last in SYN_MAP:
            out += SYN_MAP[last]
    # de-dup
    seen = set()
    cleaned = []
    for x in out:
        x = (x or "").strip().lower()
        if not x or x in seen:
            continue
        seen.add(x)
        cleaned.append(x)
    return cleaned


# =============================================================================
# D) Checklist from description (deterministic)
# =============================================================================
def _extract_checklist(description: str, demand_name: str, aliases: List[str]) -> List[str]:
    desc = (description or "").lower()
    toks = [t for t in _tokenize(desc)]
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1

    bigrams: Dict[str, int] = {}
    if CHECKLIST_NGRAM_MAX >= 2:
        for i in range(len(toks) - 1):
            bg = toks[i] + " " + toks[i + 1]
            bigrams[bg] = bigrams.get(bg, 0) + 1

    # reduce "topic-only": downweight words from demand name
    name_toks = set(_tokenize((demand_name or "").lower()))
    for nt in name_toks:
        if nt in freq:
            freq[nt] = max(1, freq[nt] - 1)

    items: List[Tuple[float, str]] = []
    for bg, c in bigrams.items():
        items.append((2.0 * c, bg))
    for t, c in freq.items():
        items.append((1.0 * c, t))

    # prioritize distinctive alias phrases
    for ap in [a for a in aliases if " " in a and len(a) >= 6]:
        items.insert(0, (999.0, ap))

    items.sort(key=lambda x: x[0], reverse=True)

    out: List[str] = []
    seen = set()
    for _, it in items:
        it = it.strip().lower()
        if not it or it in seen:
            continue
        if " " not in it and len(it) < 4:
            continue
        seen.add(it)
        out.append(it)
        if len(out) >= CHECKLIST_SIZE:
            break
    return out


def _checklist_hits(sentence: str, checklist: List[str]) -> int:
    s = (sentence or "").lower()
    hits = 0
    for item in checklist or []:
        if item and item in s:
            hits += 1
    return hits


# =============================================================================
# B) Embeddings
# =============================================================================
def _normalize(vec: List[float]) -> List[float]:
    if not vec:
        return vec
    ss = 0.0
    for x in vec:
        ss += float(x) * float(x)
    n = math.sqrt(ss) if ss > 0 else 1.0
    return [float(x) / n for x in vec]


def _cos_sim(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    s = 0.0
    for x, y in zip(a, b):
        s += float(x) * float(y)
    return float(s)


def _emb_batch(texts: List[str]) -> List[List[float]]:
    if client is None or not EMBEDDINGS_DEPLOYMENT:
        return [[] for _ in texts]
    resp = client.embeddings.create(model=EMBEDDINGS_DEPLOYMENT, input=texts)
    out: List[List[float]] = []
    for item in resp.data:
        out.append(item.embedding)
    return out


def _embed_sentence(sentence: str) -> List[float]:
    if not USE_EMBEDDINGS or client is None or not EMBEDDINGS_DEPLOYMENT:
        return []
    vecs = _emb_batch([sentence])
    v = vecs[0] if vecs else []
    return _normalize(v)


def _build_embeddings():
    global DEMAND_EMB, DEMAND_CAT_EMB

    DEMAND_EMB = {}
    DEMAND_CAT_EMB = {}

    if not USE_EMBEDDINGS:
        return
    if client is None or not EMBEDDINGS_DEPLOYMENT:
        logger.warning("USE_EMBEDDINGS=1 but embeddings deployment/client missing -> embeddings disabled")
        return

    ids = list(DEMAND_BY_ID.keys())
    texts: List[str] = []

    for did in ids:
        cat = DEMAND_CATEGORY.get(did, "unknown")
        sub = DEMAND_SUBLABEL.get(did, "")
        name = DEMAND_UI_LABEL.get(did, "")
        desc = DEMAND_DESC_RAW.get(did, "")
        aliases = DEMAND_ALIAS_LIST.get(did, [])
        checklist = DEMAND_CHECKLIST.get(did, [])

        rep = f"category: {cat}\nlabel: {name}\nsublabel: {sub}\ndescription: {desc}"
        if aliases:
            rep += "\naliases: " + ", ".join(aliases[:12])
        if checklist:
            rep += "\nconstraints: " + "; ".join(checklist[:CHECKLIST_SIZE])
        texts.append(rep)

    for i in range(0, len(texts), EMBEDDINGS_BATCH):
        batch_texts = texts[i:i + EMBEDDINGS_BATCH]
        batch_ids = ids[i:i + EMBEDDINGS_BATCH]
        vecs = _emb_batch(batch_texts)
        for did, v in zip(batch_ids, vecs):
            DEMAND_EMB[did] = _normalize(v)

    # category prototypes
    cat_to_vecs: Dict[str, List[List[float]]] = {"motor": [], "pump": []}
    for did, v in DEMAND_EMB.items():
        cat = DEMAND_CATEGORY.get(did, "unknown")
        if cat in cat_to_vecs and v:
            cat_to_vecs[cat].append(v)

    for cat, vec_list in cat_to_vecs.items():
        if not vec_list:
            continue
        dim = len(vec_list[0])
        avg = [0.0] * dim
        for v in vec_list:
            for j, x in enumerate(v):
                avg[j] += float(x)
        avg = [x / float(len(vec_list)) for x in avg]
        DEMAND_CAT_EMB[cat] = _normalize(avg)

    logger.info("Embeddings built: demands=%d category_prototypes=%s", len(DEMAND_EMB), list(DEMAND_CAT_EMB.keys()))


# =============================================================================
# A) Category prediction
# =============================================================================
def _predict_category(sentence: str) -> Tuple[str, float]:
    sl = (sentence or "").lower()
    if "motor" in sl and "pump" not in sl:
        return "motor", 1.0
    if "pump" in sl and "motor" not in sl:
        return "pump", 1.0

    if USE_EMBEDDINGS and DEMAND_CAT_EMB:
        svec = _embed_sentence(sentence)
        if not svec:
            return "unknown", 0.0
        sm = max(0.0, _cos_sim(svec, DEMAND_CAT_EMB.get("motor", [])) if DEMAND_CAT_EMB.get("motor") else 0.0)
        sp = max(0.0, _cos_sim(svec, DEMAND_CAT_EMB.get("pump", [])) if DEMAND_CAT_EMB.get("pump") else 0.0)
        if sm == 0.0 and sp == 0.0:
            return "unknown", 0.0
        if sm >= sp:
            return "motor", float(sm - sp)
        return "pump", float(sp - sm)

    return "unknown", 0.0


# =============================================================================
# Candidate retrieval (hybrid)
# =============================================================================
def _candidate_retrieval(sentence: str, k: int, category: str) -> List[str]:
    stoks = set(_tokenize(sentence))
    if not stoks:
        return []

    if category in ("motor", "pump"):
        pool = [did for did in DEMAND_BY_ID.keys() if DEMAND_CATEGORY.get(did) == category]
    else:
        pool = list(DEMAND_BY_ID.keys())

    s_lower = (sentence or "").lower()
    svec = _embed_sentence(sentence) if (USE_EMBEDDINGS and DEMAND_EMB) else []

    scored: List[Tuple[str, float]] = []
    for did in pool:
        all_toks = DEMAND_ALL_TOKS.get(did, set())
        overlap = len(stoks & all_toks) if all_toks else 0

        phrase_bonus = 0.0
        # sublabel multiword anchor
        sub = DEMAND_SUBLABEL.get(did, "")
        if sub and " " in sub and sub in s_lower:
            phrase_bonus += 2.0
        # alias phrase anchor
        for a in DEMAND_ALIAS_LIST.get(did, []):
            if " " in a and len(a) >= 6 and a in s_lower:
                phrase_bonus += 1.5
                break

        sim = 0.0
        if svec and did in DEMAND_EMB and DEMAND_EMB[did]:
            sim = max(0.0, _cos_sim(svec, DEMAND_EMB[did]))

        # keep candidates with *some* evidence (lex OR phrase OR decent sim)
        if overlap >= MIN_CANDIDATE_TOKEN_OVERLAP or phrase_bonus > 0.0 or sim > 0.30:
            score = float(overlap) + phrase_bonus + (EMBEDDING_WEIGHT * float(sim))
            scored.append((did, score))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    return [did for did, _ in scored[: max(1, k)]]


# =============================================================================
# Scoring and probability (F)
# =============================================================================
def _score_label(sentence: str, did: str) -> Tuple[float, int, int, int]:
    stoks = set(_tokenize(sentence))
    if not stoks:
        return 0.0, 0, 0, 0

    name_toks = DEMAND_NAME_TOKS.get(did, set())
    desc_toks = DEMAND_DESC_TOKS.get(did, set())

    oname = len(stoks & name_toks) if name_toks else 0
    odesc = len(stoks & desc_toks) if desc_toks else 0
    ototal = oname + odesc

    # description-first weighting
    score = (2.0 * float(oname)) + (3.2 * float(odesc))
    return float(score), int(ototal), int(oname), int(odesc)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _prob_from_scores(score: float, best: float, second: float) -> float:
    if best <= 0:
        return 0.0
    ratio = max(0.0, min(2.0, score / best))
    base = _sigmoid(SIG_A * (ratio - SIG_B))
    margin = max(0.0, best - second)
    m = _sigmoid(MARGIN_A * (margin - MARGIN_B))
    p = base * m
    p = min(PROB_CAP, 0.55 + 0.45 * max(0.0, min(1.0, p)))
    return float(p)


def _passes_instance_check(sentence: str, did: str, odesc: int, is_req: bool, rel: float) -> bool:
    """
    D) Instance-check: require checklist hits OR strong requirement + desc overlap.
    Keep recall-safe: if no checklist exists, do not block.
    """
    if not CHECKLIST_ENFORCE:
        return True
    checklist = DEMAND_CHECKLIST.get(did, [])
    if not checklist:
        return True

    hits = _checklist_hits(sentence, checklist)
    if hits >= CHECKLIST_MIN_HITS:
        return True

    # fallback: if sentence looks like a requirement and desc overlap exists
    if is_req and rel >= 0.65 and odesc >= max(1, MIN_DESC_OVERLAP_FOR_LABEL):
        return True

    return False


# =============================================================================
# E) Single-call LLM verify + explain for top candidates
# =============================================================================
def _llm_verify_explain(sentence: str, cand_ids: List[str]) -> Tuple[List[str], Dict[str, str]]:
    if not USE_LLM_VERIFY_EXPLAIN:
        return cand_ids, {}
    if client is None or not CHAT_DEPLOYMENT:
        # Do not hard-fail to zero if LLM not available
        return cand_ids, {}

    cands_block = []
    for did in cand_ids[:LLM_VERIFY_MAX_CANDS]:
        ui = DEMAND_UI_LABEL.get(did, "")
        cat = DEMAND_CATEGORY.get(did, "unknown")
        sub = DEMAND_SUBLABEL.get(did, "")
        desc = DEMAND_DESC_RAW.get(did, "")
        desc_short = desc if len(desc) <= 450 else (desc[:450] + "…")
        chk = "; ".join((DEMAND_CHECKLIST.get(did, []) or [])[:CHECKLIST_SIZE])
        cands_block.append(
            f"- id: {did}\n  label: {ui}\n  category: {cat}\n  sublabel: {sub}\n  description: {desc_short}\n  constraints: {chk}"
        )
    cands_block = "\n".join(cands_block)

    system = "Return STRICT JSON only. No markdown. No extra keys."
    user = f"""
You verify requirement-to-label matches.
You MUST use ONLY the provided candidates and the sentence. Do NOT introduce new ids.

SENTENCE:
{sentence}

CANDIDATES:
{cands_block}

Return JSON exactly:
{{
  "results": [
    {{
      "id": "<candidate_id>",
      "verdict": "PASS" | "FAIL",
      "explanation": "1-2 sentences maximum; must reference description/constraints, not keywords alone."
    }}
  ]
}}

Rules:
- PASS only if the sentence matches the DESCRIPTION/constraints (not just topic mention).
- If sentence looks like a heading/TOC/page artifact, verdict MUST be FAIL.
- If uncertain, FAIL.
""".strip()

    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=LLM_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        results = data.get("results", [])
        pass_set = set()
        expl: Dict[str, str] = {}
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                did = str(item.get("id", "")).strip()
                if did not in cand_ids:
                    continue
                verdict = str(item.get("verdict", "")).strip().upper()
                explanation = str(item.get("explanation", "") or "").strip()
                if explanation:
                    parts = re.split(r"(?<=[.!?])\s+", explanation)
                    explanation = " ".join(parts[:2]).strip()
                if verdict == "PASS":
                    pass_set.add(did)
                    if explanation:
                        expl[did] = explanation

        pass_ids = [did for did in cand_ids if did in pass_set]
        return pass_ids, expl
    except Exception:
        # If malformed JSON, don't zero everything; just skip LLM filtering
        return cand_ids, {}


# =============================================================================
# Optional LLM requirement gate
# =============================================================================
def _llm_gate(sentence: str) -> Tuple[bool, float]:
    if client is None or not CHAT_DEPLOYMENT:
        return _is_requirement_fast(sentence)

    system = "Return STRICT JSON only."
    user = f"""
Decide whether the SENTENCE contains a concrete recommendation/requirement to highlight.
Return JSON only:
{{
  "isRequirement": true|false,
  "relevance": 0.0
}}
SENTENCE:
{sentence}

Guidance:
- signals: must/shall/should/recommended/need to/has to, constraints, tests/acceptance, prohibitions, explicit conditions.
- relevance 0..1. If NOT requirement, keep it < {RELEVANCE_MIN}.
""".strip()

    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=180,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        return bool(data.get("isRequirement", False)), float(data.get("relevance", 0.0) or 0.0)
    except Exception:
        return _is_requirement_fast(sentence)


# =============================================================================
# Main per-sentence prediction
# =============================================================================
def _predict_sentence(sentence: str, top_k: int) -> Dict[str, Any]:
    if _should_reject_sentence(sentence):
        return {"isRequirement": False, "relevance": 0.0, "labels": [], "explanations": {}, "notes": "rejected_artifact"}

    is_req, rel = _is_requirement_fast(sentence)
    if USE_LLM_GATE:
        is_req, rel = _llm_gate(sentence)

    cat, margin = _predict_category(sentence)
    if margin < CATEGORY_CONF_MIN:
        cat = "unknown"

    cand_ids = _candidate_retrieval(sentence, k=max(CANDIDATES_TOPK, top_k), category=cat)
    if not cand_ids:
        return {"isRequirement": bool(is_req), "relevance": float(rel), "labels": [], "explanations": {}, "notes": "no_candidates"}

    svec = _embed_sentence(sentence) if (USE_EMBEDDINGS and DEMAND_EMB) else []

    scored: List[Tuple[str, float, int, int, int]] = []
    for did in cand_ids[:max(CANDIDATES_TOPK, top_k)]:
        lex_score, ototal, oname, odesc = _score_label(sentence, did)

        sim = 0.0
        if svec and did in DEMAND_EMB and DEMAND_EMB[did]:
            sim = max(0.0, _cos_sim(svec, DEMAND_EMB[did]))

        score = float(lex_score) + (EMBEDDING_WEIGHT * float(sim))
        scored.append((did, score, ototal, oname, odesc))

    scored.sort(key=lambda x: x[1], reverse=True)

    # deterministic guards + instance check
    filtered: List[Tuple[str, float, int, int, int]] = []
    for did, sc, ototal, oname, odesc in scored:
        if ZERO_LEXICAL_REJECT and ototal == 0:
            continue
        if ototal < MIN_LABEL_TOKEN_OVERLAP:
            continue
        if odesc < MIN_DESC_OVERLAP_FOR_LABEL:
            continue
        if not _passes_instance_check(sentence, did, odesc=odesc, is_req=bool(is_req), rel=float(rel)):
            continue
        filtered.append((did, sc, ototal, oname, odesc))
        if len(filtered) >= max(1, top_k * 3):
            break

    if not filtered:
        return {"isRequirement": bool(is_req), "relevance": float(rel), "labels": [], "explanations": {}, "notes": "no_labels_after_checks"}

    best = filtered[0][1]
    second = filtered[1][1] if len(filtered) > 1 else 0.0

    pairs: List[Tuple[str, float, int]] = []
    for did, sc, _, _, odesc in filtered[:max(1, top_k)]:
        p = _prob_from_scores(sc, best=best, second=second)
        pairs.append((did, p, odesc))

    pairs.sort(key=lambda x: x[1], reverse=True)

    # "Do not force" selection
    top_prob = pairs[0][1]
    kept: List[Tuple[str, float, int]] = []
    for did, p, odesc in pairs:
        if p < KEEP_ABSOLUTE_FLOOR:
            continue
        if p < top_prob * KEEP_RELATIVE_TO_TOP:
            continue
        if p < LABEL_MIN_PROB:
            continue
        kept.append((did, p, odesc))
        if len(kept) >= MAX_LABELS_PER_CHUNK:
            break

    if not kept:
        return {"isRequirement": bool(is_req), "relevance": float(rel), "labels": [], "explanations": {}, "notes": "thresholded_out"}

    kept_ids = [did for did, _, _ in kept][:LLM_VERIFY_MAX_CANDS]
    explanations: Dict[str, str] = {}

    if USE_LLM_VERIFY_EXPLAIN:
        pass_ids, expl = _llm_verify_explain(sentence, kept_ids)
        pass_set = set(pass_ids)
        explanations = expl or {}
        kept = [(did, p, odesc) for did, p, odesc in kept if did in pass_set]
        if not kept:
            return {"isRequirement": bool(is_req), "relevance": float(rel), "labels": [], "explanations": {}, "notes": "llm_rejected_all"}

    out_labels = [{"id": did, "probability": float(p)} for did, p, _ in kept[:MAX_LABELS_PER_CHUNK]]
    rel_out = max(float(rel), float(out_labels[0]["probability"])) if out_labels else float(rel)

    return {"isRequirement": bool(is_req), "relevance": float(rel_out), "labels": out_labels, "explanations": explanations, "notes": "ok"}


# =============================================================================
# Build demand indexes (including split demand + aliases + checklist)
# =============================================================================
def _build_demand_indexes():
    global DEMAND_BY_ID, DEMAND_CATEGORY, DEMAND_SUBLABEL, DEMAND_UI_LABEL
    global DEMAND_NAME_RAW, DEMAND_DESC_RAW, DEMAND_ALIAS_LIST
    global DEMAND_NAME_TOKS, DEMAND_DESC_TOKS, DEMAND_ALL_TOKS
    global DEMAND_CHECKLIST

    DEMAND_BY_ID = {}
    DEMAND_CATEGORY = {}
    DEMAND_SUBLABEL = {}
    DEMAND_UI_LABEL = {}

    DEMAND_NAME_RAW = {}
    DEMAND_DESC_RAW = {}
    DEMAND_ALIAS_LIST = {}

    DEMAND_NAME_TOKS = {}
    DEMAND_DESC_TOKS = {}
    DEMAND_ALL_TOKS = {}

    DEMAND_CHECKLIST = {}

    for d in DEMANDS:
        did = d["id"]
        demand = (d.get("demand", "") or "").strip()
        desc = (d.get("description", "") or "").strip()

        DEMAND_BY_ID[did] = d
        DEMAND_UI_LABEL[did] = demand

        cat, sub = _parse_category_sublabel(demand)
        DEMAND_CATEGORY[did] = cat
        DEMAND_SUBLABEL[did] = sub

        DEMAND_NAME_RAW[did] = demand.lower()
        DEMAND_DESC_RAW[did] = desc.lower()

        aliases = _parse_aliases_cell(str(d.get("aliases", "") or ""))

        if ENABLE_ALIAS_SYN:
            if sub:
                aliases += _augment_aliases_from_synonyms(sub)
            # if demand has " - something"
            if " - " in demand.lower():
                aliases += _augment_aliases_from_synonyms(demand.lower().split(" - ", 1)[1].strip())

        # de-dup aliases
        seen = set()
        clean_aliases = []
        for a in aliases:
            a = (a or "").strip().lower()
            if not a or a in seen:
                continue
            seen.add(a)
            clean_aliases.append(a)
        DEMAND_ALIAS_LIST[did] = clean_aliases

        name_toks = _tokset(demand)
        desc_toks = _tokset(desc)
        alias_toks = set()
        for a in clean_aliases:
            alias_toks |= _tokset(a)

        DEMAND_NAME_TOKS[did] = name_toks
        DEMAND_DESC_TOKS[did] = desc_toks
        DEMAND_ALL_TOKS[did] = set(name_toks | desc_toks | alias_toks)

        DEMAND_CHECKLIST[did] = _extract_checklist(desc, demand, clean_aliases)


# =============================================================================
# AML entrypoints
# =============================================================================
def init():
    global client, CHAT_DEPLOYMENT, DEMANDS, USE_MOCK, USE_EMBEDDINGS

    USE_MOCK = os.getenv("USE_MOCK", "0").strip() == "1"
    DEMANDS = _load_demands_xlsx()
    _build_demand_indexes()

    need_client = (USE_LLM_VERIFY_EXPLAIN or USE_LLM_GATE or USE_EMBEDDINGS) and not USE_MOCK
    if need_client:
        if AzureOpenAI is None:
            raise RuntimeError("openai package not available but LLM/embeddings usage enabled")

        aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        aoai_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        aoai_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip() or "2024-06-01"

        CHAT_DEPLOYMENT = (
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
            or os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        )

        if not (aoai_endpoint and aoai_key):
            raise RuntimeError("Missing AOAI env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY")

        client = AzureOpenAI(
            azure_endpoint=aoai_endpoint,
            api_key=aoai_key,
            api_version=aoai_version,
        )

        if (USE_LLM_VERIFY_EXPLAIN or USE_LLM_GATE) and not CHAT_DEPLOYMENT:
            raise RuntimeError("Missing AOAI chat deployment. Set AZURE_OPENAI_CHAT_DEPLOYMENT (or AZURE_OPENAI_DEPLOYMENT).")

        if USE_EMBEDDINGS and not EMBEDDINGS_DEPLOYMENT:
            logger.warning("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT not set; embeddings disabled")
            USE_EMBEDDINGS = False

    _build_embeddings()

    logger.info(
        "init(): USE_MOCK=%s demands=%d USE_EMBEDDINGS=%s USE_LLM_VERIFY_EXPLAIN=%s USE_LLM_GATE=%s "
        "CHECKLIST_ENFORCE=%s REJECT_TOC=%s REJECT_PAGE_ARTIFACTS=%s",
        USE_MOCK, len(DEMANDS), str(USE_EMBEDDINGS), str(USE_LLM_VERIFY_EXPLAIN), str(USE_LLM_GATE),
        str(CHECKLIST_ENFORCE), str(REJECT_TOC), str(REJECT_PAGE_ARTIFACTS)
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

            chunk_text = (str(content.get("text", "") or "")).strip()

            # Always set expected keys/types
            content["relevantProba"] = 0.0
            content["cdTransformerPredictions"] = []
            content["cdLogregPredictions"] = []
            content["highlightText"] = ""
            content["labelExplanations"] = []

            if _word_count(chunk_text) <= 3:
                continue

            if debug_left > 0:
                logger.info("chunk=%s sample=%r", cid, chunk_text[:180])
                debug_left -= 1

            sentences_raw = _split_into_sentences_or_lines(chunk_text)
            sentences = [s for s in sentences_raw if not _should_reject_sentence(s)]
            if not sentences:
                continue

            # Choose best sentence using a robust metric (prevents selecting a non-requirement sentence)
            best_sentence = ""
            best_pred: Optional[Dict[str, Any]] = None
            best_metric = -1.0

            # Pre-rank sentences to limit cost
            scored_sents: List[Tuple[float, str]] = []
            for s in sentences:
                is_req, rel = _is_requirement_fast(s)
                cat, margin = _predict_category(s)
                if margin < CATEGORY_CONF_MIN:
                    cat = "unknown"
                cands = _candidate_retrieval(s, k=6, category=cat)
                # More weight on candidate signal to avoid missing imperative requirements
                score = (0.50 * rel) + (0.10 * float(len(cands))) + (0.05 if is_req else 0.0)
                scored_sents.append((score, s))

            scored_sents.sort(key=lambda x: x[0], reverse=True)
            candidates_for_model = [s for _, s in scored_sents[:MAX_SENTENCES_TO_MODEL] if s.strip()]

            for s in candidates_for_model:
                pred = _predict_sentence(s, top_k=num_preds)
                labels = pred.get("labels", []) or []
                if not isinstance(labels, list) or not labels:
                    continue

                top1 = float(labels[0].get("probability", 0.0) or 0.0)
                is_req = bool(pred.get("isRequirement", False))
                rel = float(pred.get("relevance", 0.0) or 0.0)

                # metric favors requirement, but still allows "strong match" sentences
                metric = top1
                metric *= (1.0 + 0.25 * rel)
                if is_req:
                    metric *= 1.15

                if metric > best_metric:
                    best_metric = metric
                    best_pred = pred
                    best_sentence = s

            if not best_pred:
                continue

            labels = best_pred.get("labels", []) or []
            labels_sorted = sorted(labels, key=lambda x: float(x.get("probability", 0.0) or 0.0), reverse=True)
            labels_sorted = labels_sorted[:num_preds]
            final_labels = [{"label": str(lb["id"]), "proba": float(lb["probability"])} for lb in labels_sorted]

            top1 = float(final_labels[0]["proba"]) if final_labels else 0.0
            is_req = bool(best_pred.get("isRequirement", False))
            rel = float(best_pred.get("relevance", 0.0) or 0.0)

            # Strong match fallback to avoid 0 highlights when modals are missing
            strong_match = False
            if ALLOW_STRONG_MATCH_HIGHLIGHT and final_labels:
                did_top = str(final_labels[0]["label"])
                # estimate desc overlap quickly for top label
                _, _, _, odesc = _score_label(best_sentence, did_top)
                strong_match = (top1 >= STRONG_MATCH_PROB and odesc >= STRONG_MATCH_DESC_OVERLAP)

            should_highlight = bool((is_req and top1 >= HIGHLIGHT_MIN_PROB) or (strong_match and top1 >= HIGHLIGHT_MIN_PROB))

            if should_highlight and final_labels:
                content["relevantProba"] = float(max(rel, top1))
                content["cdTransformerPredictions"] = final_labels
                content["cdLogregPredictions"] = list(final_labels)
                content["highlightText"] = best_sentence

                expl_map = best_pred.get("explanations", {}) or {}
                if isinstance(expl_map, dict) and expl_map:
                    content["labelExplanations"] = [
                        {"label": did, "explanation": str(expl_map.get(did, ""))}
                        for did in [p["label"] for p in final_labels]
                        if str(expl_map.get(did, "")).strip()
                    ]

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

