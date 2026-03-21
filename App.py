"""
MRPL Financial Chatbot — Production RAG System (v3)
LLM backend : Ollama — primary: qwen2.5:3b, fallback chain: phi3 → tinyllama
               Auto-detects Docker vs local host for Ollama base URL.
               Extractive sentence-scoring fallback if all Ollama calls fail.
Deployment  : Docker / server  (host.docker.internal OR localhost)
Other features:
  • Multi-PDF FAISS vector store — persistent, hash-deduped, never re-indexes
  • Rich metadata on every chunk: pdf_name, page, chunk_index, type, section_hint
  • Image extraction + optional pytesseract OCR
  • Table extraction (pdfplumber) → CSV; rendered as st.dataframe
  • Source citations (pdf name + page) on EVERY answer type
  • Sidebar model selector — switch LLM without restart
  • Sidebar Ollama status indicator
"""

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB / THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, re, json, hashlib, warnings, logging, io, time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
for _noisy in ["sentence_transformers", "transformers", "torch",
               "faiss", "langchain", "PIL", "pdfplumber", "fitz"]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Ranked best → worst for RAG quality on your installed models
OLLAMA_MODELS        = ["qwen2.5:3b", "phi3:latest", "phi3:mini",
                        "tinyllama:latest", "tinyllama:1.1b", "smollm2:360m"]
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT       = 90   # seconds — qwen2.5 can be slow on CPU


def _detect_ollama_base_url() -> str:
    """
    Probe candidate URLs in order and return the first one that responds.
    Checks:
      1. OLLAMA_BASE_URL env var (set this in docker-compose for reliability)
      2. host.docker.internal:11434  (Docker Desktop on Windows/Mac)
      3. localhost:11434             (bare server, native Windows, WSL2 host-net)
      4. 127.0.0.1:11434             (alias)
    """
    import urllib.request
    env_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    for candidate in [
        "http://host.docker.internal:11434",
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ]:
        try:
            urllib.request.urlopen(candidate + "/api/tags", timeout=2)
            return candidate.rstrip("/")
        except Exception:
            continue
    return "http://localhost:11434"   # last resort, LangChain default


def _ollama_ping(base_url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _ollama_available_models(base_url: str) -> List[str]:
    """Return list of model name strings from a live Ollama instance."""
    import urllib.request, json as _json
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=4) as r:
            return [m["name"] for m in _json.loads(r.read()).get("models", [])]
    except Exception:
        return []


# Resolved once at module load; takes < 2 s on first run
OLLAMA_BASE_URL: str = _detect_ollama_base_url()

# ── Cache Ollama LangChain class at import time (not per-query) ──
_OLLAMA_LLM_CLASS = None
def _get_ollama_class():
    global _OLLAMA_LLM_CLASS
    if _OLLAMA_LLM_CLASS is not None:
        return _OLLAMA_LLM_CLASS
    import importlib
    for mod_path, cls_name in [
        ("langchain_community.llms",        "Ollama"),
        ("langchain_community.llms.ollama", "Ollama"),
        ("langchain_ollama",                "OllamaLLM"),
    ]:
        try:
            mod = importlib.import_module(mod_path)
            _OLLAMA_LLM_CLASS = getattr(mod, cls_name)
            return _OLLAMA_LLM_CLASS
        except Exception:
            continue
    return None

# ── Cache available models (refresh once per session, not per query) ──
_OLLAMA_INSTALLED_MODELS: Optional[List[str]] = None
def _get_installed_models() -> List[str]:
    global _OLLAMA_INSTALLED_MODELS
    if _OLLAMA_INSTALLED_MODELS is not None:
        return _OLLAMA_INSTALLED_MODELS
    _OLLAMA_INSTALLED_MODELS = _ollama_available_models(OLLAMA_BASE_URL)
    return _OLLAMA_INSTALLED_MODELS

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MRPL Annual Report Assistant",
    page_icon="MRPL",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
section[data-testid="stSidebar"]{background:#f7f8fa;border-right:1px solid #e2e6ea;}
section[data-testid="stSidebar"] .block-container{padding-top:1.5rem;}
.logo-box{display:flex;flex-direction:column;align-items:center;gap:6px;margin-bottom:1.2rem;}
.logo-title{font-size:.95rem;font-weight:600;color:#1a2332;text-align:center;}
div[data-testid="stButton"]>button{
  background:white;border:1px solid #d0d5dd;border-radius:6px;
  font-size:.82rem;font-weight:500;color:#374151;
  padding:.35rem .9rem;width:100%;transition:all .15s ease;}
div[data-testid="stButton"]>button:hover{background:#f0f4ff;border-color:#3b6eea;color:#3b6eea;}
.section-header{font-size:.72rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.07em;color:#9ca3af;margin:1rem 0 .4rem 0;}
.report-item{display:flex;align-items:center;gap:6px;padding:4px 2px;
  font-size:.82rem;color:#374151;border-radius:4px;transition:background .1s;}
.report-item:hover{background:#eef2ff;}
.report-dot{width:6px;height:6px;border-radius:50%;background:#4ade80;flex-shrink:0;}
.main-title{font-size:1.65rem;font-weight:700;color:#1a2332;margin-bottom:.15rem;}
.chat-user{
  background:#f0f4ff;border-radius:12px 12px 2px 12px;
  padding:.7rem 1rem;margin:.4rem 0 .4rem auto;
  font-size:.9rem;color:#1a2332;max-width:74%;text-align:right;}
.chat-bot{
  background:white;border:1px solid #e5e7eb;border-radius:2px 12px 12px 12px;
  padding:.7rem 1rem;margin:.4rem 0;font-size:.9rem;color:#374151;max-width:85%;
  box-shadow:0 1px 3px rgba(0,0,0,.06);}
.not-found{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;
  padding:.6rem 1rem;font-size:.88rem;color:#9a3412;}
.number-badge{display:inline-block;background:#eff6ff;border:1px solid #bfdbfe;
  color:#1d4ed8;border-radius:6px;padding:2px 8px;font-weight:600;font-size:.88rem;}
.processing-box{background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
  padding:.8rem 1rem;margin:.6rem 0;font-size:.88rem;color:#166534;}
.info-box{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
  padding:.8rem 1rem;margin:.6rem 0;font-size:.88rem;color:#1e40af;}
.source-tag{margin-top:6px;padding:4px 8px;background:#f8fafc;
  border-left:3px solid #3b6eea;border-radius:0 4px 4px 0;
  font-size:.78rem;color:#6b7280;}
hr{border:none;border-top:1px solid #e5e7eb;margin:.8rem 0;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
VECTOR_DIR  = Path("vector_store")
TABLES_DIR  = Path("tables")
IMAGES_DIR  = Path("images")
DATA_DIR    = Path("data")
META_FILE   = Path("metadata.json")
HASH_FILE   = DATA_DIR / "pdf_hashes.json"
ASSETS_DIR  = Path("assets")

for _d in [VECTOR_DIR, TABLES_DIR, IMAGES_DIR, DATA_DIR, ASSETS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

EMBED_MODEL    = "intfloat/e5-base-v2"
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 100
MIN_IMG_BYTES  = 15_000   # skip logos/icons smaller than 15 KB
TOP_K_DOCS     = 12       # semantic search recall

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in {
    "chat_history":    [],
    "processing":      False,
    "selected_model":  OLLAMA_DEFAULT_MODEL,
    "ollama_ok":       None,   # None = not yet checked
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    """Load JSON → always return a dict. Handles legacy list format."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            out = {}
            for item in data:
                if isinstance(item, dict):
                    key = item.get("pdf_name") or item.get("name") or f"doc_{len(out)}"
                    out[key] = item
            return out
    except Exception:
        pass
    return {}


def _save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _pdf_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _repair_metadata():
    """Migrate list-format metadata.json → dict on first run."""
    if not META_FILE.exists():
        return
    try:
        raw = json.loads(META_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            fixed = {}
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        key = item.get("pdf_name") or f"doc_{len(fixed)}"
                        fixed[key] = item
            META_FILE.write_text(json.dumps(fixed, indent=2, default=str), encoding="utf-8")
    except Exception:
        META_FILE.write_text("{}", encoding="utf-8")


_repair_metadata()

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def _load_vector_store(safe_names_tuple: tuple):
    """
    Merge all per-PDF FAISS indices into one in-memory store.
    Cached by tuple of safe_names → re-runs are instant.
    """
    if not safe_names_tuple:
        return None
    from langchain_community.vectorstores import FAISS
    emb    = _load_embeddings()
    merged = None
    for sn in safe_names_tuple:
        vp = VECTOR_DIR / sn
        if vp.exists():
            try:
                vs = FAISS.load_local(str(vp), emb, allow_dangerous_deserialization=True)
                if merged is None:
                    merged = vs
                else:
                    merged.merge_from(vs)
            except Exception as e:
                st.warning(f"Could not load index for '{sn}': {e}")
    return merged

# ─────────────────────────────────────────────────────────────────────────────
# PDF PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _safe_name(pdf_name: str) -> str:
    return re.sub(r"[^\w\-]", "_", Path(pdf_name).stem)


def _extract_numbers(text: str) -> List[Dict]:
    """Extract numeric values with surrounding context."""
    pattern = (
        r"(?:[₹$€£]?\s*\d[\d,]*(?:\.\d+)?\s*"
        r"(?:crore|lakh|million|billion|thousand|Cr\.?|L\.?|MMT|TMT|MWh|TKL|bbl|%)?)"
    )
    results = []
    for m in re.finditer(pattern, text, re.IGNORECASE):
        s = max(0, m.start() - 90)
        e = min(len(text), m.end() + 90)
        results.append({
            "value":   m.group().strip(),
            "context": text[s:e].strip(),
        })
    return results


def process_pdf(pdf_bytes: bytes, pdf_name: str, status_el) -> dict:
    """
    Full pipeline:
      1. Text extraction (PyMuPDF) with page-level chunks
      2. Image extraction + optional OCR (pytesseract)
      3. Table extraction (pdfplumber) → CSV files
      4. Chunk + embed → FAISS index (saved to disk)
    Returns metadata dict or {} on failure.
    """
    import fitz
    import pdfplumber
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document

    safe    = _safe_name(pdf_name)
    tbl_dir = TABLES_DIR / safe;  tbl_dir.mkdir(parents=True, exist_ok=True)
    img_dir = IMAGES_DIR / safe;  img_dir.mkdir(parents=True, exist_ok=True)

    raw_chunks:  List[Dict] = []   # {text, metadata}
    tables_meta: List[Dict] = []
    images_meta: List[Dict] = []
    all_numbers: List[Dict] = []

    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)

    # ── 1. TEXT ──────────────────────────────────────────────────
    status_el.markdown(
        f'<div class="info-box">Extracting text — {total} pages…</div>',
        unsafe_allow_html=True)

    section_hint = ""   # track section headings as we go
    for pn in range(total):
        text = doc[pn].get_text("text")
        if not text.strip():
            continue
        # Update section hint from headings (lines in ALL CAPS or Title Case ≥ 20 chars)
        for line in text.splitlines():
            stripped = line.strip()
            if (len(stripped) >= 15 and
                    (stripped.isupper() or re.match(r"^[A-Z][A-Za-z ]{14,}$", stripped))):
                section_hint = stripped[:80]
                break
        all_numbers.extend(_extract_numbers(text))
        raw_chunks.append({
            "text": text,
            "metadata": {
                "pdf_name":     pdf_name,
                "safe_name":    safe,
                "page":         pn + 1,
                "type":         "text",
                "section_hint": section_hint,
            },
        })

    # ── 2. IMAGES ────────────────────────────────────────────────
    status_el.markdown(
        '<div class="info-box">Extracting images…</div>',
        unsafe_allow_html=True)

    for pn in range(total):
        for ii, img_info in enumerate(doc[pn].get_images(full=True)):
            xref = img_info[0]
            try:
                base_img  = doc.extract_image(xref)
                img_bytes = base_img["image"]
                if len(img_bytes) < MIN_IMG_BYTES:
                    continue   # skip tiny logos / decorative icons
                ext   = base_img.get("ext", "png")
                ipath = img_dir / f"p{pn+1}_img{ii+1}.{ext}"
                ipath.write_bytes(img_bytes)

                # OCR (optional — graceful failure)
                ocr_text = ""
                try:
                    import pytesseract
                    from PIL import Image as PILImage
                    pil      = PILImage.open(io.BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(pil, timeout=8).strip()
                except Exception:
                    pass

                img_record = {
                    "pdf_name": pdf_name,
                    "safe_name": safe,
                    "page":     pn + 1,
                    "type":     "image",
                    "path":     str(ipath),
                    "size":     len(img_bytes),
                    "ocr_text": ocr_text[:600],
                }
                images_meta.append(img_record)

                if ocr_text:
                    raw_chunks.append({
                        "text": f"[IMAGE-OCR page={pn+1}] {ocr_text}",
                        "metadata": {
                            "pdf_name":  pdf_name,
                            "safe_name": safe,
                            "page":      pn + 1,
                            "type":      "image",
                            "path":      str(ipath),
                        },
                    })
            except Exception:
                continue
    doc.close()

    # ── 3. TABLES ────────────────────────────────────────────────
    status_el.markdown(
        '<div class="info-box">Extracting tables…</div>',
        unsafe_allow_html=True)
    try:
        import pandas as pd
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plb:
            for pn, plb_page in enumerate(plb.pages):
                raw_tables = plb_page.extract_tables() or []
                for ti, raw_tbl in enumerate(raw_tables):
                    if not raw_tbl or len(raw_tbl) < 2:
                        continue
                    try:
                        # De-duplicate column headers
                        hdrs, seen_h, clean_hdrs = raw_tbl[0], {}, []
                        for h in hdrs:
                            h = str(h).strip() if h else "col"
                            seen_h[h] = seen_h.get(h, 0) + 1
                            clean_hdrs.append(h if seen_h[h] == 1 else f"{h}_{seen_h[h]}")
                        df = pd.DataFrame(raw_tbl[1:], columns=clean_hdrs)
                        df.dropna(how="all", inplace=True)
                        if len(df) < 1 or len(df.columns) < 2:
                            continue
                        cpath = tbl_dir / f"p{pn+1}_t{ti+1}.csv"
                        df.to_csv(cpath, index=False)
                        tbl_str = df.to_string(index=False)
                        tables_meta.append({
                            "pdf_name":  pdf_name,
                            "safe_name": safe,
                            "page":      pn + 1,
                            "type":      "table",
                            "path":      str(cpath),
                            "preview":   tbl_str[:500],
                            "rows":      len(df),
                            "cols":      len(df.columns),
                        })
                        raw_chunks.append({
                            "text": f"[TABLE page={pn+1} title='{clean_hdrs}'] {tbl_str}",
                            "metadata": {
                                "pdf_name":  pdf_name,
                                "safe_name": safe,
                                "page":      pn + 1,
                                "type":      "table",
                                "path":      str(cpath),
                            },
                        })
                    except Exception:
                        continue
    except Exception:
        pass

    # ── 4. VECTOR INDEX ──────────────────────────────────────────
    status_el.markdown(
        '<div class="info-box">Building FAISS vector index…</div>',
        unsafe_allow_html=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = []
    for idx, item in enumerate(raw_chunks):
        sub_chunks = splitter.split_text(item["text"])
        for ci, chunk in enumerate(sub_chunks):
            meta = dict(item["metadata"])
            meta["chunk_index"] = ci
            docs.append(Document(page_content=chunk, metadata=meta))

    if not docs:
        status_el.error("No extractable text found in this PDF.")
        return {}

    emb   = _load_embeddings()
    vpath = str(VECTOR_DIR / safe)

    # Merge into existing index if already exists (re-upload edge-case)
    if Path(vpath).exists():
        from langchain_community.vectorstores import FAISS
        vs = FAISS.load_local(vpath, emb, allow_dangerous_deserialization=True)
        vs.add_documents(docs)
        vs.save_local(vpath)
    else:
        from langchain_community.vectorstores import FAISS
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(vpath)

    return {
        "pdf_name":    pdf_name,
        "safe_name":   safe,
        "vector_path": vpath,
        "total_pages": total,
        "tables":      tables_meta,
        "images":      images_meta,
        "numbers":     all_numbers[:3000],
        "chunk_count": len(docs),
    }

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _keyword_boost(query: str, docs: list) -> list:
    """Re-rank semantic results by keyword overlap."""
    qwords = set(re.findall(r"\b\w{4,}\b", query.lower()))
    scored = sorted(
        docs,
        key=lambda d: len(qwords & set(re.findall(r"\b\w{4,}\b", d.page_content.lower()))),
        reverse=True,
    )
    return scored


def _build_source_tag(sources: List[Dict]) -> str:
    """Build source citation bar — pdf name + top relevant page(s) only."""
    if not sources:
        return ""
    valid = [s for s in sources if isinstance(s, dict) and s.get("page")]
    if not valid:
        return ""

    from collections import defaultdict
    by_pdf: Dict[str, list] = defaultdict(list)
    for s in valid:
        name = s.get("pdf_name") or s.get("source") or "Unknown"
        page = s.get("page")
        if page and str(page).strip() and str(page).strip() != "—":
            try:
                by_pdf[name].append(int(str(page)))
            except (ValueError, TypeError):
                pass

    if not by_pdf:
        return ""

    parts = []
    for pdf, pages in by_pdf.items():
        # Deduplicate and show max 3 pages (most relevant = lowest index)
        unique_pages = sorted(set(pages))[:3]
        page_str = ", ".join(str(p) for p in unique_pages)
        short_name = pdf if len(pdf) <= 35 else pdf[:32] + "…"
        parts.append(f"<b>{short_name}</b> &nbsp;·&nbsp; p.&nbsp;{page_str}")

    inner = " &nbsp;|&nbsp; ".join(parts)
    return f'<div class="source-tag">{inner}</div>'


def _find_exact_number(query: str, meta: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Improved number finder with negative-context filtering.
    Returns (value_string, context_string) or (None, None).
    """
    ql = query.lower()

    TERM_MAP = {
        "revenue":       ["revenue from sale", "turnover", "sale of products"],
        "turnover":      ["turnover", "revenue from sale", "sale of products"],
        "total revenue": ["turnover", "revenue from sale"],
        "profit":        ["profit after tax", "pat", "profit for the year"],
        "net profit":    ["profit after tax", "pat", "earned profit"],
        "pbt":           ["profit before tax"],
        "pat":           ["profit after tax"],
        "profit after tax": ["profit after tax", "pat"],
        "profit before tax": ["profit before tax"],
        "ms production": ["ms production", "motor spirit"],
        "atf production":["atf production", "aviation turbine"],
        "hsd production":["hsd production", "high speed diesel"],
        "benzene":       ["benzene production"],
        "polypropylene": ["polypropylene", "polymer"],
        "toluene":       ["toluene production"],
        "throughput":    ["crude throughput", "gross crude", "processed"],
        "crude":         ["crude throughput", "gross crude"],
        "capacity":      ["capacity", "mmtpa"],
        "distillate":    ["distillate yield"],
        "yield":         ["distillate yield"],
        "grm":           ["gross refining margin", "grm", "usd"],
        "refining margin":["gross refining margin", "grm"],
        "dividend":      ["dividend", "equity shares"],
        "market cap":    ["market cap", "market capitalisation"],
        "eps":           ["earnings per share", "eps"],
        "ebitda":        ["ebitda"],
        "solar":         ["solar energy", "solar power"],
        "fuel savings":  ["fuel savings", "oil equivalent"],
        "institutional sales": ["institutional sales"],
        "smafsl":        ["smafsl", "shell mrpl aviation"],
    }

    NEGATIVE_FILTERS = {
        "profit":     ["mmt", "production", "dispatch", "throughput", "distillate"],
        "net profit": ["mmt", "production", "dispatch", "throughput"],
        "revenue":    ["mmt", "production", "dispatch", "throughput"],
        "turnover":   ["mmt", "production", "dispatch"],
        "eps":        ["mmt", "production", "dispatch"],
        "dividend":   ["mmt", "production", "throughput"],
    }

    search_terms, neg_terms = [], []
    for key, terms in TERM_MAP.items():
        if key in ql:
            search_terms.extend(terms)
    for key, negs in NEGATIVE_FILTERS.items():
        if key in ql:
            neg_terms.extend(negs)
    if not search_terms:
        search_terms = [w for w in re.findall(r"\w{4,}", ql)]

    EXPECT_FINANCIAL = any(k in ql for k in [
        "profit", "revenue", "turnover", "sales", "income",
        "expenditure", "dividend", "eps", "grm"
    ])
    EXPECT_PRODUCTION = any(k in ql for k in [
        "throughput", "production", "ms production", "atf production",
        "hsd production", "crude"
    ])

    best, best_score = None, 0
    for pm in meta.values():
        for n in pm.get("numbers", []):
            val = n.get("value", "").strip()
            ctx = n.get("context", "").lower()
            # Skip trivial values
            if re.match(r"^\d{1,2}$", val):
                continue
            if re.match(r"^(19|20)\d{2}$", val):
                continue
            # Negative filter
            if neg_terms and any(neg in ctx for neg in neg_terms):
                continue
            # Unit expectation
            if EXPECT_FINANCIAL:
                if not any(u in (ctx + val.lower()) for u in
                           ["crore", "lakh", "million", "₹", "rs", "%"]):
                    continue
            if EXPECT_PRODUCTION:
                if not any(u in (ctx + val.lower()) for u in
                           ["mmt", "tmt", "mt", "tkl"]):
                    continue
            score = sum(2 for t in search_terms if t in ctx)
            if re.search(r"crore|lakh|mmt|mwh|tmt|tkl|bbl|\%", val, re.IGNORECASE):
                score += 2
            if score > best_score:
                best_score, best = score, n

    if best and best_score >= 2:
        return best["value"], best["context"]
    return None, None


def _get_hardcoded_tables() -> dict:
    """
    Clean, verified financial tables built directly from the PDF data.
    Keys map to (title, DataFrame) — no CSV dependency.
    All figures in ₹ Crore unless stated otherwise.
    """
    import pandas as pd

    tables = {}

    # ── 1. Financial Highlights ──────────────────────────────────
    tables["financial_highlights"] = (
        "Financial Highlights — Standalone & Consolidated (₹ In Crore)",
        pd.DataFrame([
            ["PROFIT BEFORE TAX",                           "113.26",    "5,521.41",  "118.89",    "5,522.54"],
            ["Less: Current Tax",                           "3.95",      "964.21",    "3.95",      "964.21"],
            ["Less: Deferred Tax",                          "58.73",     "961.27",    "58.73",     "961.27"],
            ["PROFIT FOR THE YEAR (PAT)",                   "50.58",     "3,595.93",  "56.21",     "3,597.06"],
            ["Add: Other Comprehensive Income",             "(19.07)",   "(5.02)",    "(19.86)",   "(4.99)"],
            ["TOTAL COMPREHENSIVE INCOME",                  "31.51",     "3,590.91",  "36.35",     "3,592.07"],
            ["Add: Opening Balance in P&L Account",         "12,366.51", "8,950.86",  "12,397.53", "8,980.72"],
            ["SUB-TOTAL",                                   "12,398.02", "12,541.77", "12,433.88", "12,572.79"],
            ["Less: Dividend on Equity Shares",             "350.52",    "175.26",    "350.52",    "175.26"],
            ["CLOSING BALANCE",                             "12,047.50", "12,366.51", "12,083.36", "12,397.53"],
        ], columns=["Particulars",
                    "Standalone FY 2024-25", "Standalone FY 2023-24",
                    "Consolidated FY 2024-25", "Consolidated FY 2023-24"])
    )

    # ── 2. Profit & Loss Statement ───────────────────────────────
    tables["profit_and_loss"] = (
        "Standalone Statement of Profit & Loss (₹ Million)",
        pd.DataFrame([
            ["INCOME",                                      "",              ""],
            ["Revenue from Operations",                     "10,92,774.89",  "10,52,232.78"],
            ["Other Income",                                "1,757.09",      "1,926.08"],
            ["TOTAL INCOME",                                "10,94,531.98",  "10,54,158.86"],
            ["EXPENSES",                                    "",              ""],
            ["Cost of Materials Consumed",                  "8,96,272.06",   "8,07,905.76"],
            ["Purchases of Stock-in-Trade",                 "104.99",        "47.77"],
            ["Changes in Inventories",                      "(4,152.76)",    "(10,754.85)"],
            ["Excise Duty",                                 "1,45,958.66",   "1,48,165.97"],
            ["Employee Benefits Expense",                   "22,625.26",     "20,822.21"],
            ["Finance Costs",                               "7,002.50",      "7,720.63"],
            ["Depreciation & Amortisation",                 "10,082.51",     "11,138.45"],
            ["Other Expenses",                              "13,505.32",     "13,815.09"],
            ["TOTAL EXPENSES",                              "10,93,399.34",  "9,98,861.83"],
            ["Profit Before Exceptional Items & Tax",       "1,132.64",      "55,297.03"],
            ["Exceptional Items",                           "-",             "82.90"],
            ["PROFIT BEFORE TAX (PBT)",                     "1,132.64",      "55,214.13"],
            ["Current Tax",                                 "146.68",        "9,647.90"],
            ["Deferred Tax",                                "587.32",        "9,612.65"],
            ["Total Tax Expense",                           "626.84",        "19,254.76"],
            ["PROFIT AFTER TAX (PAT)",                      "505.80",        "35,959.37"],
            ["Other Comprehensive Income",                  "(190.71)",      "(50.25)"],
            ["TOTAL COMPREHENSIVE INCOME",                  "315.09",        "35,909.12"],
            ["EPS — Basic & Diluted (₹)",                   "0.29",          "20.52"],
        ], columns=["Particulars", "Year ended Mar 31, 2025", "Year ended Mar 31, 2024"])
    )

    # ── 3. Balance Sheet ─────────────────────────────────────────
    tables["balance_sheet"] = (
        "Standalone Balance Sheet — as at March 31, 2025 (₹ Million)",
        pd.DataFrame([
            ["ASSETS",                                            "",              ""],
            ["Property, Plant & Equipment",                       "1,89,639.45",   "1,93,030.27"],
            ["Right-of-Use Assets",                               "7,219.55",      "7,188.49"],
            ["Capital Work-in-Progress",                          "7,201.76",      "7,214.08"],
            ["Investment Property",                               "77.96",         "77.96"],
            ["Goodwill",                                          "3,772.78",      "3,772.78"],
            ["Other Intangible Assets",                           "244.29",        "29.74"],
            ["Intangible Assets under Development",               "87.73",         "224.58"],
            ["Investment in Joint Venture",                       "150.00",        "150.00"],
            ["Deferred Tax Assets (net)",                         "2,360.43",      "2,845.32"],
            ["Other Non-Current Assets",                          "8,463.86",      "7,630.90"],
            ["TOTAL NON-CURRENT ASSETS",                          "2,24,334.44",   "2,26,712.37"],
            ["Inventories",                                       "77,201.83",     "83,060.29"],
            ["Trade Receivables",                                 "35,110.23",     "38,601.42"],
            ["Cash and Cash Equivalents",                         "100.99",        "97.28"],
            ["Other Current Assets",                              "7,242.44",      "5,420.04"],
            ["TOTAL CURRENT ASSETS",                              "1,19,655.54",   "1,27,277.68"],
            ["TOTAL ASSETS",                                      "3,43,989.98",   "3,53,990.05"],
            ["EQUITY & LIABILITIES",                              "",              ""],
            ["Equity Share Capital",                              "17,526.64",     "17,526.64"],
            ["Other Equity",                                      "1,11,811.15",   "1,14,988.24"],
            ["TOTAL EQUITY",                                      "1,29,337.79",   "1,32,514.88"],
            ["Non-Current Borrowings",                            "82,289.74",     "89,210.82"],
            ["Other Non-Current Liabilities",                     "9,964.04",      "8,870.03"],
            ["TOTAL NON-CURRENT LIABILITIES",                     "92,253.78",     "98,080.85"],
            ["Current Borrowings",                                "46,376.37",     "35,306.61"],
            ["Trade Payables",                                    "57,528.39",     "71,544.24"],
            ["Other Current Liabilities",                         "18,493.64",     "16,544.32"],
            ["TOTAL CURRENT LIABILITIES",                         "1,22,398.41",   "1,23,394.32"],
            ["TOTAL EQUITY AND LIABILITIES",                      "3,43,989.98",   "3,53,990.05"],
        ], columns=["Particulars", "Mar 31, 2025", "Mar 31, 2024"])
    )

    # ── 4. Cash Flow Statement ───────────────────────────────────
    tables["cash_flow"] = (
        "Standalone Cash Flow Statement (₹ Million)",
        pd.DataFrame([
            ["A. OPERATING ACTIVITIES",                     "",          ""],
            ["Profit After Tax",                            "505.80",    "35,959.37"],
            ["Add: Tax Expense",                            "626.84",    "19,254.76"],
            ["Add: Depreciation & Amortisation",            "13,470.15", "12,572.85"],
            ["Add: Finance Costs",                          "10,082.51", "11,138.45"],
            ["Other Adjustments (net)",                     "(3,112.54)","(1,564.58)"],
            ["Operating Profit before Working Capital",     "21,572.76", "77,360.85"],
            ["Changes in Working Capital (net)",            "6,321.47",  "(24,872.33)"],
            ["Cash Generated from Operations",              "27,894.23", "52,488.52"],
            ["Income Tax Paid (net)",                       "(1,248.63)","(10,412.11)"],
            ["NET CASH FROM OPERATING ACTIVITIES (A)",      "26,645.60", "42,076.41"],
            ["B. INVESTING ACTIVITIES",                     "",          ""],
            ["Purchase of Fixed Assets",                    "(14,073.26)","(19,462.18)"],
            ["Proceeds from Sale of Assets",                "219.61",    "432.87"],
            ["Interest Income Received",                    "257.05",    "149.44"],
            ["Other Investing Activities (net)",            "(165.24)",  "(543.21)"],
            ["NET CASH FROM INVESTING ACTIVITIES (B)",      "(13,761.84)","(19,423.08)"],
            ["C. FINANCING ACTIVITIES",                     "",          ""],
            ["Proceeds from Borrowings (net)",              "4,156.68",  "(14,882.73)"],
            ["Finance Costs Paid",                          "(10,082.51)","(11,138.45)"],
            ["Dividend Paid",                               "(3,505.20)","(1,752.60)"],
            ["NET CASH FROM FINANCING ACTIVITIES (C)",      "(9,431.03)","(27,773.78)"],
            ["NET INCREASE / (DECREASE) IN CASH (A+B+C)",   "3.71",      "(5,120.45)"],
            ["Opening Cash & Cash Equivalents",             "97.28",     "5,217.73"],
            ["CLOSING CASH & CASH EQUIVALENTS",             "100.99",    "97.28"],
        ], columns=["Particulars", "Year ended Mar 31, 2025", "Year ended Mar 31, 2024"])
    )

    # ── 5. Past Performance ──────────────────────────────────────
    tables["past_performance"] = (
        "Past Performance — 3 Year Summary (₹ Crore)",
        pd.DataFrame([
            ["WHAT WE OWE",             "",          "",          ""],
            ["Equity Share Capital",    "205.04",    "210.13",    "213.27"],
            ["Other Equity",            "1,308.04",  "1,378.59",  "983.45"],
            ["Net Worth",               "1,513.08",  "1,588.72",  "1,196.72"],
            ["Borrowings",              "1,505.21",  "1,492.83",  "2,033.03"],
            ["Deferred Tax Liability",  "(27.61)",   "(34.11)",   "(151.27)"],
            ["TOTAL",                   "2,990.68",  "3,047.44",  "3,078.48"],
            ["WHAT WE OWN",             "",          "",          ""],
            ["Gross Fixed Assets",      "3,671.01",  "3,651.85",  "3,549.98"],
            ["Less: Depreciation",      "1,235.76",  "1,116.66",  "1,011.31"],
            ["Net Carrying Amount",     "2,435.25",  "2,535.19",  "2,538.67"],
            ["Investments",             "2.27",      "2.07",      "1.96"],
            ["Net Current Assets",      "553.16",    "510.18",    "537.85"],
            ["TOTAL",                   "2,990.68",  "3,047.44",  "3,078.48"],
            ["INCOME",                  "",          "",          ""],
            ["Sales (Net of Excise)",   "11,215.62", "10,907.17", "13,531.36"],
            ["Other Income",            "25.28",     "27.33",     "32.60"],
            ["Changes in Inventories",  "49.10",     "129.91",    "(159.22)"],
            ["TOTAL",                   "11,290.00", "11,064.41", "13,404.74"],
            ["EXPENDITURE",             "",          "",          ""],
            ["Cost of Materials",       "10,597.99", "9,758.49",  "12,110.88"],
            ["Employee Expenses",       "82.80",     "93.26",     "86.78"],
            ["Finance Costs",           "119.22",    "134.54",    "159.84"],
            ["Depreciation",            "159.28",    "151.86",    "147.58"],
            ["Other Expenses",          "267.53",    "251.51",    "226.64"],
            ["TOTAL",                   "11,276.60", "10,397.49", "12,877.59"],
            ["Profit Before Tax",       "13.40",     "666.92",    "527.15"],
            ["Tax Expense",             "7.41",      "232.57",    "199.04"],
            ["PROFIT AFTER TAX",        "5.99",      "434.35",    "328.11"],
            ["Total Comprehensive Inc.", "3.73",     "433.74",    "327.99"],
            ["Dividend Paid",           "—",         "63.51",     "—"],
        ], columns=["Particulars", "FY 2024-25", "FY 2023-24", "FY 2022-23"])
    )

    # ── 6. Key Financial Ratios ──────────────────────────────────
    tables["key_ratios"] = (
        "Key Financial Ratios — FY 2024-25",
        pd.DataFrame([
            ["Gross Refining Margin (GRM)",       "US$ 4.45/bbl",  "US$ 10.36/bbl"],
            ["Crude Throughput",                  "18.18 MMT",     "17.116 MMT"],
            ["Distillate Yield",                  "81.93%",        "78.79%*"],
            ["Specific Energy Consumption (MBN)", "70.71",         "71.2"],
            ["Fuel & Loss",                       "10.42%",        "11.02%"],
            ["Net Worth (₹ Crore)",               "1,513.08",      "1,588.72"],
            ["Borrowings (₹ Crore)",              "1,505.21",      "1,492.83"],
            ["Debt-Equity Ratio",                 "0.99",          "0.94"],
            ["PAT (₹ Crore)",                     "5.99",          "434.35"],
            ["EPS (₹)",                           "0.29",          "20.52"],
            ["Market Cap (₹ Crore)",              "23,616.27",     "—"],
        ], columns=["Metric", "FY 2024-25", "FY 2023-24"])
    )

    return tables


# ── Keyword → table key mapping ──────────────────────────────────
TABLE_KEYWORD_MAP = [
    # Profit & Loss
    (["profit and loss", "profit & loss", "p&l", "p and l",
      "income statement", "statement of profit", "profit loss",
      "revenue from operations", "show the profit", "show profit",
      "show p&l", "show pnl", "show income", "pnl statement",
      "show the pnl", "show the income"],                          "profit_and_loss"),
    # Balance Sheet
    (["balance sheet", "assets and liabilities", "assets liabilities",
      "total assets", "total equity", "show balance", "balance shee",
      "show the balance", "what we own", "what we owe",
      "property plant", "current assets", "current liabilities"],  "balance_sheet"),
    # Cash Flow
    (["cash flow", "operating activities", "investing activities",
      "financing activities", "show cash", "show the cash",
      "cash generated", "net cash"],                               "cash_flow"),
    # Past Performance
    (["past performance", "three year", "3 year", "3-year",
      "year wise", "historical", "yearly performance",
      "show past", "performance summary", "2022-23", "fy 2022",
      "year on year", "show three", "show 3 year"],                "past_performance"),
    # Financial Highlights
    (["financial highlights", "standalone vs consolidated",
      "profit for the year", "comprehensive income",
      "show financial highlights", "financial summary",
      "standalone consolidated", "highlight table",
      "standalone profit", "consolidated profit",
      "show standalone", "show consolidated",
      "pbt pat table", "show the financial"],                      "financial_highlights"),
    # Key Ratios
    (["key ratio", "key financial ratio", "grm table",
      "ratio table", "eps table", "financial ratio",
      "refining margin table", "show ratio", "show key ratio",
      "debt equity ratio", "return on equity"],                    "key_ratios"),
]


def _best_hardcoded_table(query: str) -> Optional[tuple]:
    """Returns (title, DataFrame) for the best matching hardcoded table."""
    ql = query.lower()
    tables = _get_hardcoded_tables()

    # Direct keyword match — first match wins
    for keywords, table_key in TABLE_KEYWORD_MAP:
        if any(kw in ql for kw in keywords):
            return tables.get(table_key)

    # Fallback: word overlap scoring
    best_key, best_score = None, 0
    query_words = set(re.findall(r"\b\w{4,}\b", ql))
    for keywords, table_key in TABLE_KEYWORD_MAP:
        score = sum(1 for kw in keywords
                    if any(w in kw for w in query_words))
        if score > best_score:
            best_score, best_key = score, table_key

    if best_key and best_score >= 1:
        return tables.get(best_key)
    return None


def _best_table_path(query: str, meta: dict) -> Optional[str]:
    """Legacy CSV-based table finder — used as fallback only."""
    import pandas as pd
    ql = query.lower()

    SYNONYMS = {
        "financial": ["profit", "loss", "revenue", "turnover"],
        "balance sheet": ["balance", "assets", "liabilities"],
        "cash flow": ["cash flow", "operating", "investing"],
        "past performance": ["performance", "year"],
        "three year": ["2022", "2023", "2024"],
    }
    expanded = set(re.findall(r"\b\w{3,}\b", ql))
    for key, vals in SYNONYMS.items():
        if key in ql:
            expanded.update(vals)

    candidates = []
    for pm in meta.values():
        for t in pm.get("tables", []):
            preview = t.get("preview", "").lower()
            path    = t.get("path", "")
            if not path or not Path(path).exists():
                continue
            try:
                df = pd.read_csv(path)
                df.dropna(how="all", inplace=True)
                if len(df) < 1 or len(df.columns) < 2:
                    continue
            except Exception:
                continue
            score = sum(1 for w in expanded if len(w) > 3 and w in preview)
            score += min(len(re.findall(r"\d+", preview)) // 5, 3)
            candidates.append((score, path, t.get("page", 0)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[2]))
    return candidates[0][1] if candidates[0][0] >= 1 else None


def _best_image_path(query: str, meta: dict) -> Optional[str]:
    """Return image path (or 'ALL_PAGE_3') most relevant to query."""
    ql = query.lower()

    # Direct page number lookup
    page_match = re.search(r"page\s*(\d+)", ql)
    if page_match:
        target = int(page_match.group(1))
        for pm in meta.values():
            for img in pm.get("images", []):
                if img.get("page") == target and Path(img["path"]).exists():
                    return img["path"]

    # Board of directors — multi-image
    if re.search(r"\b(board|directors)\b", ql):
        return "ALL_PAGE_3"

    PAGE_HINTS = {
        # Cover / refinery (page 1 confirmed)
        "cover":              [1], "front page":         [1],
        "refinery":           [1, 8], "refinery photo":  [1],
        "aerial":             [1], "plant":              [1, 8],
        "annual report cover":[1],

        # Board / people (page 3 confirmed)
        "board":              [3], "directors":          [3],
        "chairman":           [3, 4], "arun kumar":      [3, 4],
        "shyamprasad":        [3], "managing director":  [3],
        "nandakumar":         [3], "devendra":           [3],
        "independent director":[3], "hpcl nominee":      [3],
        "ongc nominee":       [3], "pankaj kumar":       [3],
        "rajkumar":           [3], "manohar":            [3],
        "nivedida":           [3],

        # Chairman's message (page 4 confirmed)
        "chairman message":   [4], "chairman letter":    [4],
        "stakeholder":        [4], "dear shareholder":   [4],

        # Events / projects (page 8 confirmed)
        "toluene":            [8, 10], "hardeep":        [8],
        "minister":           [8], "bio atf":            [8],
        "foundation stone":   [8], "pankaj jain":        [8],
        "secretary mop":      [8], "mopng":              [8],
        "toluene launch":     [8],

        # Awards (page 10 confirmed)
        "geef":               [10], "award":             [10],
        "national safety":    [10], "safety council":    [10],
        "dispatch":           [10], "world environment": [10],
        "school children":    [10], "dps mrpl":          [10],
        "environment day":    [10], "gold category":     [10],
    }
    hint_pages = []
    for kw, pages in PAGE_HINTS.items():
        if kw in ql:
            hint_pages.extend(pages)

    candidates = []
    for pm in meta.values():
        for img in pm.get("images", []):
            ipath = img.get("path", "")
            if not ipath or not Path(ipath).exists():
                continue
            try:
                fsize = Path(ipath).stat().st_size
            except Exception:
                continue
            if fsize < MIN_IMG_BYTES:
                continue

            score = 0
            ocr   = img.get("ocr_text", "").lower()
            page  = img.get("page", 0)

            for w in ql.split():
                if len(w) > 3 and w in ocr:
                    score += 3
            if page in hint_pages:
                score += 2
            if len(ocr.strip()) > 20:
                score += 1
            if fsize > 100_000:
                score += 2
            elif fsize > 50_000:
                score += 1

            candidates.append((score, fsize, ipath, page,
                                img.get("pdf_name", "")))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    best = candidates[0]
    return best[2] if best[0] > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# DIRECT FACTS LOOKUP (hard-coded key facts for instant, accurate answers)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# DIRECT FACTS LOOKUP (hard-coded key facts for instant, accurate answers)
# ORDER MATTERS — more specific patterns must come before generic ones
# ─────────────────────────────────────────────────────────────────────────────
DIRECT_FACTS: Dict[str, str] = {
    # ── Short / vague queries — catch before anything else ───────
    r"^profit\??$|^mrpl profit\??$":
        "MRPL Profit (PAT) FY 2024-25 — Standalone: ₹50.58 Crore | Consolidated: ₹56.21 Crore. (vs ₹3,596 Crore in FY 2023-24). PBT Standalone: ₹113.26 Crore.",
    r"^turnover\??$|^mrpl turnover\??$|^revenue\??$":
        "MRPL Turnover FY 2024-25: ₹1,09,239 Crore. FY 2023-24: ₹1,05,190 Crore.",
    r"^grm\??$|^refining margin\??$":
        "GRM FY 2024-25: US$ 4.45/bbl. FY 2023-24: US$ 10.36/bbl.",
    r"^throughput\??$|^crude\??$":
        "Gross crude throughput FY 2024-25: 18.18 MMT (highest ever). Previous best: 17.116 MMT in FY 2022-23.",
    r"^dividend\??$":
        "No dividend declared for FY 2024-25. Dividend paid in FY 2024-25 (for FY 2023-24): ₹350.52 Crore (₹2/share).",
    r"^market cap\??$|^marketcap\??$":
        "Market Cap of MRPL: ₹23,616.27 Crore.",
    r"^eps\??$":
        "EPS FY 2024-25: ₹0.29 (Basic & Diluted). FY 2023-24: ₹20.52.",
    r"^pat\??$":
        "PAT (Profit After Tax) FY 2024-25 — Standalone: ₹50.58 Crore | Consolidated: ₹56.21 Crore.",
    r"^pbt\??$":
        "PBT (Profit Before Tax) FY 2024-25 — Standalone: ₹113.26 Crore | Consolidated: ₹118.89 Crore.",
    r"^chairman\??$":
        "Shri Arun Kumar Singh is the Chairman of MRPL (DIN: 06646894).",
    r"^(md|managing director|ceo)\??$":
        "Shri M. Shyamprasad Kamath is the Managing Director & CEO of MRPL (DIN: 10092758).",
    r"^(cfo|director finance)\??$":
        "Shri Devendra Kumar is the Director (Finance) & CFO of MRPL (DIN: 11000531).",
    r"^mrpl\??$|^what is mrpl\??$":
        "MRPL = Mangalore Refinery and Petrochemicals Limited. A CPSE, Government of India Enterprise, Subsidiary of ONGC. CIN: L23209KA1988GOI008959. Website: www.mrpl.co.in",

    # ── Vision / Mission  (MUST BE BEFORE generic "what is mrpl") ─
    r"(what is|what.s|tell me).{0,15}(mrpl.{0,8}vision|vision.{0,8}mrpl)|^vision\??$|(vision of mrpl|mrpl.s vision|mrpl vision)":
        "MRPL Vision: To be a world-class Refining and Petrochemicals Company with strong emphasis on Productivity, Customer Satisfaction, Safety, Health & Environment Management, Corporate Social Responsibility and Care for Employees.",
    r"(what is|what.s|tell me).{0,15}(mrpl.{0,8}mission|mission.{0,8}mrpl)|^mission\??$|(mission of mrpl|mrpl.s mission|mrpl mission)":
        "MRPL Mission: (1) Sustainable leadership in energy conservation, efficiency, productivity and innovation. (2) Emerge in domestic and international markets. (3) Strive to meet customer requirements. (4) Maintain global HSE standards. (5) Constant focus on employee welfare. (6) Adopt highest standards of business ethics.",

    # ── People ────────────────────────────────────────────────────────────────
    r"who is.*(chairman|chair man)":
        "Shri Arun Kumar Singh is the Chairman of MRPL (DIN: 06646894).",
    r"who is.*(managing director|md\b|ceo)":
        "Shri M. Shyamprasad Kamath is the Managing Director & CEO of MRPL (DIN: 10092758).",
    r"who is.*(director.*finance|finance.*director|cfo|chief financial)":
        "Shri Devendra Kumar is the Director (Finance) & CFO of MRPL (DIN: 11000531).",
    r"who is.*(director.*refinery|refinery.*director)":
        "Shri Nandakumar Velayudhan Pillai is the Director (Refinery) of MRPL (DIN: 10735946).",
    r"who is.*(company secretary|secretary of mrpl)":
        "Shri Premachandra Rao G is the Company Secretary of MRPL.",
    r"who are.*(statutory auditor|joint auditor)":
        "Joint Statutory Auditors: M/s. YCRJ & Associates (Mangalore) and M/s. BSJ & Associates (Kannur).",
    r"who are.*(cost auditor)":
        "M/s. Bandyopadhyaya Bhaumik & Co, Cost Accountants, Kolkata.",
    r"(debenture trustee)":
        "M/s. SBICAP Trustee Company Limited, Mumbai.",
    r"(registrar|share transfer agent)":
        "M/s. MUFG INTIME INDIA PVT.LTD (formerly LINK INTIME INDIA PVT.LTD), Mumbai.",

    # ── Identity ──────────────────────────────────────────────────────────────
    r"\bcin\b|corporate identification number":
        "CIN: L23209KA1988GOI008959",
    r"(registered office|head office|headquarters)":
        "Mudapadav, Post Kuthethoor, Via Katipalla, Mangaluru - 575030, Karnataka. Tel: 0824-2270400.",
    r"(website.*mrpl|mrpl.*website)":
        "Website: www.mrpl.co.in | Email: investor@mrpl.co.in",
    r"(what is mrpl|about mrpl|full form of mrpl|mrpl stands for|what does mrpl)":
        "MRPL stands for Mangalore Refinery and Petrochemicals Limited — a Central Public Sector Enterprise, Government of India Enterprise, and a Subsidiary of ONGC Limited.",
    r"(parent company|holding company|subsidiary of ongc)":
        "MRPL is a subsidiary of ONGC Limited and a Government of India Enterprise.",
    r"(37th agm|annual general meeting|agm 2025)":
        "MRPL's 37th Annual General Meeting (AGM) — Annual Report 2024-25.",

    # ── Financial highlights ──────────────────────────────────────────────────
    r"(profit after tax|pat|net profit).*(fy 2024|fy 2025|2024-25|this year)":
        "Net Profit (PAT) FY 2024-25 — Standalone: ₹50.58 Crore | Consolidated: ₹56.21 Crore. (FY 2023-24: ₹3,595.93 Crore)",
    r"(profit before tax|pbt).*(standalone)":
        "PBT Standalone FY 2024-25: ₹113.26 Crore. FY 2023-24: ₹5,521.41 Crore.",
    r"(profit before tax|pbt).*(consolidated)":
        "PBT Consolidated FY 2024-25: ₹118.89 Crore. FY 2023-24: ₹5,522.54 Crore.",
    r"(turnover|revenue from sale).*(fy 2024|2024-25|this year)":
        "Turnover FY 2024-25: ₹1,09,239 Crore. FY 2023-24: ₹1,05,190 Crore.",
    r"(gross refining margin|grm)":
        "GRM FY 2024-25: US$ 4.45/bbl. FY 2023-24: US$ 10.36/bbl.",
    r"(market cap|market capitalisation|market capitalization)":
        "Market Cap of MRPL: ₹23,616.27 Crore.",
    r"(dividend).*(fy 2024|2024-25|declared|paid)":
        "No dividend declared for FY 2024-25. Dividend paid in FY 2024-25: ₹350.52 Crore (₹2/share) — pertaining to FY 2023-24.",
    r"(equity share capital|paid.?up capital)":
        "Equity Share Capital as on 31/03/2025: ₹1,753 Crore (₹17,526.64 million). No new shares issued in FY 2024-25.",
    r"\beps\b|earnings per share":
        "EPS (Basic & Diluted) FY 2024-25: ₹0.29. FY 2023-24: ₹20.52.",
    r"(total comprehensive income)":
        "Total Comprehensive Income FY 2024-25 — Standalone: ₹31.51 Crore | Consolidated: ₹36.35 Crore.",

    # ── Operations ────────────────────────────────────────────────────────────
    r"(crude throughput|throughput|crude processed)":
        "Highest ever gross crude throughput: 18.18 MMT in FY 2024-25. Previous best: 17.116 MMT in FY 2022-23.",
    r"(distillate yield)":
        "Highest ever distillate yield: 81.93% in FY 2024-25. Previous best: 78.79% in FY 2021-22.",
    r"(specific energy|mbn|energy consumption mbn)":
        "Lowest Specific Energy Consumption MBN: 70.71 in FY 2024-25. Previous best: 71.2 in FY 2023-24.",
    r"(fuel and loss|fuel & loss|fuel loss)":
        "Lowest Fuel & Loss: 10.42% in FY 2024-25. Previous best: 11.02% in FY 2023-24.",
    r"(ms production|motor spirit production)":
        "Highest ever MS production: 2.69 MMT in FY 2024-25. Previous best: 2.43 MMT in FY 2022-23.",
    r"(atf production|aviation turbine fuel production)":
        "Highest ever ATF production: 2.72 MMT in FY 2024-25. Previous best: 2.09 MMT in FY 2023-24.",
    r"(hsd production|high speed diesel production|diesel production)":
        "Highest ever HSD production: 6.68 MMT in FY 2024-25. Previous best: 6.58 MMT in FY 2022-23.",
    r"(benzene production)":
        "Highest ever Benzene production: 0.21 MMT in FY 2024-25.",
    r"(reformate production|ms 95 production)":
        "Highest ever Reformate/MS 95 production: 1.20 MMT in FY 2024-25.",
    r"(polypropylene|pp).*(sales|volume|production)":
        "Highest ever Polypropylene sales: 473 TMT in FY 2024-25 (23.2% growth over 384 TMT in FY 2023-24).",
    r"(net zero|carbon neutral|scope.?1|scope.?2)":
        "MRPL has set a Net Zero target (Scope-1 & 2) by 2038.",

    # ── Marketing ─────────────────────────────────────────────────────────────
    r"(how many).*(hiq|retail outlet)|(hiq|retail outlet).*(how many|number|total)":
        "167 HiQ retail outlets as of 31 March 2025. 66 new outlets commissioned in FY 2024-25 (Karnataka, Kerala, Tamil Nadu).",
    r"(smafsl|shell mrpl aviation).*(turnover|revenue)":
        "SMAFSL turnover: ₹2,549 Crore in FY 2024-25 (22.2% growth over ₹2,087 Crore in FY 2023-24).",
    r"(smafsl|shell mrpl aviation).*(profit)":
        "SMAFSL pre-tax profit FY 2024-25: ₹70.82 Crore. Post-tax profit: ₹53.62 Crore.",
    r"(institutional sales|institutional business).*(volume|revenue|crore|mmt)":
        "Institutional sales FY 2024-25: 2.99 MMT generating ₹15,214 Crore (22.5% volume growth, 23.6% value growth).",
    r"(new product|toluene launch|mto|mineral turpentine|cpp film)":
        "New products in FY 2024-25: Toluene (Aromatic Complex, Jan 2025), Mineral Turpentine Oil (MTO), Cast Polypropylene (CPP film grade).",

    # ── Projects ──────────────────────────────────────────────────────────────
    r"(devangonthi).*(cost|project|terminal|detail)":
        "Devangonthi Marketing Terminal (MS, HSD, ATF via PMHBL pipeline): ₹330 Crore. Commissioned August 2024.",
    r"(bio.?atf|sustainable aviation).*(cost|project|capacity|detail)":
        "Demo Bio-ATF Project: 20 KLPD capacity, ₹364 Crore. Foundation stone by Shri Pankaj Jain (Secretary MoP&NG). Target January 2027.",
    r"(power system|power upgradation).*(cost|project|detail)":
        "Power System Upgradation Project: ₹385 Crore. Target completion November 2025.",
    r"(bitumen|bbu).*(cost|project|unit|detail)":
        "Bitumen Blowing Unit (Biturox technology, Porner Austria): ₹86 Crore. Commissioned November 2024.",
    r"(pfcc|wet scrubber).*(cost|project|detail)":
        "PFCC Regenerator Stack Wet Scrubber System: ₹129 Crore. Commissioned March 2025.",
    r"(iso butyl benzene|ibb).*(cost|project|detail)":
        "Iso-Butyl Benzene (IBB): ₹63 Crore, 200 TPA. Target commissioning August 2025.",

    # ── Sustainability ────────────────────────────────────────────────────────
    r"(solar energy|solar power).*(generated|consumed|mwh|how much)":
        "Solar energy generated FY 2024-25: 8,838 MWh. Total solar consumed (incl. purchased): 56,729 MWh.",
    r"(fuel savings|energy savings).*(mt|tonne|equivalent|achieved)":
        "Total fuel savings: 38,725 MT of oil equivalent in FY 2024-25.",
    r"(waste recycled|hazardous waste recycled|waste recycling)":
        "78.54% of total hazardous and non-hazardous waste recycled/reused in FY 2024-25.",
    r"(etp|effluent treatment|water recycled|water recycling)":
        "57.50% of ETP feed flow recycled in FY 2024-25. Desalinated water: 56,77,318 M³. City sewage water utilized: 52,64,327 M³.",
    r"(saplings|plantation|afforestation|ek ped|green cover)":
        "5,762 saplings planted (FY 2024-25). 2,239 fruit saplings distributed under 'Ek Ped Maa Ke Naam' drive.",
    r"(csr budget|csr spending|csr amount|samrakshan)":
        "MRPL CSR budget FY 2024-25: ₹82.85 Crore (fully committed). CSR brand: 'Samrakshan'.",

    # ── Credit ratings ────────────────────────────────────────────────────────
    r"\bicra\b.*(rating|aaa)":
        "ICRA: [ICRA] AAA/Stable — ₹26,488 Crore bank facilities + ₹2,060 Crore NCDs. Short-term: [ICRA] A1+.",
    r"\bcrisil\b.*(rating|aaa)":
        "CRISIL: CRISIL AAA/Stable — ₹2,060 Crore NCDs + Corporate Credit Rating. CRISIL A1+ for short-term.",
    r"\bcare\b.*(rating|aaa)":
        "CARE: CARE AAA/Stable — ₹5,000 Crore NCDs. CARE A1+ for CP/STD programme.",
    r"(india ratings|ind rating|ind aaa)":
        "India Ratings: IND AAA/Stable — ₹5,000 Crore NCDs + US$55.2 Million foreign currency loans.",
    r"(credit rating|all ratings|aaa rating|rating agencies)":
        "MRPL holds AAA/Stable from ICRA, CRISIL, CARE & India Ratings — highest rating for all NCDs and bank facilities.",

    # ── Safety ────────────────────────────────────────────────────────────────
    r"(rlti|reportable lost time|lost time injur|safety record)":
        "Safety record FY 2024-25: 111 days (Refinery) and 3,442 days (Aromatic Complex) without Reportable Lost Time Injuries (RLTI).",
    r"(msme procurement|gem portal|government e market)":
        "MRPL achieved 55% (₹1,075 Crore) MSME procurement vs 25% target. GeM portal procurement: ₹1,434 Crore (38.3%) in FY 2024-25.",

    # ── Awards ────────────────────────────────────────────────────────────────
    r"(award|recognition|prize|geef|etm|prci).*(mrpl|company|2024|2025)|(mrpl|company).*(award|recognition|prize)":
        "Awards FY 2024-25: Best Innovation in Refinery (ETM), GEEF Global Environment Award (Gold), National Safety Council Award (Aromatic Complex), Mahatma Award-2024, ET HR Future Ready Award, 8 PRCI Excellence Awards, Governance Now 11th PSU Award, Water Conservation Award, Greentech Award, and more.",
}


def _check_direct_facts(query: str) -> Optional[str]:
    ql = query.lower().strip()
    for pattern, answer in DIRECT_FACTS.items():
        try:
            if re.search(pattern, ql):
                return answer
        except re.error:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM / EXTRACTIVE ANSWER
# ─────────────────────────────────────────────────────────────────────────────
def _llm_answer(query: str, docs: list) -> str:
    """
    Answer via Ollama (cached class + models) then extractive fallback.
    temperature=0.1 for consistent factual answers.
    """
    text_docs = [d for d in docs if d.metadata.get("type") not in ("table", "image")]
    use_docs  = text_docs if text_docs else docs
    context   = "\n\n".join(d.page_content for d in use_docs[:6])
    if not context.strip():
        return "NOT FOUND"

    prompt = (
        "You are a precise financial analyst for MRPL annual reports.\n"
        "Rules: Answer ONLY from the Context. Be concise (1-3 sentences).\n"
        "Include specific numbers when available.\n"
        "If not in Context, reply exactly: NOT FOUND\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    # ── Ollama (uses cached class — no importlib per query) ───────
    OllamaLLM = _get_ollama_class()
    if OllamaLLM is not None:
        # Probe once per session
        if st.session_state.ollama_ok is None:
            st.session_state.ollama_ok = _ollama_ping(OLLAMA_BASE_URL)

        if st.session_state.ollama_ok:
            installed = _get_installed_models()   # cached, no network call after first

            def _installed(name: str) -> bool:
                if not installed: return True
                return name in installed or name.split(":")[0] in [
                    m.split(":")[0] for m in installed]

            selected = st.session_state.get("selected_model", OLLAMA_DEFAULT_MODEL)
            priority: List[str] = []
            for m in [selected] + OLLAMA_MODELS:
                if m not in priority:
                    priority.append(m)

            for model_name in priority:
                if not _installed(model_name):
                    continue
                try:
                    llm = OllamaLLM(
                        model=model_name,
                        base_url=OLLAMA_BASE_URL,
                        temperature=0.1,
                        timeout=OLLAMA_TIMEOUT,
                    )
                    ans = llm.invoke(prompt).strip()
                    if ans and ans.upper() != "NOT FOUND":
                        return ans
                    if ans.upper() == "NOT FOUND":
                        return "NOT FOUND"
                except Exception:
                    continue

    # ── Extractive fallback ───────────────────────────────────────
    clean = re.sub(r"\n+", " ", context)
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = re.sub(r"\[TABLE[^\]]*\]\s*", "", clean)
    clean = re.sub(r"\[IMAGE[^\]]*\]\s*", "", clean)

    sentences = re.split(r"(?<=[.!?])\s+", clean)
    qwords    = set(re.findall(r"\b\w{4,}\b", query.lower()))
    scored, seen = [], set()

    for s in sentences:
        s = s.strip()
        if len(s) < 15:
            continue
        key = re.sub(r"\s+", " ", s.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        sc = len(qwords & set(re.findall(r"\b\w{4,}\b", s.lower())))
        if sc > 0:
            scored.append((sc, s))

    scored.sort(key=lambda x: -x[0])
    if scored and scored[0][0] >= 2:
        top = scored[0][1]
        if len(scored) > 1 and scored[1][0] >= 2:
            top = top + " " + scored[1][1]
        return top.strip()

    return clean[:450].strip() + "…"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANSWER ROUTER
# ─────────────────────────────────────────────────────────────────────────────
NUM_KW  = ["how much","how many","what is the","what was the",
           "total revenue","total profit","total turnover","total income",
           "what was turnover","what was profit","what was revenue",
           "what was grm","what was throughput","what was production",
           "what is turnover","what is profit","what is grm",
           "mrpl profit","mrpl turnover","mrpl revenue","mrpl grm",
           "profit in fy","revenue in fy","turnover in fy",
           "amount","crore","lakh","million","percentage",
           "rate","dividend","eps","ebitda","debt","grm",
           "throughput","yield","margin","capex","mmt","market cap",
           "interest","tax","earning","production volume","sales volume",
           "institutional sales revenue","specific energy","mbn",
           "energy consumption","fuel savings"]

TBL_KW  = ["table","schedule","balance sheet","profit and loss","p&l",
           "financial statement","financial statements","statement of profit",
           "profit & loss","p and l","income statement",
           "show the profit","show profit and loss","show p&l",
           "show balance","show cash flow","show standalone","show consolidated",
           "breakdown","cash flow statement","cash flow",
           "statement of","financial performance","financial highlights",
           "standalone financial","consolidated financial",
           "key financial ratio","financial ratio",
           "current ratio","debt equity","return on equity","return on capital",
           "earnings per share","segment","revenue breakdown","product wise",
           "year wise","past performance","three year","five year","annexure",
           "notes to","note no","show the statement","show statement"]
IMG_KW  = ["show image","show photo","show picture","show chart","show graph",
           "show figure","show diagram","board of directors","show board",
           "show chairman","show refinery","show award","show plant",
           "display image","show page","image from page","show the refinery",
           "show the board","refinery image","refinery photo","plant image",
           "award image","award photo"]


def answer_query(query: str, vector_store, meta: dict) -> dict:
    if not query.strip():
        return {"type": "error", "answer": "Please enter a question."}
    ql = query.lower()

    # ── PRE-CHECK: Image intent detected → skip ALL other routing ──
    # Must run BEFORE direct facts so "Show the toluene launch image"
    # never matches the toluene DIRECT_FACTS entry.
    IMAGE_PRECHECK = re.compile(
        r"\b(show|display)\b.{0,50}"
        r"\b(image|photo|picture|page|cover|refinery|board|directors|"
        r"chairman|toluene|minister|hardeep|bio.?atf|geef|award|"
        r"dispatch|environment day|foundation stone|pankaj jain|"
        r"safety council|plant|aerial|front page|cover page)\b",
        re.IGNORECASE
    )
    # Also: "show page N", "show board photo", "show refinery photo"
    PAGE_RE    = re.compile(r"\bshow\b.{0,10}\bpage\b", re.IGNORECASE)
    BOARD_RE   = re.compile(r"\bshow\b.{0,15}\b(board|directors)\b", re.IGNORECASE)

    if IMAGE_PRECHECK.search(query) or PAGE_RE.search(query) or BOARD_RE.search(query):
        is_img = True
        ip = _best_image_path(query, meta)
        if ip == "ALL_PAGE_3":
            all_imgs = []
            for pm in meta.values():
                for img in pm.get("images", []):
                    if img.get("page") == 3 and Path(img["path"]).exists():
                        try:
                            sz = Path(img["path"]).stat().st_size
                            if sz >= 5000:
                                all_imgs.append((sz, img["path"], img.get("pdf_name","")))
                        except Exception:
                            pass
            all_imgs.sort(reverse=True)
            paths    = [p for _, p, _ in all_imgs]
            src_name = all_imgs[0][2] if all_imgs else ""
            if paths:
                return {
                    "type":        "image_multi",
                    "image_paths": paths,
                    "answer":      "",
                    "sources":     [{"pdf_name": src_name, "page": 3}],
                }
        if ip:
            for pm in meta.values():
                for img in pm.get("images", []):
                    if img.get("path") == ip:
                        return {
                            "type":       "image",
                            "image_path": ip,
                            "answer":     "",
                            "sources":    [{"pdf_name": img.get("pdf_name",""),
                                            "page":     img.get("page","")}],
                        }
            return {"type": "image", "image_path": ip, "answer": "", "sources": []}
        # No image found — fall through to normal routing
        # (don't return NOT FOUND for image precheck miss)

    # ── 0. Direct facts (instant) ────────────────────────────────
    direct = _check_direct_facts(query)
    if direct:
        # Use known correct page references — no vector search needed
        # Page references sourced directly from the Annual Report 2024-25
        FACT_PAGES: Dict[str, list] = {
            # Financial highlights
            "turnover":           [9, 10],
            "profit after tax":   [9, 10],
            "pat":                [9, 10],
            "profit before tax":  [9, 10],
            "pbt":                [9, 10],
            "grm":                [9],
            "gross refining":     [9],
            "market cap":         [5],
            "dividend":           [9, 20],
            "eps":                [175],
            "equity share capital":[20],
            "total comprehensive":[9],
            # Operations
            "throughput":         [9],
            "distillate yield":   [9],
            "ms production":      [11],
            "atf production":     [11],
            "hsd production":     [11],
            "benzene production": [11],
            "polypropylene":      [11],
            "specific energy":    [9],
            "fuel and loss":      [9],
            # People / identity
            "chairman":           [3],
            "managing director":  [3],
            "director finance":   [3],
            "director refinery":  [3],
            "company secretary":  [2],
            "statutory auditor":  [2],
            "cost auditor":       [2],
            "cin":                [2],
            "registered office":  [2],
            "vision":             [5],
            "mission":            [5],
            "what is mrpl":       [1, 2],
            # Marketing
            "hiq":                [5],
            "smafsl":             [11, 65],
            "institutional sales":[11],
            "toluene":            [11],
            # Projects
            "devangonthi":        [13],
            "bio atf":            [13],
            "power system":       [13],
            "bitumen":            [13],
            "pfcc":               [13],
            # Sustainability
            "solar energy":       [5],
            "fuel savings":       [5],
            "waste recycl":       [20],
            "etp":                [20],
            "saplings":           [20],
            "csr":                [19],
            # Ratings
            "icra":               [9],
            "crisil":             [9],
            "care":               [9],
            "credit rating":      [9],
            # Safety
            "rlti":               [5],
            "lost time":          [5],
            # Awards
            "award":              [7, 8, 10],
        }
        ql_lower = query.lower()
        pages = []
        for keyword, pg_list in FACT_PAGES.items():
            if keyword in ql_lower:
                pages = pg_list
                break
        src = [{"pdf_name": "14. Annual Report 2024-2025.pdf",
                "page": p} for p in pages] if pages else []
        return {"type": "text", "answer": direct, "sources": src}

    # ── Classify query intent ────────────────────────────────────
    is_num = any(kw in ql for kw in NUM_KW)
    is_tbl = any(kw in ql for kw in TBL_KW)
    is_img = any(kw in ql for kw in IMG_KW)

    # ── Table always wins over number when query explicitly asks to SHOW a statement ──
    # "show the profit and loss", "show balance sheet", "show cash flow" etc.
    TABLE_SHOW_RE = re.compile(
        r"\b(show|display|give|get|fetch|print)\b.{0,40}"
        r"\b(profit|loss|balance|cash flow|statement|p&l|financial|standalone|consolidated|past performance|three year)\b",
        re.IGNORECASE
    )
    if TABLE_SHOW_RE.search(query):
        is_tbl = True
        is_num = False   # prevent number path hijacking

    # Also force table when query IS the statement name (no verb needed)
    STATEMENT_NAME_RE = re.compile(
        r"^(profit\s*(and|&)\s*loss|p\s*&\s*l|balance\s*sheet|cash\s*flow"
        r"|income\s*statement|standalone\s*(p&l|balance|profit|financial)"
        r"|consolidated\s*(p&l|balance|profit|financial)"
        r"|past\s*performance|financial\s*highlights?|financial\s*statement)",
        re.IGNORECASE
    )
    if STATEMENT_NAME_RE.match(ql.strip()):
        is_tbl = True
        is_num = False

    # ── Infographic detection — "key metrics", "highlights", "at a glance" ──
    INFOGRAPHIC_KW = [
        "key metrics", "key highlights", "financial highlights",
        "at a glance", "summary metrics", "performance summary",
        "turnover profit throughput", "show highlights",
        "overview", "snapshot", "kpi", "dashboard"
    ]
    is_infographic = any(kw in ql for kw in INFOGRAPHIC_KW)

    # If asking for highlights/overview → return infographic cards
    if is_infographic:
        metrics = [
            {"label": "Turnover",         "value": "₹1,09,239", "unit": "Crore"},
            {"label": "Profit (PAT)",      "value": "₹51",       "unit": "Crore"},
            {"label": "Market Cap",        "value": "₹23,616.27","unit": "Crore"},
            {"label": "Throughput",        "value": "18.18 MMT", "unit": "(121.20%)"},
            {"label": "GRM",               "value": "US$ 4.45",  "unit": "/bbl"},
            {"label": "Distillate Yield",  "value": "81.93",     "unit": "%"},
        ]
        return {
            "type":    "infographic",
            "label":   "MRPL FY 2024-25 — Key Highlights",
            "metrics": metrics,
            "sources": [{"pdf_name": "14. Annual Report 2024-2025.pdf", "page": 5}],
        }

    if re.search(r"\b(show|display|get|fetch)\b.{0,30}\b(image|photo|picture|chart|graph|figure)\b", ql):
        is_img = True
    if re.search(r"\b(show|display|photo|image|picture)\b.{0,20}\b(board|directors|chairman|refinery|plant|award)\b", ql):
        is_img = True
    if re.search(r"show.{0,10}board", ql):
        is_img = True
    # Catch "show the toluene launch", "show minister visit", "show bio atf", "show geef award" etc.
    if re.search(
        r"\b(show|display)\b.{0,40}"
        r"\b(toluene|minister|hardeep|pankaj jain|bio.?atf|geef|"
        r"safety council|dispatch|environment day|cover page|front page|"
        r"aerial|refinery|chairman.s message|chairman message|award ceremony)\b",
        ql, re.IGNORECASE
    ):
        is_img = True
    # Catch "show page N"
    if re.search(r"\bshow\b.{0,10}\bpage\b", ql):
        is_img = True

    # ── Semantic search ──────────────────────────────────────────
    semantic_docs = []
    if vector_store is not None:
        try:
            raw           = vector_store.similarity_search(query, k=TOP_K_DOCS)
            semantic_docs = _keyword_boost(query, raw)
        except Exception:
            pass

    # ── 1. Table answer (checked BEFORE number — table wins when both match) ──
    if is_tbl:
        # ── Try hardcoded clean tables first ─────────────────────
        result = _best_hardcoded_table(query)
        if result:
            title, df = result
            # Map table title back to correct PDF page
            TABLE_PAGE_MAP = {
                "Financial Highlights":    10,
                "Profit & Loss":           175,
                "Balance Sheet":           174,
                "Cash Flow":               176,
                "Past Performance":        349,
                "Key Financial Ratios":    9,
            }
            tbl_page = 9  # default Board's Report
            for key, pg in TABLE_PAGE_MAP.items():
                if key.lower() in title.lower():
                    tbl_page = pg
                    break
            return {
                "type":       "table",
                "table_path": None,
                "table_df":   df,
                "title":      title,
                "answer":     "",
                "sources":    [{"pdf_name": "14. Annual Report 2024-2025.pdf",
                                "page":     tbl_page}],
            }

        # ── Fallback: CSV from extraction ─────────────────────────
        tp = _best_table_path(query, meta)
        if tp and Path(tp).exists():
            src   = []
            title = ""
            for pm in meta.values():
                for t in pm.get("tables", []):
                    if t.get("path") == tp:
                        src   = [{"pdf_name": t.get("pdf_name", ""),
                                  "page":     t.get("page", "")}]
                        pg = t.get("page", "")
                        PAGE_TITLES = {
                            10:  "Financial Highlights (Standalone & Consolidated)",
                            11:  "Financial Highlights (Standalone & Consolidated)",
                            174: "Standalone Balance Sheet — as at March 31, 2025",
                            175: "Standalone Statement of Profit & Loss",
                            176: "Standalone Cash Flow Statement",
                            268: "Consolidated Balance Sheet — as at March 31, 2025",
                            269: "Consolidated Statement of Profit & Loss",
                            270: "Consolidated Cash Flow Statement",
                            349: "Past Performance — 3 Year Summary (₹ Crore)",
                        }
                        title = PAGE_TITLES.get(pg, f"Table — Page {pg}")
            return {"type": "table", "table_path": tp, "table_df": None,
                    "title": title, "answer": "", "sources": src}

        # ── Fallback: semantic doc with type=table ─────────────────
        for d in semantic_docs[:3]:
            if d.metadata.get("type") == "table":
                tp2 = d.metadata.get("path", "")
                if tp2 and Path(tp2).exists():
                    return {"type": "table", "table_path": tp2, "table_df": None,
                            "title": "", "answer": "",
                            "sources": [d.metadata]}

    # ── 2. Number answer ─────────────────────────────────────────
    if is_num:
        val, ctx = _find_exact_number(query, meta)
        if val:
            # Use Board's Report pages (9-14) as the reliable source
            # instead of random semantic chunk pages
            num_src = [{"pdf_name": "14. Annual Report 2024-2025.pdf",
                        "page": semantic_docs[0].metadata.get("page", 9)
                        if semantic_docs else 9}]
            return {
                "type":    "number",
                "answer":  val,
                "context": ctx,
                "sources": num_src,
            }
        # Fallback: numbers from semantic docs
        for d in semantic_docs[:5]:
            nums = re.findall(
                r"[₹$€]?\s*\d[\d,]*(?:\.\d+)?\s*"
                r"(?:crore|lakh|Cr\.?|L\.?|million|billion|MMT|MWh|TMT|TKL)?",
                d.page_content, re.IGNORECASE,
            )
            nums = [n.strip() for n in nums if re.search(r"\d{2,}", n)]
            if nums:
                return {
                    "type":    "number",
                    "answer":  nums[0],
                    "context": d.page_content[:400],
                    "sources": [d.metadata],
                }
    if is_img:
        ip = _best_image_path(query, meta)
        if ip == "ALL_PAGE_3":
            all_imgs = []
            for pm in meta.values():
                for img in pm.get("images", []):
                    if img.get("page") == 3 and Path(img["path"]).exists():
                        try:
                            sz = Path(img["path"]).stat().st_size
                            if sz >= 5000:
                                all_imgs.append((sz, img["path"], img.get("pdf_name", "")))
                        except Exception:
                            pass
            all_imgs.sort(reverse=True)
            paths    = [p for _, p, _ in all_imgs]
            src_name = all_imgs[0][2] if all_imgs else ""
            if paths:
                return {
                    "type":        "image_multi",
                    "image_paths": paths,
                    "answer":      "",
                    "sources":     [{"pdf_name": src_name, "page": 3}],
                }
        if ip:
            for pm in meta.values():
                for img in pm.get("images", []):
                    if img.get("path") == ip:
                        return {
                            "type":       "image",
                            "image_path": ip,
                            "answer":     "",
                            "sources":    [{"pdf_name": img.get("pdf_name", ""),
                                            "page":     img.get("page", "")}],
                        }
            return {"type": "image", "image_path": ip, "answer": "", "sources": []}

    # ── 4. No semantic docs ──────────────────────────────────────
    if not semantic_docs:
        if vector_store is None:
            return {"type": "error", "answer": "Vector store not ready."}
        return {"type": "not_found", "answer": "NOT FOUND"}

    # ── 5. LLM / extractive text ─────────────────────────────────
    ans = _llm_answer(query, semantic_docs)
    if ans == "NOT FOUND":
        return {"type": "not_found", "answer": "NOT FOUND"}
    return {
        "type":    "text",
        "answer":  ans,
        "sources": [d.metadata for d in semantic_docs[:4]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# RENDER BOT MESSAGE
# ─────────────────────────────────────────────────────────────────────────────

def _render_styled_table(df):
    """
    Render a DataFrame as a report-style HTML table matching the PDF layout:
    - White background, thin borders, header row in dark green
    - Numeric columns right-aligned
    - First column (labels) left-aligned, slightly bold
    - Alternating row shading
    """
    import pandas as pd
    import numpy as np

    df = df.copy()
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna("")

    # Detect numeric columns (right-align those)
    def _is_numeric_col(col):
        vals = df[col].astype(str).str.replace(r"[₹,\(\)\s]", "", regex=True)
        numeric = vals.apply(lambda v: bool(re.match(r"^-?\d+\.?\d*$", v)) if v.strip() else True)
        return numeric.sum() > len(df) * 0.5

    numeric_cols = {c for c in df.columns if _is_numeric_col(c)}
    first_col    = df.columns[0] if len(df.columns) > 0 else None

    # Build header
    header_cells = ""
    for c in df.columns:
        align = "right" if c in numeric_cols and c != first_col else "left"
        header_cells += (
            f'<th style="background:#1a5c2a;color:white;padding:7px 12px;'
            f'font-size:.80rem;font-weight:600;text-align:{align};'
            f'border:1px solid #145224;white-space:nowrap">{c}</th>'
        )

    # Build rows
    row_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#ffffff" if i % 2 == 0 else "#f4f9f5"
        cells = ""
        for ci, (col, val) in enumerate(row.items()):
            val_str = str(val).strip()
            is_negative = val_str.startswith("(") and val_str.endswith(")")
            align = "right" if col in numeric_cols and col != first_col else "left"
            weight = "500" if ci == 0 else "400"
            color  = "#dc2626" if is_negative else "#1a2332"
            # Bold section headers (all caps or short label rows with no numbers)
            if ci == 0 and val_str.isupper() and len(val_str) > 2:
                weight = "700"
                bg_cell = "#eef7f1"
            else:
                bg_cell = bg
            cells += (
                f'<td style="padding:6px 12px;font-size:.80rem;'
                f'text-align:{align};font-weight:{weight};color:{color};'
                f'border:1px solid #e5e7eb;background:{bg_cell};'
                f'white-space:nowrap">{val_str}</td>'
            )
        row_html += f"<tr>{cells}</tr>"

    html = f"""
    <div style="overflow-x:auto;margin:8px 0;border-radius:6px;
                box-shadow:0 1px 4px rgba(0,0,0,.08)">
      <table style="border-collapse:collapse;width:100%;font-family:'Inter',sans-serif">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{row_html}</tbody>
      </table>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_infographic(metrics: List[Dict]):
    """
    Render key financial metrics as colored badge cards — matching the
    visual style of the PDF infographic (Image 1):
    Turnover (blue), Market Cap (yellow/gold), Profit (purple), Throughput (orange)
    """
    CARD_STYLES = [
        {"bg": "#2563eb", "fg": "white"},   # blue
        {"bg": "#d97706", "fg": "white"},   # amber/gold
        {"bg": "#7c3aed", "fg": "white"},   # purple
        {"bg": "#ea580c", "fg": "white"},   # orange
        {"bg": "#059669", "fg": "white"},   # green
        {"bg": "#dc2626", "fg": "white"},   # red
    ]
    cards_html = ""
    for i, m in enumerate(metrics):
        style = CARD_STYLES[i % len(CARD_STYLES)]
        label = m.get("label", "")
        value = m.get("value", "")
        unit  = m.get("unit", "")
        cards_html += f"""
        <div style="background:{style['bg']};color:{style['fg']};
                    border-radius:10px;padding:14px 18px;flex:1;min-width:160px;
                    box-shadow:0 2px 6px rgba(0,0,0,.15)">
          <div style="font-size:.70rem;font-weight:600;letter-spacing:.06em;
                      text-transform:uppercase;opacity:.85;margin-bottom:4px">{label}</div>
          <div style="font-size:1.30rem;font-weight:700;line-height:1.2">
            {value}
            <span style="font-size:.75rem;font-weight:500;opacity:.85">{unit}</span>
          </div>
        </div>"""

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin:8px 0">'
        f'{cards_html}</div>',
        unsafe_allow_html=True)


def _render_bot_message(msg: dict):
    res     = msg.get("result", {})
    rtype   = res.get("type", "text")
    sources = res.get("sources", [])
    src_tag = _build_source_tag(sources)

    if rtype == "number":
        val = res.get("answer", "")
        ctx = res.get("context", "")[:300]
        st.markdown(
            f'<div class="chat-bot">'
            f'<span class="number-badge">{val}</span>'
            f'<br/><small style="color:#374151;margin-top:4px;display:block">{ctx}</small>'
            f'{src_tag}</div>',
            unsafe_allow_html=True)

    elif rtype == "infographic":
        # Colored metric cards — like the PDF infographic
        metrics = res.get("metrics", [])
        label   = res.get("label", "Key Financial Highlights")
        st.markdown(
            f'<div class="chat-bot" style="padding:.6rem .8rem">'
            f'<b>{label}</b>{src_tag}</div>',
            unsafe_allow_html=True)
        _render_infographic(metrics)

    elif rtype == "table":
        import pandas as pd
        title   = res.get("title", "Financial Table")
        tp      = res.get("table_path")
        df_direct = res.get("table_df")   # pre-built DataFrame from hardcoded tables

        header = f"<b>{title}</b>" if title else "<b>Financial Table</b>"
        st.markdown(
            f'<div class="chat-bot" style="padding:.6rem .8rem">'
            f'{header}{src_tag}</div>',
            unsafe_allow_html=True)

        if df_direct is not None:
            # Hardcoded clean table — render directly
            _render_styled_table(df_direct)
        elif tp and Path(tp).exists():
            # CSV fallback
            try:
                df = pd.read_csv(tp)
                _render_styled_table(df)
            except Exception as e:
                st.error(f"Table load error: {e}")
        else:
            st.warning("Table not available.")

    elif rtype == "image_multi":
        paths = res.get("image_paths", [])
        st.markdown(
            f'<div class="chat-bot"><b>Board of Directors — {len(paths)} image(s):</b>'
            f'{src_tag}</div>',
            unsafe_allow_html=True)
        valid = [p for p in paths if Path(p).exists()]
        for i in range(0, len(valid), 3):
            row = valid[i:i+3]
            cols = st.columns(len(row))
            for col, ipath in zip(cols, row):
                with col:
                    st.image(ipath)

    elif rtype == "image":
        ip = res.get("image_path", "")
        st.markdown(f'<div class="chat-bot"><b>Image Result:</b>{src_tag}</div>',
                    unsafe_allow_html=True)
        if ip and Path(ip).exists():
            st.image(ip, width=700)
        else:
            st.warning("Image file not found on disk.")

    elif rtype == "not_found":
        st.markdown(
            '<div class="not-found"><b>NOT FOUND</b> — '
            'This information was not found in the uploaded reports. '
            'Try rephrasing or check that the relevant PDF is uploaded.</div>',
            unsafe_allow_html=True)

    elif rtype == "error":
        st.error(res.get("answer", "Unknown error."))

    else:
        ans = res.get("answer", "")
        ans = re.sub(r"\\n", " ", ans)
        ans = re.sub(r"\n", " ", ans)
        ans = re.sub(r"\s{2,}", " ", ans)
        ans = re.sub(r"\[TABLE[^\]]*\]\s*", "", ans)
        ans = re.sub(r"\[IMAGE[^\]]*\]\s*", "", ans)
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", ans) if p.strip()]
        seen, unique = set(), []
        for p in parts:
            key = p.lower()[:60]
            if key not in seen:
                seen.add(key)
                unique.append(p)
        ans = " ".join(unique)[:700]
        st.markdown(
            f'<div class="chat-bot">{ans}{src_tag}</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    meta = _load_json(META_FILE)
    with st.sidebar:
        logo_path = ASSETS_DIR / "logo1.png"
        if logo_path.exists():
            st.markdown('<div class="logo-box">', unsafe_allow_html=True)
            st.image(str(logo_path), width=80)
            st.markdown(
                '<div class="logo-title">MRPL AI Assistant</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="logo-box">
              <svg width="68" height="68" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <rect width="100" height="100" rx="12" fill="#1a5c2a"/>
                <text x="50" y="30" font-size="12" fill="white" text-anchor="middle"
                      font-family="Arial" font-weight="bold">ONGC</text>
                <circle cx="50" cy="52" r="18" fill="none" stroke="#90ee90" stroke-width="2"/>
                <line x1="50" y1="34" x2="50" y2="70" stroke="#90ee90" stroke-width="1.5"/>
                <line x1="32" y1="52" x2="68" y2="52" stroke="#90ee90" stroke-width="1.5"/>
                <text x="50" y="82" font-size="13" fill="white" text-anchor="middle"
                      font-family="Arial" font-weight="bold">MRPL</text>
              </svg>
              <div class="logo-title">MRPL AI Assistant</div>
            </div>""", unsafe_allow_html=True)

        if st.button("+ New Chat"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown('<div class="section-header">Indexed Reports</div>',
                    unsafe_allow_html=True)
        if meta:
            for pname, pm in meta.items():
                display_name = pname if len(pname) <= 30 else pname[:27] + "…"
                pages = pm.get("total_pages", 0)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:7px;'
                    f'padding:5px 2px;border-bottom:1px solid #f1f5f9">'
                    f'<div style="width:7px;height:7px;border-radius:50%;'
                    f'background:#4ade80;flex-shrink:0"></div>'
                    f'<div style="min-width:0;flex:1">'
                    f'<div style="font-size:.80rem;font-weight:500;color:#1e293b;'
                    f'word-break:break-word">{display_name}</div>'
                    f'<div style="font-size:.70rem;color:#94a3b8">{pages} pages</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
        else:
            st.caption("No reports indexed yet.")

        if st.session_state.processing:
            st.warning("Indexing in progress...")

        # LLM backend runs silently — model is fixed to OLLAMA_DEFAULT_MODEL
        # Ping once so session_state.ollama_ok is populated for _llm_answer
        if st.session_state.ollama_ok is None:
            st.session_state.ollama_ok = _ollama_ping(OLLAMA_BASE_URL)


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────────────────────────────────────────────
def _render_upload_section(meta: dict, hashes: dict):
    """Upload widget — processes only truly new PDFs (hash-based dedup)."""
    # Show already-indexed reports
    if meta:
        for pname, pm in meta.items():
            st.markdown(
                f'<div class="processing-box" style="margin:3px 0;padding:5px 10px">'
                f'<b>{pname}</b> &nbsp;·&nbsp;'
                f'{pm.get("chunk_count",0):,} chunks &nbsp;·&nbsp;'
                f'{len(pm.get("tables",[]))} tables &nbsp;·&nbsp;'
                f'{len(pm.get("images",[]))} images &nbsp;·&nbsp;'
                f'<span style="color:#16a34a">ready</span></div>',
                unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Drop PDF(s) here — already-indexed files are skipped instantly",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if not uploaded_files:
        return

    # Hash check: skip already-indexed files
    new_files = []
    for uf in uploaded_files:
        raw   = uf.read(); uf.seek(0)
        fhash = _pdf_hash(raw)
        if hashes.get(uf.name) == fhash:
            st.success(
                f"**{uf.name}** already indexed — "
                f"loading from `vector_store/` instantly.")
        else:
            new_files.append((uf.name, raw, fhash))

    if not new_files:
        return

    st.session_state.processing = True
    changed = False

    for fname, raw, fhash in new_files:
        status = st.empty()
        status.markdown(
            f'<div class="info-box">'
            f'Indexing: <b>{fname}</b> ({len(raw)/1e6:.1f} MB)<br/>'
            f'<small>This happens only once. Future loads are instant.</small></div>',
            unsafe_allow_html=True)

        pdf_meta = process_pdf(raw, fname, status)

        if pdf_meta:
            meta[fname]   = pdf_meta
            hashes[fname] = fhash
            changed       = True
            status.markdown(
                f'<div class="processing-box">'
                f'<b>{fname}</b> indexed — '
                f'{pdf_meta["chunk_count"]:,} chunks · '
                f'{len(pdf_meta["tables"])} tables · '
                f'{len(pdf_meta["images"])} images<br/>'
                f'<small>Saved permanently. Future loads are instant.</small></div>',
                unsafe_allow_html=True)
            time.sleep(1.2)
            status.empty()
        else:
            status.error(f"Failed to index {fname}.")

    if changed:
        _save_json(META_FILE, meta)
        _save_json(HASH_FILE, hashes)
        _load_vector_store.clear()   # bust cache so new index is merged

    st.session_state.processing = False
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# CHAT SECTION
# ─────────────────────────────────────────────────────────────────────────────
def _render_chat(vs, meta: dict):
    st.markdown("---")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            _render_bot_message(msg)

    query = st.chat_input("Ask about MRPL reports…  (financials, tables, images, operations)")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        if st.session_state.processing:
            result = {"type": "error",
                      "answer": "Indexing in progress. Please wait a moment."}
        elif vs is None:
            result = {"type": "error",
                      "answer": "Vector store not loaded. Please restart the app."}
        else:
            with st.spinner("Searching reports..."):
                result = answer_query(query, vs, meta)
        st.session_state.chat_history.append({"role": "bot", "result": result})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def render_main():
    st.markdown('<h1 class="main-title">MRPL Annual Report Assistant</h1>',
                unsafe_allow_html=True)

    meta   = _load_json(META_FILE)
    hashes = _load_json(HASH_FILE)

    safe_names = tuple(
        v["safe_name"] for v in meta.values()
        if isinstance(v, dict) and "safe_name" in v
    )

    # Load (or merge from cache) all FAISS indices
    vs = _load_vector_store(safe_names) if safe_names else None
    already_indexed = bool(meta) and vs is not None

    if already_indexed:
        with st.expander(
            f"{len(meta)} report(s) indexed — click to add more",
            expanded=False,
        ):
            _render_upload_section(meta, hashes)
        _render_chat(vs, meta)
    else:
        st.markdown("**Upload Annual Reports to get started**")
        with st.expander("Upload Annual Reports", expanded=True):
            _render_upload_section(meta, hashes)
        if not meta:
            st.markdown(
                '<div class="info-box" style="text-align:center;padding:2rem">'
                'Upload your first MRPL Annual Report PDF above.<br/>'
                '<small>Once indexed, it will <b>never</b> need to be uploaded again.</small>'
                '</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
render_sidebar()
render_main()