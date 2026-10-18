"""
All retriever experiments live here.

- TESTS: one config per version (each version = previous version + ONE change)
- build_test_retriever(config): turns a config into a working retriever
- Quick check one version without the judge:
      python retriever_tests.py v3-tables "What was Apple's net sales in Q3 2023?"
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "retriever"))

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langsmith import traceable
from data_processing import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    load_reports, chunk_reports, load_and_chunk_with_tables, add_header, get_embeddings,
)
from retriever import detect_filter

STORES_DIR = PROJECT_DIR / "chroma_stores"      # one vector store per ingestion setup
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# =====================================================================
# 1. THE TESTS  (each version = previous version + one change)
# =====================================================================
BASE = {
    "loader": "pypdf",            # "pypdf" or "pdfplumber" (table-aware)
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "add_header": False,          # prepend "Company | Report | Page" to every chunk
    "k": 5,                       # chunks returned
    "metadata_filter": False,     # search only the company/quarter named in the question
    "hybrid": False,              # BM25 keyword search + dense search
    "rerank": False,              # cross-encoder re-sorts the candidates
    "fetch_k": 20,                # candidates fetched before hybrid/rerank/mmr
    "mmr": False,                 # MMR: pick relevant AND diverse chunks
    "embedding_model": EMBEDDING_MODEL,
}

TESTS = [
    {**BASE, "version": "v1-baseline",
     "change_note": "Baseline: bge-small, chunk 1000/150, Chroma L2, k=5"},
    {**BASE, "version": "v2-header", "add_header": True,
     "change_note": "Prepend company/report/page header to every chunk"},
    {**BASE, "version": "v3-tables", "add_header": True, "loader": "pdfplumber",
     "change_note": "Table-aware parsing: tables as markdown, kept whole (header row repeated)"},
    {**BASE, "version": "v4-filter", "add_header": True, "loader": "pdfplumber",
     "metadata_filter": True,
     "change_note": "Metadata filter: search only the company/quarter named in the question"},
    {**BASE, "version": "v5-rerank", "add_header": True, "loader": "pdfplumber",
     "metadata_filter": True, "rerank": True,
     "change_note": "Cross-encoder reranker: 20 candidates -> best 5"},

    # ---- extra versions: each one = v4-filter (best so far) + ONE change ----
    {**BASE, "version": "v6-mmr", "add_header": True, "loader": "pdfplumber",
     "metadata_filter": True, "mmr": True,
     "change_note": "v4 + MMR: 20 candidates -> 5 relevant but diverse chunks"},
    {**BASE, "version": "v7-k8", "add_header": True, "loader": "pdfplumber",
     "metadata_filter": True, "k": 8,
     "change_note": "v4 + k 5 -> 8 (more room for multi-chunk/multi-doc answers)"},
    {**BASE, "version": "v8-hybrid", "add_header": True, "loader": "pdfplumber",
     "metadata_filter": True, "hybrid": True,
     "change_note": "v4 + hybrid search: BM25 keywords + dense, fused (RRF)"},
    {**BASE, "version": "v9-chunk1500", "add_header": True, "loader": "pdfplumber",
     "metadata_filter": True, "chunk_size": 1500, "chunk_overlap": 200,
     "change_note": "v4 + bigger chunks: 1000/150 -> 1500/200 (more context per chunk)"},
    {**BASE, "version": "v10-chunk500", "add_header": True, "loader": "pdfplumber",
     "metadata_filter": True, "chunk_size": 500, "chunk_overlap": 100,
     "change_note": "v4 + smaller chunks: 1000/150 -> 500/100 (more focused chunks)"},
]

TESTS_BY_VERSION = {t["version"]: t for t in TESTS}


# =====================================================================
# 2. INGESTION: load -> chunk -> (header) -> vector store
#    (the loading/chunking/header code itself lives in retriever/data_processing.py)
# =====================================================================
def embeddings_for(config):
    if config["embedding_model"] == EMBEDDING_MODEL:
        return get_embeddings()
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=config["embedding_model"])


def store_dir_for(config):
    default_embedding = config["embedding_model"] == EMBEDDING_MODEL
    name = (f"{config['loader']}_c{config['chunk_size']}_o{config['chunk_overlap']}"
            f"_{'hdr' if config['add_header'] else 'nohdr'}")
    if not default_embedding:
        name += "_" + config["embedding_model"].split("/")[-1]   # e.g. ..._bge-base-en-v1.5
    return STORES_DIR / name


def load_or_build_store(config):
    path = store_dir_for(config)
    embeddings = embeddings_for(config)
    if path.exists():
        return Chroma(persist_directory=str(path), embedding_function=embeddings)

    print(f"Building vector store: {path.name} (one time, a few minutes)...")
    if config["loader"] == "pdfplumber":
        chunks = load_and_chunk_with_tables(config["chunk_size"], config["chunk_overlap"])
    else:
        chunks = chunk_reports(load_reports(), config["chunk_size"], config["chunk_overlap"])
    if config["add_header"]:
        chunks = [add_header(c) for c in chunks]
    print(f"  {len(chunks)} chunks")
    return Chroma.from_documents(chunks, embeddings, persist_directory=str(path))


# =====================================================================
# 3. RETRIEVAL HELPERS: BM25, fusion, reranker (metadata filter comes from retriever.py)
# =====================================================================
def matches(meta, flt):
    if flt is None:
        return True
    if "$and" in flt:
        return all(matches(meta, c) for c in flt["$and"])
    key, value = next(iter(flt.items()))
    return meta.get(key) in value["$in"] if isinstance(value, dict) else meta.get(key) == value


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(*ranked_lists, top_n):
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:200])
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            docs[key] = doc
    best = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [docs[key] for key in best]


# =====================================================================
# 4. BUILD A RETRIEVER FROM A CONFIG
# =====================================================================
def build_test_retriever(config):
    store = load_or_build_store(config)

    bm25 = corpus = None
    if config["hybrid"]:
        from rank_bm25 import BM25Okapi
        data = store.get(include=["documents", "metadatas"])
        corpus = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
        bm25 = BM25Okapi([tokenize(d.page_content) for d in corpus])

    reranker = None
    if config["rerank"]:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL)

    k = config["k"]
    n_candidates = config["fetch_k"] if (config["hybrid"] or config["rerank"]) else k

    @traceable(run_type="retriever", name=f"retriever-{config['version']}")
    def retrieve(question):
        flt = detect_filter(question) if config["metadata_filter"] else None

        def search(f):
            if config["mmr"]:   # relevant but not repetitive
                return store.max_marginal_relevance_search(
                    question, k=n_candidates, fetch_k=config["fetch_k"], lambda_mult=0.5, filter=f)
            return store.similarity_search(question, k=n_candidates, filter=f)

        docs = search(flt)
        if flt and not docs:                       # filter too strict -> search everything
            flt = None
            docs = search(None)

        if bm25 is not None:
            scores = bm25.get_scores(tokenize(question))
            keyword_ranked = sorted(
                (i for i in range(len(corpus)) if matches(corpus[i].metadata, flt)),
                key=lambda i: scores[i], reverse=True,
            )[:n_candidates]
            docs = reciprocal_rank_fusion(docs, [corpus[i] for i in keyword_ranked], top_n=n_candidates)

        if reranker is not None:
            pair_scores = reranker.predict([(question, d.page_content) for d in docs])
            docs = [d for _, d in sorted(zip(pair_scores, docs), key=lambda x: x[0], reverse=True)]

        return docs[:k]

    return retrieve


def settings_of(config):
    """The settings recorded in CSV / LangSmith for a version."""
    keys = ["embedding_model", "loader", "chunk_size", "chunk_overlap", "add_header", "k",
            "metadata_filter", "mmr", "hybrid", "rerank", "fetch_k"]
    return {key: config[key] for key in keys}


# =====================================================================
# QUICK CHECK:  python retriever_tests.py <version> "<question>"
# =====================================================================
if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v1-baseline"
    question = sys.argv[2] if len(sys.argv) > 2 else "What was Apple's total net sales in Q3 2023?"
    retrieve = build_test_retriever(TESTS_BY_VERSION[version])
    print(f"[{version}] {question}\n")
    for d in retrieve(question):
        m = d.metadata
        print(f"[{m['source']} | page {m['page']} | {m.get('kind', 'text')}] {d.page_content[:150]}...\n")