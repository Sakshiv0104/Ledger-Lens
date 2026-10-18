"""
Retrieval: question -> top-k chunks from the vector store.

Final setup (chosen by the retriever evaluation, version v7-k8):
  - metadata filter: search only the company / quarter named in the question
  - dense similarity search (Chroma, L2), k = 8

Try it:  python retriever.py
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import re

from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

from data_processing import DB_DIR, COMPANIES, get_embeddings

TOP_K = 8
USE_METADATA_FILTER = True


# 1. OPEN ---- load the saved vector store (no re-embedding)
def load_store():
    if not DB_DIR.exists():
        raise FileNotFoundError("No vector store found. Run data_processing.py first.")
    return Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())


# 2. FILTER ---- which company / quarter does the question name?
def detect_filter(question):
    found = [t for t, name in COMPANIES.items()
             if re.search(rf"\b({name}|{t})\b", question, re.IGNORECASE)]
    conditions = []
    if len(found) == 1:
        conditions.append({"company": found[0]})
    elif len(found) > 1:
        conditions.append({"company": {"$in": found}})

    m = re.search(r"\bQ([1-4])\s*(?:of\s*)?(20\d\d)\b|\b(20\d\d)\s*Q([1-4])\b", question, re.IGNORECASE)
    if m:
        quarter = m.group(1) or m.group(4)
        year = m.group(2) or m.group(3)
        conditions += [{"year": year}, {"quarter": f"Q{quarter}"}]

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


# 3. RETRIEVER ---- returns the top k chunks (works with .invoke like any LangChain retriever)
def build_retriever(k=TOP_K, use_filter=USE_METADATA_FILTER):
    store = load_store()

    def retrieve(question):
        flt = detect_filter(question) if use_filter else None
        docs = store.similarity_search(question, k=k, filter=flt)
        if flt and not docs:                       # filter too strict -> search everything
            docs = store.similarity_search(question, k=k)
        return docs

    return RunnableLambda(retrieve).with_config(run_name="retriever")


# TRY IT ---- python retriever.py
if __name__ == "__main__":
    retriever = build_retriever()
    question = "What was Apple's total net sales in Q3 2023?"
    print("Question:", question, "| filter:", detect_filter(question), "\n")
    for d in retriever.invoke(question):
        m = d.metadata
        print(f"[{m['source']} | page {m['page']} | {m.get('kind', 'text')}] {d.page_content[:150]}...\n")