
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_chroma import Chroma

from data_processing import DB_DIR, get_embeddings
from langchain_chroma import Chroma

# 1. OPEN ---- load the saved vector store (no re-embedding)
def load_store():
    if not DB_DIR.exists():
        raise FileNotFoundError("No vector store found. Run data_processing.py first.")
    return Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())


# 2. RETRIEVER ---- returns the top k most similar chunks
def build_retriever(k=5):
    return load_store().as_retriever(search_kwargs={"k": k})


# 3. TRY IT ---- python retriever.py
if __name__ == "__main__":
    retriever = build_retriever()

    question = "What was Apple's total net sales in the third quarter of 2022?"
    results = retriever.invoke(question)

    print("Question:", question, "\n")
    for r in results:
        m = r.metadata
        print(f"[{m['source']} | page {m['page']}] {r.page_content[:150]}...\n")