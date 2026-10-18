from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # loads settings from .env (e.g. LangSmith)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_DIR / "data" / "docs"
DB_DIR = PROJECT_DIR / "chroma_store"

# ---- Settings (the single place to change them) ----
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# 1. LOAD ---- read every PDF, one Document per page
def load_reports():
    docs = []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        # File names look like "2023 Q3 NVDA.pdf"
        year, quarter, company = pdf_path.stem.split()

        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue  # skip empty pages

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "company": company,
                    "year": year,
                    "quarter": quarter,
                    "page": page_number,
                },
            ))
    return docs


# 2. CHUNK ---- split pages into smaller overlapping pieces
def chunk_reports(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# 3. EMBED + STORE ---- turn chunks into vectors and save them in Chroma
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store():
    if DB_DIR.exists():
        print("Vector store already exists, skipping.")
        print("(Delete the 'chroma_store' folder to rebuild it.)")
        return

    pages = load_reports()
    print(f"Loaded {len(pages)} pages")

    chunks = chunk_reports(pages)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding and saving (takes a few minutes)...")
    Chroma.from_documents(chunks, get_embeddings(), persist_directory=str(DB_DIR))
    print("Done! Saved to:", DB_DIR)


# RUN IT ---- python data_processing.py
if __name__ == "__main__":
    build_vector_store()