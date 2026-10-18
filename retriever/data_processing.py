"""
Ingestion: PDFs -> chunks -> embeddings -> Chroma vector store.

Final setup (chosen by the retriever evaluation, version v7-k8):
  - table-aware loading with pdfplumber (tables kept whole as markdown)
  - "Company | Report | Page" header on every chunk
  - chunk 1000 / overlap 150, embeddings: bge-small

Run once:  python data_processing.py
"""
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
USE_TABLES = True      # pdfplumber table-aware loading (False = plain pypdf)
ADD_HEADER = True      # prepend "Company | Report | Page" to every chunk

COMPANIES = {"AAPL": "Apple", "AMZN": "Amazon", "INTC": "Intel", "MSFT": "Microsoft", "NVDA": "NVIDIA"}


def _meta(pdf_path, page_number):
    # File names look like "2023 Q3 NVDA.pdf"
    year, quarter, company = pdf_path.stem.split()
    return {"source": pdf_path.name, "company": company, "year": year,
            "quarter": quarter, "page": page_number}


# 1a. LOAD (plain) ---- pypdf, one Document per page
def load_reports():
    docs = []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata=_meta(pdf_path, page_number)))
    return docs


# 2. CHUNK ---- split pages into smaller overlapping pieces
def chunk_reports(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# 1b. LOAD (table-aware) ---- text chunked as usual, every table kept whole as markdown
def table_to_markdown(rows):
    rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in rows]
    rows = [["" if c == "$" else c for c in r] for r in rows]          # drop lone "$" cells
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] for r in rows)]       # drop empty columns
    rows = [[r[i] for i in keep] for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * len(keep)]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def split_table(markdown, max_chars, caption=""):
    """Split a long table by rows, repeating the caption + header row in every piece."""
    lines = markdown.split("\n")
    header, rows = ([caption] if caption else []) + lines[:2], lines[2:]
    pieces, current = [], []
    for row in rows:
        if current and len("\n".join(header + current + [row])) > max_chars:
            pieces.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join(header + current))
    return pieces


def load_and_chunk_with_tables(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    import pdfplumber

    text_pages, table_chunks = [], []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                meta = _meta(pdf_path, page_number)
                tables = page.find_tables()

                # text outside the tables
                text_page = page
                for t in tables:
                    try:
                        text_page = text_page.outside_bbox(t.bbox)
                    except ValueError:
                        pass
                text = text_page.extract_text() or ""
                if text.strip():
                    text_pages.append(Document(page_content=text, metadata={**meta, "kind": "text"}))

                # tables: the lines just above a table usually hold its column headers
                for t in tables:
                    md = table_to_markdown(t.extract())
                    if not md:
                        continue
                    top = t.bbox[1]
                    try:
                        caption = page.crop((0, max(0, top - 45), page.width, top)).extract_text() or ""
                    except ValueError:
                        caption = ""
                    caption = " ".join(caption.split())
                    for piece in split_table(md, max_chars=chunk_size * 2, caption=caption):
                        table_chunks.append(Document(page_content=piece, metadata={**meta, "kind": "table"}))

    return chunk_reports(text_pages, chunk_size, chunk_overlap) + table_chunks


# 3. HEADER ---- tell every chunk whose report it comes from
def add_header(chunk):
    m = chunk.metadata
    header = (f"Company: {COMPANIES.get(m['company'], m['company'])} ({m['company']}) | "
              f"Report: {m['year']} {m['quarter']} 10-Q | Page {m['page']}\n")
    return Document(page_content=header + chunk.page_content, metadata=m)


def build_chunks():
    """The final ingestion recipe: load -> chunk -> header."""
    if USE_TABLES:
        chunks = load_and_chunk_with_tables()
    else:
        chunks = chunk_reports(load_reports())
    if ADD_HEADER:
        chunks = [add_header(c) for c in chunks]
    return chunks


# 4. EMBED + STORE ---- turn chunks into vectors and save them in Chroma
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store():
    if DB_DIR.exists():
        print("Vector store already exists, skipping.")
        print("(Delete the 'chroma_store' folder to rebuild it.)")
        return

    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks")

    print("Embedding and saving (takes a few minutes)...")
    Chroma.from_documents(chunks, get_embeddings(), persist_directory=str(DB_DIR))
    print("Done! Saved to:", DB_DIR)


# RUN IT ---- python data_processing.py
if __name__ == "__main__":
    build_vector_store()