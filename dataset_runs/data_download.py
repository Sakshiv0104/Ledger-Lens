import io
import zipfile
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
QNA_FILE = PROJECT_DIR / "goldens" / "qna_data.csv"

ZIP_URL = "https://github.com/docugami/KG-RAG-datasets/archive/refs/heads/main.zip"
PREFIX = "KG-RAG-datasets-main/sec-10-q/data/v1/"

# 1. Download the repo zip (kept in memory, not saved to disk)
print("Downloading dataset (this can take a minute)...")
response = requests.get(ZIP_URL)
response.raise_for_status()

# 2. Extract only the 20 PDFs and the golden CSV
DOCS_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    for member in z.namelist():
        if member.startswith(PREFIX + "docs/") and member.endswith(".pdf"):
            pdf_name = Path(member).name
            (DOCS_DIR / pdf_name).write_bytes(z.read(member))
        elif member == PREFIX + "qna_data.csv":
            QNA_FILE.write_bytes(z.read(member))

# 3. Show what we got
pdfs = sorted(DOCS_DIR.glob("*.pdf"))
print(f"\nSaved {len(pdfs)} PDFs to: {DOCS_DIR}")
for pdf in pdfs:
    print("  ", pdf.name)
print(f"\nSaved golden dataset to: {QNA_FILE}")