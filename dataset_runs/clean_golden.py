import re
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_FILE = PROJECT_DIR / "goldens" / "qna_data.csv"
OUTPUT_FILE = PROJECT_DIR / "goldens" / "qna_data_clean.csv"

PDF = r"\d{4} Q\d [A-Z]+\.pdf"   # matches e.g. 2023 Q3 NVDA.pdf


# Some answers have broken characters like "Managementâ€™s" -> fix to "Management’s"
def fix_encoding(text):
    if "â€" in text:
        try:
            return text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text


# All PDF names mentioned anywhere in the answer
def extract_sources(text):
    found = re.findall(PDF, text)
    return ", ".join(sorted(set(found)))


# The answer with the source citations removed
def remove_sources(text):
    # 1. Lines starting with "SOURCE(S):" or "SOURCE:"
    text = re.sub(r"^\s*SOURCE(?:\(S\))?:.*$", "", text, flags=re.MULTILINE)
    # 2. Inline "(SOURCE: ...)"
    text = re.sub(r"\s*\(SOURCE[^)]*\)", "", text)
    # 3. Inline brackets holding only PDF names, e.g. ("2023 Q3 AAPL.pdf") or (from "2023 Q3 AAPL.pdf")
    text = re.sub(rf'\s*\((?:from\s+)?"*{PDF}"*(?:,\s*"*{PDF}"*)*\)', "", text)
    # 4. Tidy up empty lines and spaces left behind
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


qna = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
qna["Answer"] = qna["Answer"].fillna("").apply(fix_encoding)

qna["Answer Clean"] = qna["Answer"].apply(remove_sources)
qna["Answer Sources"] = qna["Answer"].apply(extract_sources)

qna.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"Saved {len(qna)} rows to: {OUTPUT_FILE}")
print("Columns:", list(qna.columns))
print("Rows with no source found:", (qna["Answer Sources"] == "").sum())

print("\n--- Example ---")
row = qna.iloc[0]
print("Answer Clean:\n", row["Answer Clean"])
print("\nAnswer Sources:", row["Answer Sources"])