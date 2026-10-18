from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
GOLDENS_DIR = PROJECT_DIR / "goldens"
INPUT_FILE = GOLDENS_DIR / "qna_data_clean.csv"
OUTPUT_FILE = GOLDENS_DIR / "retriever_golden_20.csv"
SAMPLE_SIZE = 20
SEED = 42  # same seed -> same 30 questions every time

qna = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

# 1. How many questions to take from each question type (keeps the mix fair)
type_counts = qna["Question Type"].value_counts()
share = type_counts / len(qna) * SAMPLE_SIZE
per_type = share.astype(int)
# hand out the leftover slots to the types with the biggest remainders
leftover = SAMPLE_SIZE - per_type.sum()
for q_type in (share - per_type).sort_values(ascending=False).index[:leftover]:
    per_type[q_type] += 1

# 2. Randomly pick that many questions from each type
sampled = pd.concat([
    qna[qna["Question Type"] == q_type].sample(n=n, random_state=SEED)
    for q_type, n in per_type.items()
])

# 3. Shuffle so the types are mixed, then keep only the columns we need
sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)
golden = pd.DataFrame({
    "question": sampled["Question"],
    "ideal_answer": sampled["Answer Clean"],
    "sources": sampled["Answer Sources"],
    "question_type": sampled["Question Type"],
    "chunk_type": sampled["Source Chunk Type"],
})

golden.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"Saved {len(golden)} questions to: {OUTPUT_FILE}")
print("\nQuestion types:")
print(golden["question_type"].value_counts())
print("\nChunk types:")
print(golden["chunk_type"].value_counts())
print("\nFirst 3 questions:")
for q in golden["question"].head(3):
    print(" -", q)