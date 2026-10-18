"""
Runs the retriever tests from retriever_tests.py and records every result.

    python retriever_eval.py              -> runs every version not finished yet (in order)
    python retriever_eval.py v3-tables    -> runs only that version

Safe to stop (Ctrl+C) at any time: finished questions are saved after each one,
and the next run continues the same version from where it stopped.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import Client, evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from retriever_tests import TESTS, TESTS_BY_VERSION, build_test_retriever, settings_of

PROJECT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_DIR / ".env")

GOLDEN_FILE = PROJECT_DIR / "goldens" / "retriever_golden_20.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNS_LOG = RESULTS_DIR / "runs_log.csv"          # one line per finished version
STATE_FILE = RESULTS_DIR / "experiments.json"    # version -> LangSmith experiment id
DATASET_NAME = "ledger-lens-retriever-golden-20"

JUDGE_MODEL = "gpt-oss-120b"   # Cerebras free tier; keep the SAME judge for every version
THRESHOLD = 0.7
CALL_GAP = 12          # Cerebras free tier: 5 requests per minute -> one call every 12 seconds
MAX_RETRIES = 5
SCORES = ["contextual_recall", "contextual_precision", "source_hit"]

RESULTS_DIR.mkdir(exist_ok=True)


# ---------- retry: busy server / rate limit / broken JSON ----------
def with_retry(fn):
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(CALL_GAP)
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            msg = str(e)
            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
            if m:   # the provider tells us exactly how long to wait
                wait = int(m.group(1) or 0) * 60 + float(m.group(2)) + 5
            else:
                wait = 20 * attempt
            print(f"   ! {type(e).__name__}: {msg[:300]} -> retry in {wait}s")
            time.sleep(wait)


# ---------- the judge (Cerebras, OpenAI-compatible API) ----------
class Judge(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("CEREBRAS_API_KEY"), base_url="https://api.cerebras.ai/v1")

    def load_model(self):
        return self.client

    def generate(self, prompt, schema=None):
        messages = [{"role": "user", "content": prompt}]
        text = ""
        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0 if attempt == 0 else 0.3,   # only vary on a repair attempt
                reasoning_effort="low",                   # gpt-oss: think less -> fewer tokens
            )
            text = (response.choices[0].message.content or "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if not schema:
                return text
            start, end = text.find("{"), text.rfind("}")
            try:
                return schema.model_validate(json.loads(text[start:end + 1]))
            except Exception as e:
                # show the judge its broken answer and ask for corrected JSON only
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That was not valid JSON ({e}). "
                                                "Reply again with ONLY the corrected JSON, nothing else."},
                ]
        raise ValueError(f"Judge returned invalid JSON 3 times: {text[:200]}")

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


judge = Judge(JUDGE_MODEL)


# ---------- evaluators ----------
def make_test_case(inputs, outputs, reference_outputs):
    return LLMTestCase(
        input=inputs["question"],
        expected_output=reference_outputs["ideal_answer"],
        retrieval_context=outputs["chunks"],
        actual_output="(generator not evaluated)",
    )


def contextual_recall(inputs, outputs, reference_outputs):
    metric = ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=False, async_mode=False)
    with_retry(lambda: metric.measure(make_test_case(inputs, outputs, reference_outputs)))
    found = sum(v.verdict.strip().lower() == "yes" for v in metric.verdicts)
    return {"key": "contextual_recall", "score": metric.score,
            "comment": f"{found}/{len(metric.verdicts)} claims found"}


def contextual_precision(inputs, outputs, reference_outputs):
    metric = ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=False, async_mode=False)
    with_retry(lambda: metric.measure(make_test_case(inputs, outputs, reference_outputs)))
    ranks = " ".join("Y" if v.verdict.strip().lower() == "yes" else "N" for v in metric.verdicts)
    return {"key": "contextual_precision", "score": metric.score, "comment": f"ranks: {ranks}"}


def source_hit(outputs, reference_outputs):
    golden = {s.strip() for s in reference_outputs["sources"].split(",") if s.strip() and s != "nan"}
    if not golden:
        return {"key": "source_hit", "score": None, "comment": "no golden sources"}
    hit = golden & set(outputs["retrieved_pdfs"])
    return {"key": "source_hit", "score": len(hit) / len(golden), "comment": f"{len(hit)}/{len(golden)} PDFs found"}


# ---------- LangSmith dataset (created once) ----------
client = Client()
if not client.has_dataset(dataset_name=DATASET_NAME):
    goldens = pd.read_csv(GOLDEN_FILE, encoding="utf-8-sig")
    dataset = client.create_dataset(DATASET_NAME, description="20 seeded retriever goldens")
    for g in goldens.itertuples():
        client.create_example(
            dataset_id=dataset.id,
            inputs={"question": g.question},
            outputs={"ideal_answer": g.ideal_answer, "sources": str(g.sources)},
            metadata={"question_type": g.question_type, "chunk_type": g.chunk_type},
        )
EXAMPLES = list(client.list_examples(dataset_name=DATASET_NAME))


# ---------- bookkeeping: what is already done? ----------
def finished_versions():
    if not RUNS_LOG.exists():
        return set()
    return set(pd.read_csv(RUNS_LOG, encoding="utf-8-sig")["version"].astype(str))


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def row_from_result(r, config):
    feedback = {e.key: e for e in r["evaluation_results"]["results"]}
    meta = r["example"].metadata or {}
    row = {
        "version": config["version"],
        "question": r["example"].inputs["question"],
        "question_type": meta.get("question_type"),
        "chunk_type": meta.get("chunk_type"),
    }
    for key in SCORES:
        e = feedback.get(key)
        row[key] = e.score if e else None
        row[f"{key}_detail"] = e.comment if e else None
    row["retrieved"] = " | ".join((r["run"].outputs or {}).get("retrieved", []))
    row["change_note"] = config["change_note"]
    row.update(settings_of(config))
    row["judge_model"] = JUDGE_MODEL
    return row


# ---------- run one version (resumable) ----------
def run_version(config):
    version = config["version"]
    out_file = RESULTS_DIR / f"{version}.csv"
    done = set()
    if out_file.exists():
        done = set(pd.read_csv(out_file, encoding="utf-8-sig")["question"])
    todo = [ex for ex in EXAMPLES if ex.inputs["question"] not in done]
    print(f"\n========== {version}: {config['change_note']}")
    print(f"{len(done)} done, {len(todo)} to go")

    retrieve = build_test_retriever(config)

    def target(inputs):
        docs = retrieve(inputs["question"])
        return {
            "chunks": [d.page_content for d in docs],
            "retrieved": [f"{d.metadata['source']} p{d.metadata['page']}" for d in docs],
            "retrieved_pdfs": sorted({d.metadata["source"] for d in docs}),
        }

    state = load_state()
    metadata = {"version": version, "change_note": config["change_note"],
                "judge_model": JUDGE_MODEL, **settings_of(config)}

    for i, example in enumerate(todo, start=len(done) + 1):
        print(f"[{version} {i}/{len(EXAMPLES)}] {example.inputs['question'][:70]}")
        experiment_id = state.get(version)
        results = evaluate(
            target,
            data=[example],
            evaluators=[contextual_recall, contextual_precision, source_hit],
            experiment=experiment_id,                      # continue the same experiment
            experiment_prefix=None if experiment_id else version,
            description=config["change_note"],
            metadata=metadata,
            max_concurrency=0,
        )
        if not experiment_id:
            state[version] = str(results.experiment_id)
            save_state(state)

        rows = [row_from_result(r, config) for r in results]
        r = rows[0]
        if r["contextual_recall"] is None or r["contextual_precision"] is None:
            # a judge call failed: don't save it, so the next run retries this question
            print("   x judge failed on this question - not saved, will retry on next run")
            continue
        pd.DataFrame(rows).to_csv(out_file, mode="a", header=not out_file.exists(),
                                  index=False, encoding="utf-8-sig")
        print(f"   recall={r['contextual_recall']}  precision={r['contextual_precision']}  "
              f"source_hit={r['source_hit']}")

    summarize(config, out_file)


def summarize(config, out_file):
    df = pd.read_csv(out_file, encoding="utf-8-sig")
    print(f"\n===== SUMMARY {config['version']} =====")
    print("Overall averages:")
    print(df[SCORES].mean().round(3).to_string())
    print("\nBy question type:")
    print(df.groupby("question_type")[SCORES].mean().round(3).to_string())
    print("\nBy chunk type (table vs text):")
    print(df.groupby("chunk_type")[SCORES].mean().round(3).to_string())

    if config["version"] not in finished_versions() and len(df) >= len(EXAMPLES):
        log_row = {"version": config["version"], "change_note": config["change_note"],
                   "experiment_id": load_state().get(config["version"]),
                   **settings_of(config), "judge_model": JUDGE_MODEL,
                   **df[SCORES].mean().round(3).to_dict()}
        log = pd.read_csv(RUNS_LOG, encoding="utf-8-sig") if RUNS_LOG.exists() else pd.DataFrame()
        log = pd.concat([log, pd.DataFrame([log_row])], ignore_index=True)   # columns line up safely
        log.to_csv(RUNS_LOG, index=False, encoding="utf-8-sig")
        print(f"\nAdded {config['version']} to {RUNS_LOG.name}")


# ---------- main ----------
if __name__ == "__main__":
    requested = sys.argv[1:]
    for name in requested:
        if name not in TESTS_BY_VERSION:
            sys.exit(f"Unknown version '{name}'. Choose from: {', '.join(TESTS_BY_VERSION)}")

    to_run = [TESTS_BY_VERSION[v] for v in requested] if requested else TESTS
    finished = finished_versions()

    for config in to_run:
        if config["version"] in finished:
            print(f"Skipping {config['version']} (already finished, see {RUNS_LOG.name})")
            continue
        run_version(config)

    print("\nAll requested versions done.")