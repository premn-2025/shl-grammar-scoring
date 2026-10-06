"""Local LLM grammar judge through Ollama (runs on this laptop; transcripts never leave it).

For every verbatim Whisper transcript the model returns structured JSON (schema-constrained):
rubric grammar score 1-5 (half points), error counts by type (tense, subject-verb agreement,
articles/prepositions, word order, incomplete sentences), number of complex structures used,
and a confidence. Deterministic: temperature 0, fixed seed, thinking disabled.
Responses are cached in cache/llm_judge.jsonl (resumable); features in cache/features_llm_judge.csv.

Usage: python src/llm_judge_ollama.py [model]        default model: qwen3.5:9b
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "http://localhost:11434/api/chat"
ARGS = sys.argv[1:]
MODEL = ARGS[0] if ARGS and not ARGS[0].startswith("--") else "qwen3.5:9b"
TAG = ARGS[ARGS.index("--tag") + 1] if "--tag" in ARGS else ""
ANCHORED = "--anchors" in ARGS
SUFFIX = f"_{TAG}" if TAG else ""
CACHE = ROOT / "cache" / f"llm_judge{SUFFIX}.jsonl"
OUT = ROOT / "cache" / f"features_llm_judge{SUFFIX}.csv"
BASE_CACHE = ROOT / "cache" / "llm_judge.jsonl"   # zero-shot Qwen3.5-9B run (for anchor clips)
ANCHOR_LEVELS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

RUBRIC = """Grammar rubric (spoken English, 1-5):
1 = struggles with sentence structure, relies on memorised patterns, many errors.
2 = simple structures with consistent basic errors; incomplete sentences are common.
3 = decent grasp of grammar but noticeable errors in grammar or syntax.
4 = strong control of grammar; occasional minor errors, often self-corrected.
5 = high grammatical accuracy; complex structures handled well; self-corrects where needed."""

PROMPT = """You are an experienced English speaking examiner. Below is an automatic transcript of a
candidate's spoken answer (about 45-60 seconds). Fillers (um, uh), repetitions and cut-off words
come from the speech itself. Punctuation was added by the transcription system: ignore it.
Judge ONLY grammar - not pronunciation, accent, vocabulary choice, fluency or content.

{rubric}

Transcript:
\"\"\"{text}\"\"\"

Count the grammatical errors by type and the complex structures (subordinate/relative clauses,
conditionals, passives, perfect tenses) used correctly, then give the rubric score.

Answer with ONLY a JSON object with exactly these keys (no example values are given on purpose,
so that you are not anchored to them):
tense_errors (integer), agreement_errors (integer), article_preposition_errors (integer),
word_order_errors (integer), incomplete_sentences (integer), complex_structures (integer),
grammar_score (number from 1 to 5 in steps of 0.5), confidence (number from 0 to 1)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "tense_errors": {"type": "integer"},
        "agreement_errors": {"type": "integer"},
        "article_preposition_errors": {"type": "integer"},
        "word_order_errors": {"type": "integer"},
        "incomplete_sentences": {"type": "integer"},
        "complex_structures": {"type": "integer"},
        "grammar_score": {"type": "number"},
        "confidence": {"type": "number"},
    },
    "required": ["tense_errors", "agreement_errors", "article_preposition_errors",
                 "word_order_errors", "incomplete_sentences", "complex_structures",
                 "grammar_score", "confidence"],
}


def anchors():
    """Few-shot calibration examples: for each score level, the non-zero TRAINING clip whose
    transcript length is closest to 110 words (deterministic, label-blind within a level).
    Returns (prompt block, set of anchor filenames)."""
    tr = pd.read_csv(ROOT / "cache" / "transcripts.csv").fillna({"text": ""})
    lab = pd.read_csv(ROOT / "data" / "train.csv")
    d = tr[tr.split == "train"].merge(lab, on="filename")
    d = d[(d.duration >= 40)].assign(nw=d.text.str.split().str.len())
    blocks, names = [], set()
    for lv in ANCHOR_LEVELS:
        c = d[d.label == lv]
        if c.empty:
            continue
        r = c.iloc[(c.nw - 110).abs().argsort().iloc[0]]
        names.add(r.filename)
        blocks.append(f'Example rated {lv} by trained human raters:\n"""{" ".join(r.text.split()[:130])}"""')
    head = ("To calibrate your scale, here are real answers to similar tasks with the grammar "
            "score that trained human raters gave them:\n\n" + "\n\n".join(blocks) + "\n\n")
    return head, names


ANCHOR_HEAD, ANCHOR_NAMES = anchors() if ANCHORED else ("", set())


def judge(text):
    # anchors first (identical prefix for every clip -> Ollama reuses its prompt cache)
    content = ANCHOR_HEAD + PROMPT.format(rubric=RUBRIC, text=text)
    body = {"model": MODEL, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 6144 if ANCHORED else 4096},
            "messages": [{"role": "user", "content": content}]}
    if MODEL.startswith("qwen3"):
        body["think"] = False                       # Qwen3-family: no hidden reasoning
    last = None
    for attempt in range(3):
        try:
            r = requests.post(URL, json=body, timeout=300)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            m = re.search(r"\{.*\}", content, re.S)       # the JSON object, even if wrapped
            res = json.loads(m.group(0)) if m else {}
            missing = [k for k in SCHEMA["required"] if k not in res]
            if missing:
                raise ValueError(f"missing keys {missing}: {content[:200]}")
            return {k: res[k] for k in SCHEMA["required"]}
        except Exception as e:  # noqa: BLE001 - retry transient server/parse errors
            last = e
            time.sleep(2)
    raise last


def main():
    tr = pd.read_csv(ROOT / "cache" / "transcripts.csv").fillna({"text": ""})
    done = set()
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done.add((d["split"], d["filename"]))
    todo = [r for r in tr.itertuples() if (r.split, r.filename) not in done]
    print(f"{MODEL}: {len(done)} cached, {len(todo)} to go", flush=True)
    t0 = time.time()
    with open(CACHE, "a", encoding="utf-8") as f:
        for k, r in enumerate(todo):
            text = " ".join(r.text.split()[:400]) or "(no speech)"
            res = judge(text)
            f.write(json.dumps({"filename": r.filename, "split": r.split, "model": MODEL, **res}) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"{k}/{len(todo)}  {(time.time() - t0) / (k + 1):.1f} s/clip", flush=True)
    rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines()]
    if ANCHOR_NAMES:
        # an anchor clip was rated while its own label was in the prompt -> use its zero-shot
        # rating instead, so no label leaks into the features
        base = {(r["split"], r["filename"]): r for r in
                (json.loads(l) for l in BASE_CACHE.read_text(encoding="utf-8").splitlines())}
        rows = [({**base[("train", r["filename"])], "model": r["model"]}
                 if r["split"] == "train" and r["filename"] in ANCHOR_NAMES else r) for r in rows]
        print("anchor clips replaced with zero-shot ratings:", sorted(ANCHOR_NAMES))
    d = pd.DataFrame(rows)
    words = tr.set_index(["split", "filename"]).text.str.split().str.len().clip(lower=1)
    n = d.set_index(["split", "filename"]).index.map(lambda i: words.get(i, 1)).values
    feats = pd.DataFrame({"filename": d.filename, "split": d.split,
                          "llm_score": d.grammar_score.clip(0, 5), "llm_conf": d.confidence})
    for c in ["tense_errors", "agreement_errors", "article_preposition_errors", "word_order_errors",
              "incomplete_sentences", "complex_structures"]:
        feats[f"llm_{c}_per100"] = 100 * d[c].clip(lower=0) / n
    feats["llm_errors_per100"] = feats[[c for c in feats.columns if c.endswith("errors_per100")]].sum(1)
    feats.to_csv(OUT, index=False)
    print("wrote", OUT, len(feats))


if __name__ == "__main__":
    main()
