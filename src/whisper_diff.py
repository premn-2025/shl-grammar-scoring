"""'What did Whisper fix?' features: verbatim pass (disfluent prompt) vs clean pass (no prompt).

Both transcripts are lower-cased word sequences; difflib aligns them and we count, per 100
verbatim words:
  diff_del_per100      words in verbatim that the clean pass dropped (fillers, repetitions,
                       false starts)
  diff_del_nonfill_per100   same, excluding filler words (um/uh/...) -> restarts, repairs
  diff_ins_per100      words the clean pass added (e.g. a missing article or auxiliary)
  diff_sub_per100      words replaced (e.g. "go" -> "went", "he don't" -> "he doesn't")
  diff_similarity      overall similarity ratio of the two sequences (1 = identical)
More fixing => weaker grammatical control. Output: cache/features_diff.csv
"""
import difflib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "eh"}


def words(t):
    return re.findall(r"[a-z']+", str(t).lower())


def diff_feats(verb, clean):
    a, b = words(verb), words(clean)
    n = max(len(a), 1)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    dele = dele_nf = ins = sub = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "delete":
            dele += i2 - i1
            dele_nf += sum(w not in FILLERS for w in a[i1:i2])
        elif op == "insert":
            ins += j2 - j1
        elif op == "replace":
            sub += max(i2 - i1, j2 - j1)
    return {"diff_del_per100": 100 * dele / n, "diff_del_nonfill_per100": 100 * dele_nf / n,
            "diff_ins_per100": 100 * ins / n, "diff_sub_per100": 100 * sub / n,
            "diff_similarity": sm.ratio()}


def main():
    v = pd.read_csv(ROOT / "cache" / "transcripts.csv").fillna({"text": ""})
    c = pd.read_csv(ROOT / "cache" / "transcripts_clean.csv").fillna({"text": ""})
    m = v.merge(c, on=["filename", "split"], suffixes=("_v", "_c"))
    f = pd.DataFrame([diff_feats(a, b) for a, b in zip(m.text_v, m.text_c)])
    f.insert(0, "split", m.split.values)
    f.insert(0, "filename", m.filename.values)
    f.to_csv(ROOT / "cache" / "features_diff.csv", index=False)
    lab = pd.read_csv(ROOT / "data" / "train.csv")
    d = f[f.split == "train"].merge(lab, on="filename")
    d = d[d.label > 0]
    print(d.drop(columns=["filename", "split", "label"]).corrwith(d.label).round(3).to_string())


if __name__ == "__main__":
    main()
