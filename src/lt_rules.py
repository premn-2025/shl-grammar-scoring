"""Experiment 3: fine-grained grammar error types from LanguageTool rule IDs.

Our first feature set only kept broad LanguageTool categories (GRAMMAR, TYPOS, ...). Here we
keep every match's rule ID, then build per-100-word rates for
 (a) the most frequent individual rules in the corpus (data-driven, e.g. agreement rules), and
 (b) linguistically named groups (agreement, tense/verb form, article/determiner, preposition,
     pronoun, plural, word order) by keyword-matching rule IDs.
Punctuation/casing/typography rules are still excluded (ASR punctuation is not the speaker's).
Output: cache/lt_rules.csv (raw matches) and cache/features_errtype.csv.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
IGNORED_CATEGORIES = {"PUNCTUATION", "CASING", "TYPOGRAPHY"}
GROUPS = {
    "agreement": r"AGREEMENT|AGR|SVA|NON3PRS|HE_VERB|PERS_PRON|SINGULAR|DOES_X|THIS_NNS|THESE_NN",
    "tense_verbform": r"TENSE|PAST|VBZ|VBD|VBG|VBN|BASEFORM|HAVE_PART|BEEN_PART|MD_|DID_|TO_NON_BASE|GERUND|PARTICIPLE",
    "article_det": r"(^|_)AN?_|ARTICLE|DT_|DETERMINER|THE_|A_VS_AN|MISSING_DET",
    "preposition": r"PREP|(^|_)(IN|ON|AT|TO|FOR|OF)_",
    "pronoun": r"PRP|PRONOUN|(^|_)I_|IT_IS|THEY_",
    "plural": r"PLURAL|NNS|NOUN_NUMBER",
    "word_order": r"ORDER|POSITION",
    "repetition_style": r"REP|DUPLICATE|ENGLISH_WORD_REPEAT|WORD_REPEAT",
}


def collect():
    import language_tool_python
    from tqdm import tqdm
    tr = pd.read_csv(CACHE / "transcripts.csv").fillna({"text": ""})
    tool = language_tool_python.LanguageTool("en-US")
    rows = []
    for r in tqdm(tr.itertuples(), total=len(tr)):
        if not r.text.strip():
            continue
        for m in tool.check(r.text):
            cat = getattr(m, "category", "") or ""
            rule = getattr(m, "rule_id", None) or getattr(m, "ruleId", "")
            if cat in IGNORED_CATEGORIES:
                continue
            rows.append({"filename": r.filename, "split": r.split, "rule": rule, "category": cat})
    tool.close()
    out = pd.DataFrame(rows)
    out.to_csv(CACHE / "lt_rules.csv", index=False)
    return out


def build(top_k=25):
    m = pd.read_csv(CACHE / "lt_rules.csv")
    f = pd.read_csv(CACHE / "features.csv")[["filename", "split", "n_words"]]
    n = f.set_index(["filename", "split"]).n_words.clip(lower=1)
    top = [r for r, _ in Counter(m[m.split == "train"].rule).most_common(top_k)]
    feats = pd.DataFrame(index=n.index)
    for r in top:
        c = m[m.rule == r].groupby(["filename", "split"]).size()
        feats[f"rule_{r}"] = (100 * c.reindex(n.index).fillna(0) / n).values
    for g, pat in GROUPS.items():
        hit = m[m.rule.str.contains(pat, regex=True)]
        c = hit.groupby(["filename", "split"]).size()
        feats[f"err_{g}"] = (100 * c.reindex(n.index).fillna(0) / n).values
    feats = feats.reset_index()
    feats.to_csv(CACHE / "features_errtype.csv", index=False)
    lab = pd.read_csv(ROOT / "data" / "train.csv")
    d = feats[feats.split == "train"].merge(lab, on="filename")
    d = d[d.label > 0]
    cols = [c for c in feats.columns if c.startswith(("err_", "rule_"))]
    corr = d[cols].corrwith(d.label).sort_values()
    print("top rules:", top[:12])
    print(corr.round(3).to_string())
    return feats


if __name__ == "__main__":
    if not (CACHE / "lt_rules.csv").exists():
        collect()
    build()
