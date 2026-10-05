"""Step 3: interpretable features from transcript text + Whisper word timestamps.

Usage:  python src/features.py            -> cache/features.csv (full clips)
                                              cache/features_crop45.csv (first 45 s only)
                                              plots/feature_corr.png

Design notes
- Test clips are shorter than train (median 45 s vs 60 s), so we prefer *rates* (per minute,
  per 100 words, per sentence) over raw counts; raw counts are kept only for diagnosis.
- Because every word has a timestamp, "cropping to 45 s" is just dropping later words and
  recomputing -- no need to re-run Whisper.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
PLOTS = ROOT / "plots"

FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "eh"}
SUBORD_DEPS = {"mark", "advcl", "ccomp", "relcl", "xcomp", "acl"}

# LanguageTool rules/categories we ignore. ASR punctuation and capitalisation are produced
# by Whisper, not by the speaker, so "errors" there say nothing about the speaker's grammar.
IGNORED_CATEGORIES = {"PUNCTUATION", "CASING", "TYPOGRAPHY"}
IGNORED_RULES = {"UPPERCASE_SENTENCE_START", "WHITESPACE_RULE", "COMMA_PARENTHESIS_WHITESPACE",
                 "EN_QUOTES", "DOUBLE_PUNCTUATION", "PUNCTUATION_PARAGRAPH_END",
                 "SENTENCE_WHITESPACE", "EN_UNPAIRED_BRACKETS", "EN_UNPAIRED_QUOTES"}
LT_CATS = ["GRAMMAR", "TYPOS", "CONFUSED_WORDS", "STYLE", "REDUNDANCY", "COLLOCATIONS",
           "MISC", "SEMANTICS", "NONSTANDARD_PHRASES"]


def norm_tokens(text):
    return re.findall(r"[a-z']+", text.lower())


# ---------------------------------------------------------------- fluency (timestamps)
def fluency_features(words, duration):
    if not words:
        return {"wpm": 0.0, "articulation_wpm": 0.0, "pause_rate_pm": 0.0,
                "mean_pause": 0.0, "pause_time_frac": 0.0, "speech_ratio": 0.0,
                "long_pause_rate_pm": 0.0}
    starts = np.array([w[0] for w in words])
    ends = np.array([w[1] for w in words])
    span = max(ends[-1] - starts[0], 1e-3)                  # first word -> last word
    speech_time = float(np.sum(ends - starts))
    gaps = starts[1:] - ends[:-1]
    pauses = gaps[gaps > 0.5]                                # pause = gap > 0.5 s
    return {
        "wpm": len(words) / span * 60,                       # speech rate incl. pauses
        "articulation_wpm": len(words) / max(speech_time, 1e-3) * 60,  # rate while talking
        "pause_rate_pm": len(pauses) / span * 60,
        "mean_pause": float(pauses.mean()) if len(pauses) else 0.0,
        "pause_time_frac": float(pauses.sum()) / span,
        "long_pause_rate_pm": int((gaps > 1.5).sum()) / span * 60,
        "speech_ratio": speech_time / max(duration, 1e-3),
    }


# ---------------------------------------------------------------- disfluency (text)
def disfluency_features(text):
    toks = norm_tokens(text)
    n = max(len(toks), 1)
    fillers = sum(t in FILLERS for t in toks)
    like = sum(t == "like" for t in toks)
    you_know = sum(1 for a, b in zip(toks, toks[1:]) if a == "you" and b == "know")
    reps = sum(1 for a, b in zip(toks, toks[1:]) if a == b and a not in FILLERS)
    # Self-correction / restart: an n-gram (n=2,3) immediately repeated, e.g. "I was I was".
    restarts = 0
    for k in (2, 3):
        restarts += sum(1 for i in range(len(toks) - 2 * k + 1)
                        if toks[i:i + k] == toks[i + k:i + 2 * k])
    truncated = len(re.findall(r"\b\w+-(?=\s|$)", text))     # cut-off words like "wh-"
    return {
        "filler_per100": 100 * fillers / n,
        "like_per100": 100 * like / n,
        "youknow_per100": 100 * you_know / n,
        "repetition_per100": 100 * reps / n,
        "restart_per100": 100 * restarts / n,
        "truncated_per100": 100 * truncated / n,
    }


# ---------------------------------------------------------------- complexity (spaCy)
def depth(tok):
    # Compare indices, not identity: spaCy builds a new Token object on every .head access,
    # so `tok.head is not tok` would never become False at the root (infinite loop).
    d = 0
    while tok.head.i != tok.i:
        tok, d = tok.head, d + 1
    return d


def complexity_features(doc):
    content = [t for t in doc if t.is_alpha and t.lower_ not in FILLERS]
    n = max(len(content), 1)
    sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
    sent_lens = [sum(t.is_alpha for t in s) for s in sents] or [0]
    depths = [max(depth(t) for t in s) for s in sents] or [0]
    subord = sum(t.dep_ in SUBORD_DEPS for t in content)
    tenses = {(t.morph.get("Tense") or ["-"])[0] + "|" + (t.morph.get("VerbForm") or ["-"])[0]
              for t in content if t.pos_ in ("VERB", "AUX")}
    lemmas = [t.lemma_.lower() for t in content]
    # MATTR (moving-average type-token ratio, 40-word window) is length-robust, unlike TTR,
    # which falls as texts get longer -- important because test clips are shorter.
    win = 40
    if len(lemmas) >= win:
        mattr = float(np.mean([len(set(lemmas[i:i + win])) / win
                               for i in range(len(lemmas) - win + 1)]))
    else:
        mattr = len(set(lemmas)) / n
    pos = [t.pos_ for t in content]
    return {
        "n_words": len(content),
        "n_sents": len(sents),
        "mean_sent_len": float(np.mean(sent_lens)),
        "max_sent_len": float(np.max(sent_lens)),
        "mean_tree_depth": float(np.mean(depths)),
        "max_tree_depth": float(np.max(depths)),
        "subord_per_sent": subord / max(len(sents), 1),
        "subord_per100": 100 * subord / n,
        "tense_variety": len(tenses),
        "ttr": len(set(lemmas)) / n,
        "mattr": mattr,
        "mean_word_len": float(np.mean([len(t.text) for t in content])) if content else 0.0,
        "noun_frac": pos.count("NOUN") / n,
        "verb_frac": (pos.count("VERB") + pos.count("AUX")) / n,
        "adj_adv_frac": (pos.count("ADJ") + pos.count("ADV")) / n,
        "pron_frac": pos.count("PRON") / n,
        "conj_frac": (pos.count("CCONJ") + pos.count("SCONJ")) / n,
    }


# ---------------------------------------------------------------- grammar errors (LT)
def lt_features(tool, text, n_words):
    out = {f"lt_{c.lower()}_per100": 0.0 for c in LT_CATS}
    out["lt_other_per100"] = 0.0
    if not text.strip():
        out.update(lt_errors=0, lt_err_per100=0.0)
        return out
    matches = tool.check(text)
    kept = 0
    n = max(n_words, 1)
    for m in matches:
        cat = getattr(m, "category", "") or ""
        rule = getattr(m, "rule_id", None) or getattr(m, "ruleId", "")
        if cat in IGNORED_CATEGORIES or rule in IGNORED_RULES:
            continue
        kept += 1
        key = f"lt_{cat.lower()}_per100" if cat in LT_CATS else "lt_other_per100"
        out[key] += 100 / n
    out.update(lt_errors=kept, lt_err_per100=100 * kept / n)
    return out


# ---------------------------------------------------------------- whisper confidence
def asr_features(row, words):
    probs = np.array([w[3] for w in words]) if words else np.array([0.0])
    return {"avg_logprob": row.avg_logprob, "no_speech_prob": row.no_speech_prob,
            "mean_word_prob": float(probs.mean()), "low_prob_frac": float((probs < 0.5).mean())}


def build(transcripts, crop_s=None):
    import language_tool_python
    import spacy
    from tqdm import tqdm

    nlp = spacy.load("en_core_web_sm")
    tool = language_tool_python.LanguageTool("en-US")
    rows = []
    for r in tqdm(transcripts.itertuples(), total=len(transcripts),
                  desc=f"features crop={crop_s}"):
        words = json.loads(r.words)
        duration = r.duration
        text = r.text if isinstance(r.text, str) else ""
        if crop_s is not None and duration > crop_s:
            words = [w for w in words if w[1] <= crop_s]
            text = "".join(w[2] for w in words).strip()  # whisper words carry leading spaces
            duration = crop_s
        doc = nlp(text)
        f = {"filename": r.filename, "split": r.split, "duration": duration}
        f.update(complexity_features(doc))
        f.update(fluency_features(words, duration))
        f.update(disfluency_features(text))
        f.update(lt_features(tool, text, f["n_words"]))
        f.update(asr_features(r, words))
        rows.append(f)
    tool.close()
    return pd.DataFrame(rows)


def correlation_report(feats, labels, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = feats[feats.split == "train"].merge(labels, on="filename")
    # Exclude the label-0.0 batch: it is absent from test and would dominate correlations.
    df = df[df.label > 0]
    num = df.drop(columns=["filename", "split", "label"])
    num = num.loc[:, num.std() > 0]  # drop constant columns (undefined correlation)
    corr = pd.DataFrame({
        "pearson_label": num.corrwith(df.label),
        "spearman_label": num.corrwith(df.label, method="spearman"),
        # Drift check: if a feature correlates with clip duration, the shorter test clips
        # will shift it -- such features are risky.
        "pearson_duration": num.corrwith(df.duration),
    }).sort_values("pearson_label")
    test = feats[feats.split == "test"]
    corr["train_mean"] = num.mean()
    corr["test_mean"] = test[num.columns].mean()
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 11))
    c = corr.pearson_label.drop("duration", errors="ignore")
    ax.set_axisbelow(True)
    ax.barh(c.index, c.values, color=np.where(c.values > 0, "#2a78d6", "#eb6834"), height=0.7)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Pearson r with grammar label (train, non-zero labels)")
    ax.set_title("Feature–label correlation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    return corr


if __name__ == "__main__":
    tr = pd.read_csv(CACHE / "transcripts.csv")
    labels = pd.read_csv(ROOT / "data" / "train.csv")
    PLOTS.mkdir(exist_ok=True)
    full = build(tr)
    full.to_csv(CACHE / "features.csv", index=False)
    crop = build(tr, crop_s=45)
    crop.to_csv(CACHE / "features_crop45.csv", index=False)
    corr = correlation_report(full, labels, PLOTS / "feature_corr.png")
    corr.to_csv(CACHE / "feature_corr.csv")
