"""Round 2, group B: new interpretable features (prosody, timing, lexical, syntax).

prosody   librosa YIN pitch (60-400 Hz, 20 ms hop) on voiced frames: mean / std / range of
          f0 in semitones, std of frame-to-frame pitch change; RMS energy: mean, coefficient
          of variation, share of low-energy frames. Broken sentences often show pitch resets.
timing    short pauses (> 0.25 s) per minute and their share of time; tempo variation =
          std / mean of words per 5 s window (from Whisper word timestamps).
lexical   MTLD (measure of textual lexical diversity, threshold 0.72), length-robust.
syntax    mean dependency distance (MDD, a dependency-parse proxy for Yngve-style depth);
          fragment share = sentences whose root is not a verb/auxiliary.
Output: cache/features_round2.csv. Usage: python src/round2_b.py [--eval]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (lightgbm before sklearn)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "features_round2.csv"


def prosody(path):
    import librosa
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    f0 = librosa.yin(wav, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=320)
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=320)[0]
    voiced = (rms > np.percentile(rms, 30)) & (f0 > 60) & (f0 < 395)
    st = 12 * np.log2(f0[voiced] / 100.0) if voiced.sum() > 10 else np.zeros(2)
    return {"f0_mean_st": st.mean(), "f0_std_st": st.std(),
            "f0_range_st": np.percentile(st, 90) - np.percentile(st, 10),
            "f0_jump_std": np.diff(st).std() if len(st) > 2 else 0.0,
            "energy_mean": rms.mean(), "energy_cv": rms.std() / (rms.mean() + 1e-8),
            "low_energy_frac": (rms < 0.2 * np.median(rms)).mean()}


def timing(words, duration):
    if len(words) < 2:
        return {"short_pause_pm": 0.0, "short_pause_frac": 0.0, "tempo_cv": 0.0}
    s = np.array([w[0] for w in words])
    e = np.array([w[1] for w in words])
    span = max(e[-1] - s[0], 1e-3)
    gaps = s[1:] - e[:-1]
    sp = gaps[gaps > 0.25]
    bins = np.arange(s[0], e[-1] + 5, 5.0)
    counts = np.histogram(s, bins)[0] if len(bins) > 2 else np.array([len(words)])
    return {"short_pause_pm": len(sp) / span * 60, "short_pause_frac": sp.sum() / span,
            "tempo_cv": counts.std() / (counts.mean() + 1e-8)}


def mtld(tokens, thr=0.72):
    def one_way(toks):
        factors, types, n = 0.0, set(), 0
        for t in toks:
            n += 1
            types.add(t)
            if len(types) / n <= thr:
                factors += 1
                types, n = set(), 0
        if n:
            factors += (1 - len(types) / n) / (1 - thr)
        return len(toks) / factors if factors else len(toks)
    if len(tokens) < 10:
        return 0.0
    return (one_way(tokens) + one_way(tokens[::-1])) / 2


def build():
    import spacy
    from tqdm import tqdm
    nlp = spacy.load("en_core_web_sm")
    tr = pd.read_csv(ROOT / "cache" / "transcripts.csv").fillna({"text": ""})
    rows = []
    for r in tqdm(tr.itertuples(), total=len(tr)):
        f = {"filename": r.filename, "split": r.split}
        f.update(prosody(ROOT / "data" / r.split / r.filename))
        f.update(timing(json.loads(r.words), r.duration))
        doc = nlp(r.text)
        toks = [t.lower_ for t in doc if t.is_alpha]
        f["mtld"] = mtld(toks)
        dists = [abs(t.i - t.head.i) for t in doc if t.is_alpha and t.head.i != t.i]
        f["mean_dep_dist"] = float(np.mean(dists)) if dists else 0.0
        sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
        f["fragment_frac"] = (np.mean([s.root.pos_ not in ("VERB", "AUX") for s in sents])
                              if sents else 0.0)
        rows.append(f)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    return df


def block(path, cols=None):
    train, test = T.load_labels()
    f = pd.read_csv(path)
    cols = cols or [c for c in f.columns if c not in ("filename", "split")]
    a = train[["filename"]].merge(f[f.split == "train"], how="left")[cols].fillna(0).values
    b = test[["filename"]].merge(f[f.split == "test"], how="left")[cols].fillna(0).values
    return a, b, cols


def evaluate(path, tag):
    import gating as G
    train, _ = T.load_labels()
    y = train.label.values
    Ntr, Nte, cols = block(path)
    d = pd.DataFrame(Ntr, columns=cols)[y > 0].corrwith(pd.Series(y[y > 0]).reset_index(drop=True))
    print(f"correlations with label ({tag}, non-zero clips):")
    print(pd.DataFrame(Ntr[y > 0], columns=cols).corrwith(pd.Series(y[y > 0])).round(3).to_string())
    Htr, Hte, _ = T.hand_features()
    c = T.load_emb("cola")
    H = (np.hstack([Htr.values, c[0], Ntr]), np.hstack([Hte.values, c[1], Nte]))
    _, W = T.audio_pool("wavlm_large", list(range(18, 24)))
    E = T.load_emb("whisper_enc")
    D = T.load_emb("deberta")
    base = np.load(T.CACHE / "baseline_per_split.npy")
    B = G.BASE
    T.cv_and_full(f"svr_all_{tag}", T.svr, np.hstack([W[0], E[0], D[0], H[0]]), y,
                  np.hstack([W[1], E[1], D[1], H[1]]))
    T.cv_and_full(f"ridge_deb_hand_{tag}", T.ridge, np.hstack([D[0], H[0]]), y,
                  np.hstack([D[1], H[1]]), drop_zero=True)
    T.cv_and_full(f"ridge_{tag}", T.ridge, Ntr, y, Nte, drop_zero=True)
    G.compare(G.per_split([x if x != "svr_all" else f"svr_all_{tag}" for x in B]), base,
              f"{tag}: features inside joint SVR")
    G.compare(G.per_split([x if x != "ridge_deb_hand_dz" else f"ridge_deb_hand_{tag}_dz" for x in B]),
              base, f"{tag}: features inside text Ridge")
    G.compare(G.per_split(B + [f"ridge_{tag}_dz"]), base, f"{tag}: own Ridge component")
    meta = np.vstack([Ntr]).astype(float)
    G.compare(G.per_split(extra=(meta - meta.mean(0)) / (meta.std(0) + 1e-8)), base,
              f"{tag}: features as stacker meta-features")


def component():
    """Final-model component `ridge_r2w_dz`: Ridge on the round-2 features, each winsorised to
    the TRAINING 1st-99th percentile (test clips cannot push the linear model into
    extrapolation), trained without the 0.0 clips."""
    train, _ = T.load_labels()
    y = train.label.values
    a, b, _ = block(OUT)
    lo, hi = np.percentile(a, 1, axis=0), np.percentile(a, 99, axis=0)
    T.cv_and_full("ridge_r2w", T.ridge, np.clip(a, lo, hi), y, np.clip(b, lo, hi), drop_zero=True)


if __name__ == "__main__":
    if not OUT.exists():
        build()
    if "--eval" in sys.argv:
        evaluate(OUT, "r2")
    if "--component" in sys.argv:
        component()
