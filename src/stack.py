"""Idea E: duration-aware Ridge stacking (alternative to the fixed-weight blend).

Meta-features per clip: each component's OOF prediction p_k, clip duration, words/min, Whisper
mean word probability, and the interactions p_k * (duration - 55 s)/10. The interactions let
the meta-model trust components differently for short answers (test clips are mostly 45 s,
and short clips are the hardest: CV RMSE ~0.53 vs ~0.45 for full-length clips).

Honesty: the reported CV is NESTED (stacker fitted on 4/5 of non-zero clips' OOF predictions,
scored on 1/5), and it is also reported separately on short (<50 s) clips, which resemble the
test set. Trained on non-zero clips only (the test has no 0.0 batch).

Usage: python src/stack.py comp1,comp2,... --out final_stack
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

META = ["duration", "wpm", "mean_word_prob"]


def meta_frame(split):
    train, test = T.load_labels()
    names = train if split == "train" else test
    f = pd.read_csv(T.CACHE / "features.csv")
    return names[["filename"]].merge(f[f.split == split], how="left")[META].fillna(0).values


def design(P, meta):
    # Clamp to the train duration range (20-61 s): a few test clips are shorter (one is 6.5 s)
    # and the linear interaction terms must not extrapolate beyond what the stacker has seen.
    meta = meta.copy()
    meta[:, 0] = np.clip(meta[:, 0], 20, 61)
    dur = (meta[:, [0]] - 55) / 10
    return np.hstack([P, meta, P * dur])


def ridge():
    return RidgeCV(alphas=np.logspace(-3, 3, 25))


def main(names, out):
    Z = [np.load(T.PREDS / f"{n}.npz") for n in names]
    y = Z[0]["y"]
    nz = np.where(y > 0)[0]
    Xtr = design(np.column_stack([z["oof"] for z in Z]), meta_frame("train"))
    Xte = design(np.column_stack([z["test"] for z in Z]), meta_frame("test"))

    nested = np.zeros(len(nz))
    if T.CV_MODE == "group":   # hold out whole prompts in the stacker's own check too
        from sklearn.model_selection import GroupKFold
        p = pd.read_csv(T.CACHE / "prompts.csv")
        train, _ = T.load_labels()
        groups = train[["filename"]].merge(p[p.split == "train"], how="left").prompt.values[nz]
        splits = GroupKFold(5).split(nz, groups=groups)
    else:
        splits = KFold(5, shuffle=True, random_state=T.SEED).split(nz)
    for a, b in splits:
        nested[b] = np.clip(ridge().fit(Xtr[nz][a], y[nz][a]).predict(Xtr[nz][b]), T.LO, T.HI)
    short = meta_frame("train")[nz, 0] < 50
    res = {"nested_rmse_nz": T.rmse(nested, y[nz]),
           "nested_rmse_short": T.rmse(nested[short], y[nz][short]),
           "nested_rmse_long": T.rmse(nested[~short], y[nz][~short])}

    m = ridge().fit(Xtr[nz], y[nz])
    test = np.clip(m.predict(Xte), T.LO, T.HI)
    # training RMSE: stacker applied to the components' full-train (in-sample) predictions
    fulls = []
    for n in names:
        f = T.PREDS / f"{n}_fulltrain.npy"
        if n == "ft_roberta":
            fs = sorted(T.PREDS.glob("ft_roberta-base-CoLA_e4_s*_fulltrain.npy"))
            fulls.append(np.mean([np.load(x) for x in fs], 0) if fs else None)
        else:
            fulls.append(np.load(f) if f.exists() else None)
    if all(f is not None for f in fulls):
        ins = np.clip(m.predict(design(np.column_stack(fulls), meta_frame("train"))), T.LO, T.HI)
        res["train_insample"] = T.metrics(ins, y)
    res["alpha"] = float(m.alpha_)
    res["components"] = names
    cols = names + META + [f"{n}*dur" for n in names]
    res["coef"] = dict(zip(cols, map(float, m.coef_)))
    print(json.dumps(res, indent=2))
    (T.CACHE / f"{out}_report.json").write_text(json.dumps(res, indent=2))
    oof_full = np.zeros(len(y))
    oof_full[nz] = nested
    np.savez(T.PREDS / f"{out}.npz", oof=oof_full, test=test, y=y)
    T.write_submission(out, test)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0].split(","), a[a.index("--out") + 1] if "--out" in a else "final_stack")
