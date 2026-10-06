"""Gating / reliability meta-features for the stack ("which model to trust for which clip").

Extra per-clip meta-features (no labels used):
  novelty      1 - max cosine similarity of the clip's transcript embedding (mpnet) to the
               TRAINING clips (leave-one-out for train clips). High = answer unlike anything
               in training (new prompt).
  novelty_k    1 - mean of the top-5 similarities.
  disagree     std of the component predictions for that clip (model disagreement).
  asr_conf     Whisper mean word probability (already in the base meta-features).
Interactions component_pred x {novelty, disagree} let the linear stacker shift weight between
audio and text experts depending on how novel / contested a clip is - a linear gating model.

`evaluate(design_fn, names)` is the shared yardstick: Ridge stacker, 8 different 5-fold
splits of the non-zero clips, reports mean RMSE on all and on short (<50 s, test-like) clips.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import stack as S  # noqa: E402

BASE = ["svr_all", "svr_wavlm_large", "ridge_deb_hand_dz", "ft_roberta", "attnpool_wavlmL",
        "svr_whisperL29_31"]
SPLITS = [42, 7, 123, 2024, 99, 1, 2, 3]


def novelty():
    """(train (n,2), test (m,2)) novelty features from mpnet transcript embeddings."""
    Xtr, Xte = T.load_emb("mpnet")
    Xtr, Xte = normalize(Xtr), normalize(Xte)
    s_tr = Xtr @ Xtr.T
    np.fill_diagonal(s_tr, -1)            # leave-one-out: a clip is not its own neighbour
    s_te = Xte @ Xtr.T
    f = lambda s: np.column_stack([1 - s.max(1), 1 - np.sort(s, 1)[:, -5:].mean(1)])
    return f(s_tr), f(s_te)


def preds(names, split="oof", folder=None):
    P = folder or T.PREDS
    return np.column_stack([np.load(P / f"{n}.npz")[split] for n in names])


def evaluate(design_fn, names=BASE, label=""):
    train, _ = T.load_labels()
    y = train.label.values
    nz = np.where(y > 0)[0]
    short = S.meta_frame("train")[nz, 0] < 50
    X = design_fn(preds(names), "train")[nz]
    yy = y[nz]
    res = []
    for seed in SPLITS:
        o = np.zeros(len(nz))
        for a, b in KFold(5, shuffle=True, random_state=seed).split(nz):
            o[b] = np.clip(S.ridge().fit(X[a], yy[a]).predict(X[b]), 0, 5)
        res.append((T.rmse(o, yy), T.rmse(o[short], yy[short])))
    r = np.array(res)
    print(f"{label:45s} all {r[:, 0].mean():.4f}   SHORT {r[:, 1].mean():.4f} (sd {r[:, 1].std():.4f})",
          flush=True)
    return r.mean(0)


def stack_oof(names=BASE, design_fn=None, folder=None, model=None, extra=None):
    """Nested stacker OOF predictions for each of the 8 SPLITS: list of (8) arrays over the
    non-zero clips. model: stacker factory (default Ridge). extra: optional (n_train, k) block
    appended to the design."""
    design_fn = design_fn or base_design
    model = model or S.ridge
    train, _ = T.load_labels()
    y = train.label.values
    nz = np.where(y > 0)[0]
    X = design_fn(preds(names, folder=folder), "train")
    if extra is not None:
        X = np.hstack([X, extra])
    X, yy = X[nz], y[nz]
    outs = []
    for seed in SPLITS:
        o = np.zeros(len(nz))
        for a, b in KFold(5, shuffle=True, random_state=seed).split(nz):
            o[b] = np.clip(model().fit(X[a], yy[a]).predict(X[b]), 0, 5)
        outs.append(o)
    return outs


def score(outs):
    """(8, 2) array [short RMSE, all RMSE] for a list of per-split OOF arrays."""
    train, _ = T.load_labels()
    y = train.label.values
    nz = np.where(y > 0)[0]
    short = S.meta_frame("train")[nz, 0] < 50
    yy = y[nz]
    return np.array([(T.rmse(o[short], yy[short]), T.rmse(o, yy)) for o in outs])


def per_split(names=BASE, design_fn=None, folder=None, model=None, extra=None):
    """Short-clip and all-clip RMSE of the stacker for each of the 8 SPLITS (paired design)."""
    return score(stack_oof(names, design_fn, folder, model, extra))


def compare(cand, base=None, label=""):
    """Paired per-split comparison on SHORT clips (the LB compass).
    Keep rule: wins on >= 6 of 8 splits AND mean gain > std of the per-split differences."""
    base = per_split() if base is None else base
    d = base[:, 0] - cand[:, 0]            # positive = candidate better
    wins = int((d > 0).sum())
    keep = wins >= 6 and d.mean() > d.std()
    print(f"{label:42s} short {cand[:, 0].mean():.4f} vs {base[:, 0].mean():.4f} | gain {d.mean():+.4f} "
          f"(sd {d.std():.4f}) wins {wins}/8 | all {cand[:, 1].mean():.4f} -> {'KEEP' if keep else 'reject'}",
          flush=True)
    return keep


def base_design(P, split):
    return S.design(P, S.meta_frame(split))


_NOV = None


def gated_design(use_nov=True, use_dis=True, inter=True):
    def f(P, split):
        global _NOV
        if _NOV is None:
            _NOV = novelty()
        X = [S.design(P, S.meta_frame(split))]
        nov = _NOV[0] if split == "train" else _NOV[1]
        dis = P.std(1, keepdims=True)
        if use_nov:
            X.append(nov)
            if inter:
                X.append(P * nov[:, [0]] * 10)
        if use_dis:
            X.append(dis)
            if inter:
                X.append(P * dis)
        return np.hstack(X)
    return f


if __name__ == "__main__":
    evaluate(base_design, label="current final_stack_w design")
    evaluate(gated_design(True, False, False), label="+ novelty (main effect)")
    evaluate(gated_design(True, False, True), label="+ novelty x preds")
    evaluate(gated_design(False, True, False), label="+ disagreement (main effect)")
    evaluate(gated_design(False, True, True), label="+ disagreement x preds")
    evaluate(gated_design(True, True, True), label="+ novelty & disagreement, all interactions")
