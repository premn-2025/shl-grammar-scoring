"""Round 2, group A: post-processing, alternative stackers, KNN neighbour feature.

All judged with the paired rule against final_stack_w (same 8 stacker splits, short clips).
Post-processing parameters are fitted on 4/5 of each split's OOF predictions and scored on
the remaining 1/5 (second-level CV), so no step is tuned on the clips it is scored on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (must come first: imports lightgbm before sklearn)
import numpy as np  # noqa: E402
from sklearn.linear_model import BayesianRidge, HuberRegressor  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler, normalize  # noqa: E402
import gating as G  # noqa: E402
import stack as S  # noqa: E402

BASE = np.load(T.CACHE / "baseline_per_split.npy")
train, test = T.load_labels()
y = train.label.values
nz = np.where(y > 0)[0]
yy = y[nz]
dur = S.meta_frame("train")[nz, 0]
OUTS = G.stack_oof()                         # baseline per-split OOF of final_stack_w


def cv_post(fit_fn, apply_fn):
    """Second-level CV of a post-processing step on each split's OOF."""
    res = []
    for o in OUTS:
        p = o.copy()
        for a, b in KFold(5, shuffle=True, random_state=0).split(o):
            theta = fit_fn(o[a], yy[a], a)
            p[b] = np.clip(apply_fn(o[b], theta, b), 0, 5)
        res.append(p)
    return G.score(res)


def grid_fit(apply_fn, grid):
    def fit(o, t, idx):
        errs = [T.rmse(np.clip(apply_fn(o, g, idx), 0, 5), t) for g in grid]
        return grid[int(np.argmin(errs))]
    return fit


def main():
    # A1 prompt-aware shrinkage towards the mean for novel answers
    nov = G.novelty()[0][nz, 0]
    hi = nov > np.percentile(G.novelty()[0][:, 0], 80)
    m = yy.mean()
    shrink = lambda o, lam, idx: np.where(hi[idx], (1 - lam) * o + lam * m, o)
    G.compare(cv_post(grid_fit(shrink, [0, .05, .1, .2, .3]), shrink), BASE,
              "A1 shrink novel clips towards mean")

    # A2 duration-bucket clipping at the max training label of the bucket
    bins = np.array([0, 40, 50, 58, 99])
    bucket = np.digitize(dur, bins) - 1

    def fit_clip(o, t, idx):
        return {k: t[bucket[idx] == k].max() if (bucket[idx] == k).any() else 5 for k in range(4)}
    clip = lambda o, th, idx: np.minimum(o, np.array([th[k] for k in bucket[idx]]))
    G.compare(cv_post(fit_clip, clip), BASE, "A2 duration-bucket clipping")

    # A3 quantile alignment of predictions to the training label distribution
    def fit_q(o, t, idx):
        return (np.sort(o), np.sort(t))
    qmap = lambda o, th, idx: np.interp(o, th[0], th[1])
    G.compare(cv_post(fit_q, qmap), BASE, "A3 quantile alignment to label distribution")

    # A4 alternative meta-learners (same design matrix)
    from lightgbm import LGBMRegressor
    for name, mk in [("Huber", lambda: make_pipeline(StandardScaler(), HuberRegressor(alpha=1.0, max_iter=2000))),
                     ("BayesianRidge", lambda: make_pipeline(StandardScaler(), BayesianRidge())),
                     ("LightGBM depth 2", lambda: LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=2,
                                                                num_leaves=4, min_child_samples=40, reg_lambda=10,
                                                                subsample=0.8, subsample_freq=1, verbose=-1))]:
        G.compare(G.per_split(model=mk), BASE, f"A4 stacker = {name}")

    # A5 KNN neighbour feature on WavLM-large L18-23 (out-of-fold, same base folds)
    _, (W, Wte) = T.audio_pool("wavlm_large", list(range(18, 24)))
    sc = StandardScaler().fit(W)
    Wn, Wten = normalize(sc.transform(W)), normalize(sc.transform(Wte))

    def knn(train_idx, query, k=10):
        s = query @ Wn[train_idx].T
        top = np.argsort(-s, 1)[:, :k]
        w = np.take_along_axis(s, top, 1).clip(min=0) + 1e-6
        return (w * y[train_idx][top]).sum(1) / w.sum(1)

    for mode in ["random", "speaker"]:
        T.CV_MODE = mode
        oof = np.zeros(len(y))
        tp = np.zeros(len(Wte))
        for tr, va in T.get_splits(y):
            oof[va] = knn(tr, Wn[va])
            tp += knn(tr, Wten) / T.N_FOLDS
        folder = T.CACHE / ("preds" if mode == "random" else "preds_s")
        np.savez(folder / "knn_wavlm.npz", oof=oof, test=tp, y=y)
        print(f"knn_wavlm alone ({mode} folds):", {k: round(v, 4) for k, v in T.metrics(oof, y).items()})
    T.CV_MODE = "random"
    G.compare(G.per_split(G.BASE + ["knn_wavlm"]), BASE, "A5 + KNN neighbour feature (random folds)")
    base_s = G.per_split(folder=T.CACHE / "preds_s")
    G.compare(G.per_split(G.BASE + ["knn_wavlm"], folder=T.CACHE / "preds_s"), base_s,
              "A5 + KNN feature, SPEAKER-grouped OOF")


if __name__ == "__main__":
    main()
