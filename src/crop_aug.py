"""Idea B: crop augmentation + test-time augmentation (TTA) for the WavLM-large SVR.

From the cached frames (frames.py, groups 16-19 and 20-23 averaged ~ our best layers 18-23)
we build, per clip, 9 "views": the full clip + 8 random windows of 30-45 s (fixed seed),
each summarised by mean+std over time.

Why: (1) 9x more training rows for a data-starved regressor, and the model learns that the
score should not depend on which stretch of speech it hears; (2) test clips are shorter
(median 45 s) than train (60 s) - windows bridge that gap; (3) averaging predictions over
views (TTA) reduces variance.

Leakage control: views of a clip always stay in that clip's fold (same 5 outer folds as
everything else), and the inner C search uses GroupKFold by clip.
Usage: python src/crop_aug.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (imports lightgbm first)
from frames import frame_path  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

CACHE = T.CACHE / "emb_crops.npz"
N_WIN, MIN_F, MAX_F = 8, 750, 1125   # 30-45 s at 25 frames/s


def pooled(x):
    return np.concatenate([x.mean(0), x.std(0)])


def build():
    rng = np.random.default_rng(42)
    train, test = T.load_labels()
    clips = [("train", f) for f in train.filename] + [("test", f) for f in test.filename]
    out = np.zeros((len(clips), N_WIN + 1, 2048), np.float32)
    for i, (split, f) in enumerate(clips):
        g = np.load(frame_path(split, f, "grp"), mmap_mode="r")
        x = np.asarray(g[1:3], dtype=np.float32).mean(0)      # (T, 1024)
        out[i, 0] = pooled(x)
        for k in range(N_WIN):
            L = int(rng.integers(MIN_F, MAX_F + 1))
            if x.shape[0] <= L:
                out[i, k + 1] = out[i, 0]
            else:
                s = int(rng.integers(0, x.shape[0] - L + 1))
                out[i, k + 1] = pooled(x[s:s + L])
        if i % 100 == 0:
            print("views", i, flush=True)
    np.savez(CACHE, X=out, split=np.array([c[0] for c in clips]))
    return out


def make_svr():
    return GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel="rbf", epsilon=0.1)),
                        {"svr__C": [0.3, 1, 3, 10]}, cv=GroupKFold(3),
                        scoring="neg_root_mean_squared_error")


def run(V_tr, V_te, y, name, aug=True, tta=True):
    """V_*: (n_clips, n_views, d). aug: train on all views; tta: predict = mean over views."""
    oof = np.zeros(len(y))
    tp = np.zeros(len(V_te))
    skf = StratifiedKFold(T.N_FOLDS, shuffle=True, random_state=T.SEED)
    nv = V_tr.shape[1]
    views = range(nv) if aug else [0]
    for tr, va in skf.split(V_tr[:, 0], T.strat_bins(y)):
        X = np.concatenate([V_tr[tr, v] for v in views])
        Y = np.concatenate([y[tr]] * len(views))
        G = np.concatenate([tr] * len(views))
        m = make_svr().fit(X, Y, groups=G)
        pv = (lambda V: np.mean([m.predict(V[:, v]) for v in range(nv)], 0)) if tta else \
             (lambda V: m.predict(V[:, 0]))
        oof[va] = pv(V_tr[va])
        tp += pv(V_te) / T.N_FOLDS
    oof, tp = np.clip(oof, T.LO, T.HI), np.clip(tp, T.LO, T.HI)
    np.savez(T.PREDS / f"{name}.npz", oof=oof, test=tp, y=y)
    print(f"{name:35s}", {k: round(v, 4) for k, v in T.metrics(oof, y).items()}, flush=True)
    return oof, tp, (views, tta)


def full_fit(V_tr, y, name, views, tta):
    X = np.concatenate([V_tr[:, v] for v in views])
    Y = np.concatenate([y] * len(views))
    G = np.concatenate([np.arange(len(y))] * len(views))
    m = make_svr().fit(X, Y, groups=G)
    nv = V_tr.shape[1]
    p = np.mean([m.predict(V_tr[:, v]) for v in range(nv)], 0) if tta else m.predict(V_tr[:, 0])
    np.save(T.PREDS / f"{name}_fulltrain.npy", np.clip(p, T.LO, T.HI))


def main():
    z = np.load(CACHE) if CACHE.exists() else None
    V = z["X"] if z is not None else build()
    train, _ = T.load_labels()
    y = train.label.values
    n = len(y)
    V_tr, V_te = V[:n], V[n:]
    run(V_tr, V_te, y, "svr_crops_base", aug=False, tta=False)   # sanity: ~ svr_wavlm_large
    run(V_tr, V_te, y, "svr_crops_tta", aug=False, tta=True)
    _, _, cfg = run(V_tr, V_te, y, "svr_crops_aug_tta", aug=True, tta=True)
    full_fit(V_tr, y, "svr_crops_aug_tta", *cfg)


if __name__ == "__main__":
    main()
