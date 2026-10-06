"""Step 3b: leak-free pseudo-labelling (self-training) for the joint SVR.

Inside every outer fold:
  1. fit the SVR on the fold's training clips only, predict the 216 test clips -> pseudo-labels
  2. refit on training clips + test clips with those pseudo-labels (sample weight W_PSEUDO)
  3. predict the validation fold (and the test set, averaged over folds as usual)
The pseudo-labels never come from a model that saw the validation clips, so the out-of-fold
score stays honest. Uses no test labels (none exist); test audio/text is not external data.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from final import design_matrices  # noqa: E402

W_PSEUDO = 0.5


def main(name="svr_all"):
    y, comps = design_matrices()
    make, (Xtr, Xte), dz = comps[name]
    oof, tp = np.zeros(len(y)), np.zeros(len(Xte))
    for tr, va in T.get_splits(y):
        if dz:
            tr = tr[y[tr] > 0]
        m1 = make().fit(Xtr[tr], y[tr])
        pseudo = np.clip(m1.predict(Xte), T.LO, T.HI)
        X2 = np.vstack([Xtr[tr], Xte])
        y2 = np.concatenate([y[tr], pseudo])
        w = np.concatenate([np.ones(len(tr)), np.full(len(Xte), W_PSEUDO)])
        m2 = make().fit(X2, y2, svr__sample_weight=w)
        oof[va] = m2.predict(Xtr[va])
        tp += m2.predict(Xte) / T.N_FOLDS
    oof, tp = np.clip(oof, T.LO, T.HI), np.clip(tp, T.LO, T.HI)
    out = f"{name}_pl"
    np.savez(T.PREDS / f"{out}.npz", oof=oof, test=tp, y=y)
    f = T.PREDS / f"{name}_fulltrain.npy"   # training-RMSE report: reuse the plain full fit
    if f.exists():
        np.save(T.PREDS / f"{out}_fulltrain.npy", np.load(f))
    print(out, {k: round(v, 4) for k, v in T.metrics(oof, y).items()}, flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "svr_all")
