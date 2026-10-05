"""Idea C: nested tuning of the two strongest SVRs over C, epsilon and RBF width (gamma).

Outer: the usual 5 folds (honest score). Inner: 3-fold grid search on the training part only.
The chosen parameters are printed per outer fold - if they jump around wildly, the tuning is
fitting noise (a red flag for overfitting the CV) and we should prefer the simpler default.
Usage: python src/tune_svr.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from final import design_matrices  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402


def make(d):
    g = 1.0 / d   # = gamma "scale" after standardisation
    return GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                        {"svr__C": [1, 3, 10, 30], "svr__epsilon": [0.05, 0.15, 0.3],
                         "svr__gamma": [0.5 * g, g, 2 * g]},
                        cv=3, scoring="neg_root_mean_squared_error", n_jobs=4)


def main():
    y, comps = design_matrices()
    for name in ["svr_wavlm_large", "svr_all"]:
        _, (Xtr, Xte), _ = comps[name]
        d = Xtr.shape[1]
        oof, tp = np.zeros(len(y)), np.zeros(len(Xte))
        skf = StratifiedKFold(T.N_FOLDS, shuffle=True, random_state=T.SEED)
        for f, (tr, va) in enumerate(skf.split(Xtr, T.strat_bins(y))):
            m = make(d).fit(Xtr[tr], y[tr])
            p = m.best_params_
            print(f"{name} fold {f}: C={p['svr__C']} eps={p['svr__epsilon']} "
                  f"gamma={p['svr__gamma'] * d:.1f}/d", flush=True)
            oof[va] = m.predict(Xtr[va])
            tp += m.predict(Xte) / T.N_FOLDS
        oof, tp = np.clip(oof, T.LO, T.HI), np.clip(tp, T.LO, T.HI)
        out = f"{name}_tuned"
        np.savez(T.PREDS / f"{out}.npz", oof=oof, test=tp, y=y)
        full = make(d).fit(Xtr, y)
        np.save(T.PREDS / f"{out}_fulltrain.npy", np.clip(full.predict(Xtr), T.LO, T.HI))
        print(out, {k: round(v, 4) for k, v in T.metrics(oof, y).items()},
              "| full-data choice:", full.best_params_, flush=True)


if __name__ == "__main__":
    main()
