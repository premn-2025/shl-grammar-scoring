"""Step 4+: cross-validated models. Every model goes through the same `run_cv`, which
 - uses 5-fold StratifiedKFold on the rounded label (seed 42), so each fold sees the full
   label range (including the rare 0.0 and 1.x clips);
 - fits ALL preprocessing (scaling, alpha/C tuning) inside the training folds only;
 - returns out-of-fold (OOF) predictions for honest evaluation, plus test predictions
   averaged over the 5 fold models;
 - saves both to cache/preds/<name>.npz so the ensemble step can reuse them.

Usage:  python src/train.py hand        (hand-feature models, Step 4)
"""
import sys
from pathlib import Path

# Must be imported before scikit-learn: on this Windows setup, loading sklearn's native libs
# first makes LightGBM crash with an access violation (OpenMP runtime clash).
import lightgbm  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
PREDS = CACHE / "preds"
SUBS = ROOT / "submissions"
SEED, N_FOLDS = 42, 5
LO, HI = 0.0, 5.0  # competition score range

# Raw counts grow with clip length; test clips are shorter, so these would drift.
DRIFT_COLS = ["duration", "n_words", "n_sents", "lt_errors", "max_sent_len"]
ALPHAS = np.logspace(-2, 4, 30)


def load_labels():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    return train, test


def strat_bins(y):
    """Stratify on rounded label; merge the 1.x clips (only 4) into class 2 so every
    class has >= 5 members for 5 folds."""
    b = np.round(y).astype(int)
    b[b == 1] = 2
    return b


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def metrics(oof, y):
    """The 37 label-0.0 clips are a separate recording batch (ids 5037-5073) that the test set
    does not contain (see EXPERIMENTS.md), so the primary metric is RMSE on non-zero clips."""
    nz = y > 0
    return {"rmse_nz": rmse(oof[nz], y[nz]), "pearson_nz": pearsonr(oof[nz], y[nz])[0],
            "rmse_all": rmse(oof, y)}


def run_cv(name, make_model, X, y, X_test, fit_kw=None, verbose=True, drop_zero=False):
    """X, X_test: numpy arrays. make_model: () -> estimator with fit/predict.
    fit_kw: optional fn(X_tr, y_tr, X_va, y_va) -> dict of extra fit kwargs (e.g. LightGBM
    early stopping). drop_zero: exclude label-0.0 clips from the *training* folds (folds and
    evaluation are unchanged, so results are directly comparable)."""
    if drop_zero:
        name += "_dz"
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, strat_bins(y)):
        if drop_zero:
            tr = tr[y[tr] > 0]
        m = make_model()
        kw = fit_kw(X[tr], y[tr], X[va], y[va]) if fit_kw else {}
        m.fit(X[tr], y[tr], **kw)
        oof[va] = m.predict(X[va])
        test_pred += m.predict(X_test) / N_FOLDS
    oof, test_pred = np.clip(oof, LO, HI), np.clip(test_pred, LO, HI)
    res = {"model": name, **metrics(oof, y)}
    PREDS.mkdir(parents=True, exist_ok=True)
    np.savez(PREDS / f"{name}.npz", oof=oof, test=test_pred, y=y)
    if verbose:
        print(f"{name:45s} RMSE(non-zero) {res['rmse_nz']:.4f}  Pearson(nz) "
              f"{res['pearson_nz']:.4f}  | RMSE(all) {res['rmse_all']:.4f}", flush=True)
    return res, oof, test_pred


def write_submission(name, test_pred):
    _, test = load_labels()
    sub = test[["filename"]].copy()
    sub["label"] = np.clip(test_pred, LO, HI)
    SUBS.mkdir(exist_ok=True)
    path = SUBS / f"{name}.csv"
    sub.to_csv(path, index=False)
    print("wrote", path)
    return path


# ---------------------------------------------------------------- model factories
def ridge():
    # RidgeCV picks alpha by efficient leave-one-out *within the training fold* -> no leakage.
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def lgbm():
    from lightgbm import LGBMRegressor
    # Small, heavily regularised trees: 769 samples is tiny for boosting.
    return LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=15,
                         min_child_samples=20, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED,
                         verbose=-1)


def lgbm_es(X_tr, y_tr, X_va, y_va):
    # Early stopping on the fold's validation split (as specified). Caveat for the interview:
    # this makes OOF slightly optimistic since the stop point is chosen on the same data.
    from lightgbm import early_stopping
    return {"eval_set": [(X_va, y_va)], "callbacks": [early_stopping(100, verbose=False)]}


def hand_features(kind="features"):
    """Return (feature DataFrame for train in label order, for test in test order)."""
    train, test = load_labels()
    f = pd.read_csv(CACHE / f"{kind}.csv")
    ftr = train[["filename"]].merge(f[f.split == "train"], on="filename", how="left")
    fte = test[["filename"]].merge(f[f.split == "test"], on="filename", how="left")
    cols = [c for c in f.columns if c not in ["filename", "split"] + DRIFT_COLS]
    return ftr[cols].fillna(0), fte[cols].fillna(0), cols


def main_hand():
    train, _ = load_labels()
    y = train.label.values
    results = []
    for kind in ["features", "features_crop45"]:
        Xtr, Xte, cols = hand_features(kind)
        tag = "hand" if kind == "features" else "hand_crop45"
        results.append(run_cv(f"ridge_{tag}", ridge, Xtr.values, y, Xte.values)[0])
        results.append(run_cv(f"lgbm_{tag}", lgbm, Xtr.values, y, Xte.values, lgbm_es)[0])
    res = pd.DataFrame(results).sort_values("rmse_nz")
    print(res.to_string(index=False))
    return res


# ---------------------------------------------------------------- embeddings (Steps 5-6)
def svr():
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR
    # C tuned by an inner 3-fold CV on the *training fold only* (nested CV -> no leakage).
    return GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel="rbf", epsilon=0.1)),
                        {"svr__C": [0.3, 1, 3, 10]}, cv=3,
                        scoring="neg_root_mean_squared_error")


def load_emb(name, layer=None):
    """Return (X_train in label order, X_test in test order) for cache/emb_<name>.npz."""
    train, test = load_labels()
    z = np.load(CACHE / f"emb_{name}.npz", allow_pickle=True)
    X = z["X"] if layer is None else z["X"][:, layer]
    key = pd.Series(range(len(X)), index=[f"{s}/{f}" for s, f in zip(z["split"], z["filename"])])
    itr = key[["train/" + f for f in train.filename]].values
    ite = key[["test/" + f for f in test.filename]].values
    return X[itr], X[ite]


def main_emb():
    train, _ = load_labels()
    y = train.label.values
    Htr, Hte, _ = hand_features()
    sets = {}
    for name in ["mpnet", "deberta"]:
        if (CACHE / f"emb_{name}.npz").exists():
            sets[name] = load_emb(name)
    if (CACHE / "emb_wavlm.npz").exists():
        for layer in (6, 12):  # middle vs last transformer layer
            sets[f"wavlm_L{layer}"] = load_emb("wavlm", layer)
    if (CACHE / "emb_cola.npz").exists():
        c = load_emb("cola")
        Htr = np.hstack([Htr.values, c[0]]); Hte = np.hstack([Hte.values, c[1]])
        run_cv("ridge_hand_cola", ridge, Htr, y, Hte)
        run_cv("lgbm_hand_cola", lgbm, Htr, y, Hte, lgbm_es)
    else:
        Htr, Hte = Htr.values, Hte.values
    results = []
    for name, (Xtr, Xte) in sets.items():
        results.append(run_cv(f"ridge_{name}", ridge, Xtr, y, Xte)[0])
        results.append(run_cv(f"svr_{name}", svr, Xtr, y, Xte)[0])
        # Embeddings + hand features (all scaled inside the pipeline).
        results.append(run_cv(f"ridge_{name}+hand", ridge,
                              np.hstack([Xtr, Htr]), y, np.hstack([Xte, Hte]))[0])
        results.append(run_cv(f"svr_{name}+hand", svr,
                              np.hstack([Xtr, Htr]), y, np.hstack([Xte, Hte]))[0])
    res = pd.DataFrame(results).sort_values("rmse_nz")
    print(res.to_string(index=False))
    return res


def wavlm_layer_scan(name="wavlm"):
    """Ridge OOF RMSE per WavLM layer -- shows where grammar-relevant info lives."""
    train, _ = load_labels()
    y = train.label.values
    n_layers = np.load(CACHE / f"emb_{name}.npz", allow_pickle=True)["X"].shape[1]
    out = []
    for layer in range(n_layers):
        Xtr, Xte = load_emb(name, layer)
        out.append(run_cv(f"ridge_{name}_L{layer}", ridge, Xtr, y, Xte)[0])
    return pd.DataFrame(out)


def audio_pool(name, layers):
    """Average of several layers' mean-pooled vectors, optionally with std pooling appended."""
    mtr, mte = load_emb(name)
    str_, ste = load_emb(name + "_std")
    avg = lambda X: X[:, layers].mean(1)
    return (avg(mtr), avg(mte)), (np.hstack([avg(mtr), avg(str_)]), np.hstack([avg(mte), avg(ste)]))


def audio_variants(name, layers):
    train, _ = load_labels()
    y = train.label.values
    tag = f"{name}_L{layers[0]}-{layers[-1]}"
    (mtr, mte), (mstr, mste) = audio_pool(name, layers)
    for dz in (False, True):
        run_cv(f"ridge_{tag}_mean", ridge, mtr, y, mte, drop_zero=dz)
        run_cv(f"ridge_{tag}_meanstd", ridge, mstr, y, mste, drop_zero=dz)
        run_cv(f"svr_{tag}_meanstd", svr, mstr, y, mste, drop_zero=dz)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "hand"
    if cmd == "wavlm_scan":
        wavlm_layer_scan(sys.argv[2] if len(sys.argv) > 2 else "wavlm")
    elif cmd == "audio":  # python src/train.py audio wavlm 6 9
        audio_variants(sys.argv[2], list(range(int(sys.argv[3]), int(sys.argv[4]) + 1)))
    else:
        {"hand": main_hand, "emb": main_emb}[cmd]()
