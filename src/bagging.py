"""Step 1: repeated-CV bagging of the SVR/Ridge stack components.

Seed noise (different fold splits / seeds) is about as large as our recent gains, so we
average it away: each component is cross-validated on several different 5-fold splits
(seed 42 = the existing run, plus seeds 1-4). For every clip the out-of-fold predictions are
averaged over the repeats (each still comes from models that never saw that clip), and test
predictions are averaged over all fold models (5 repeats x 5 folds = 25 models; no full refit).
Output: cache/preds/<name>_bag.npz. The fine-tuned components keep their 3-seed averages.

Usage: python src/bagging.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from final import design_matrices  # noqa: E402

SEEDS = [1, 2, 3, 4]
BAG = T.CACHE / "preds_bag"


def matrices():
    y, comps = design_matrices()
    Wl = T.load_emb("whisper_layers")
    comps["svr_whisperL29_31"] = (T.svr, (Wl[0][:, 29:32].mean(1), Wl[1][:, 29:32].mean(1)), False)
    return y, comps


def main():
    y, comps = matrices()
    base_preds = T.PREDS
    for seed in SEEDS:
        out = BAG / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        # route run_cv to this split seed and output folder
        T.get_splits = lambda yy, s=seed: list(
            StratifiedKFold(T.N_FOLDS, shuffle=True, random_state=s).split(np.zeros(len(yy)), T.strat_bins(yy)))
        T.PREDS = out
        for name, (make, (Xtr, Xte), dz) in comps.items():
            if not (out / f"{name}{'_dz' if dz else ''}.npz").exists():
                print(f"seed {seed}:", end=" ", flush=True)
                T.run_cv(name, make, Xtr, y, Xte, drop_zero=dz)
    T.PREDS = base_preds
    for name, (_, _, dz) in comps.items():
        n = name + ("_dz" if dz else "")
        runs = [np.load(base_preds / f"{n}.npz")] + [np.load(BAG / f"seed{s}" / f"{n}.npz") for s in SEEDS]
        oof = np.mean([r["oof"] for r in runs], 0)
        test = np.mean([r["test"] for r in runs], 0)
        np.savez(base_preds / f"{n}_bag.npz", oof=oof, test=test, y=y)
        # bagged full-train file for the training-RMSE report = the single full fit (unchanged)
        f = base_preds / f"{n}_fulltrain.npy"
        if f.exists():
            np.save(base_preds / f"{n}_bag_fulltrain.npy", np.load(f))
        print(f"{n}_bag ({len(runs)} repeats)", {k: round(v, 4) for k, v in T.metrics(oof, y).items()})


if __name__ == "__main__":
    main()
