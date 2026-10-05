"""Step 7: weighted blend of saved OOF predictions.

Weights are non-negative and sum to 1, chosen to minimise OOF RMSE (scipy SLSQP). With only
a handful of models this is a low-capacity fit (k-1 free parameters on 769 points), so the
optimism from tuning weights on OOF predictions is small; we also report the equal-weight
average for comparison.

Usage:  python src/ensemble.py model_a model_b [model_c ...] [--out final]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import LO, HI, PREDS, rmse, write_submission  # noqa: E402


def blend(names, verbose=True):
    """Weights are fitted on the non-zero clips only, because the test set has no 0.0 clips."""
    P = [np.load(PREDS / f"{n}.npz") for n in names]
    y = P[0]["y"]
    nz = y > 0
    oof = np.column_stack([p["oof"] for p in P])
    test = np.column_stack([p["test"] for p in P])
    k = len(names)
    res = minimize(lambda w: rmse(oof[nz] @ w, y[nz]), np.full(k, 1 / k), method="SLSQP",
                   bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w = res.x
    b = np.clip(oof @ w, LO, HI)
    if verbose:
        for i, (n, wi) in enumerate(zip(names, w)):
            print(f"  {n:50s} weight {wi:.3f}  (alone {rmse(oof[nz, i], y[nz]):.4f})")
        eq = oof.mean(1)
        print(f"equal-weight blend : RMSE(nz) {rmse(eq[nz], y[nz]):.4f}")
        print(f"optimised blend    : RMSE(nz) {rmse(b[nz], y[nz]):.4f}  "
              f"Pearson(nz) {pearsonr(b[nz], y[nz])[0]:.4f}")
    return w, b, np.clip(test @ w, LO, HI)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    w, oof, test = blend(args)
    if out:
        np.savez(PREDS / f"{out}.npz", oof=oof, test=test, y=np.load(PREDS / f"{args[0]}.npz")["y"])
        write_submission(out, test)
