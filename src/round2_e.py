"""Cross-ASR view (cache/features_ctc.csv from src/ctc_view.py) judged against final_stack_pw
under random, prompt-held-out and speaker-grouped CV (features winsorised to train range).
  X1 own Ridge component on the CTC/cross-ASR features
  X2 CTC features merged into the prosody/timing Ridge (replaces ridge_r2w_dz)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import gating as G  # noqa: E402
from round2_b import OUT, block  # noqa: E402
from round2_d import wins  # noqa: E402

CTC = T.CACHE / "features_ctc.csv"
NEW = G.BASE + ["ridge_r2w_dz"]


def main():
    train, _ = T.load_labels()
    y = train.label.values
    ca, cb, ccols = block(CTC)
    ra, rb, _ = block(OUT)
    nz = y > 0
    print("cross-ASR feature correlations (non-zero clips):")
    print(pd.DataFrame(ca[nz], columns=ccols).corrwith(pd.Series(y[nz])).round(3).to_string())
    C = wins(ca, cb)
    RC = wins(np.hstack([ra, ca]), np.hstack([rb, cb]))
    for mode, folder in [("random", "preds"), ("group", "preds_g"), ("speaker", "preds_s")]:
        T.CV_MODE = mode
        T.PREDS = T.CACHE / folder
        base = G.per_split(NEW, folder=T.PREDS)
        fit = T.cv_and_full if mode == "random" else (lambda *a, **k: T.run_cv(*a, verbose=False, **k))
        fit("ridge_ctc", T.ridge, C[0], y, C[1], drop_zero=True)
        fit("ridge_r2c", T.ridge, RC[0], y, RC[1], drop_zero=True)
        G.compare(G.per_split(NEW + ["ridge_ctc_dz"], folder=T.PREDS), base, f"[{mode}] X1 + cross-ASR component")
        G.compare(G.per_split(G.BASE + ["ridge_r2c_dz"], folder=T.PREDS), base,
                  f"[{mode}] X2 prosody+cross-ASR in one Ridge")


if __name__ == "__main__":
    main()
