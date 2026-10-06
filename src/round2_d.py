"""Round 2, B2: Whisper decoder-entropy features (cache/features_entropy.csv).

Two candidate uses, each judged against the final_stack_pw baseline under random,
prompt-held-out and speaker-grouped CV (all winsorised to the training 1st-99th percentile):
  E1 own Ridge component on the entropy features
  E2 entropy features merged into the round-2 prosody/timing Ridge (replaces ridge_r2w_dz)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import gating as G  # noqa: E402
from round2_b import OUT, block  # noqa: E402

ENT = T.CACHE / "features_entropy.csv"
NEW = G.BASE + ["ridge_r2w_dz"]


def wins(a, b):
    lo, hi = np.percentile(a, 1, axis=0), np.percentile(a, 99, axis=0)
    return np.clip(a, lo, hi), np.clip(b, lo, hi)


def main():
    train, _ = T.load_labels()
    y = train.label.values
    ea, eb, ecols = block(ENT)
    ra, rb, _ = block(OUT)
    nz = y > 0
    print("entropy feature correlations (non-zero clips):")
    print(pd.DataFrame(ea[nz], columns=ecols).corrwith(pd.Series(y[nz])).round(3).to_string())
    E = wins(ea, eb)
    RE = wins(np.hstack([ra, ea]), np.hstack([rb, eb]))
    for mode, folder in [("random", "preds"), ("group", "preds_g"), ("speaker", "preds_s")]:
        T.CV_MODE = mode
        T.PREDS = T.CACHE / folder
        if not (T.PREDS / "ridge_r2w_dz.npz").exists():   # prosody component in this CV mode
            wa, wb = wins(*block(OUT)[:2])
            T.run_cv("ridge_r2w", T.ridge, wa, y, wb, drop_zero=True, verbose=False)
        base = G.per_split(NEW, folder=T.PREDS)
        fit = T.cv_and_full if mode == "random" else (lambda *a, **k: T.run_cv(*a, verbose=False, **k))
        fit("ridge_ent", T.ridge, E[0], y, E[1], drop_zero=True)
        fit("ridge_r2e", T.ridge, RE[0], y, RE[1], drop_zero=True)
        G.compare(G.per_split(NEW + ["ridge_ent_dz"], folder=T.PREDS), base, f"[{mode}] E1 + entropy Ridge component")
        G.compare(G.per_split(G.BASE + ["ridge_r2e_dz"], folder=T.PREDS), base,
                  f"[{mode}] E2 prosody+entropy in one Ridge")


if __name__ == "__main__":
    main()
