"""Step 7: final model = weighted blend of 4 components + a linear "stretch" calibration.

Components (all 5-fold CV, OOF + fold-averaged test predictions):
  A  svr_all          SVR on [WavLM-large L18-23 mean+std | Whisper encoder | DeBERTa
                      embedding | hand features + CoLA]       (trained incl. 0.0 clips)
  B  svr_wavlm_large  SVR on WavLM-large L18-23 mean+std        (trained incl. 0.0 clips)
  C  ridge_deb_hand   Ridge on [DeBERTa embedding | hand+CoLA]  (text; trained w/o 0.0 clips)
  D  ft_roberta       fine-tuned roberta-base-CoLA, mean of seeds (text; trained w/o 0.0 clips)

Blend weights: non-negative, sum to 1, minimise OOF RMSE on non-zero clips.
Stretch: y ~ a + b * blend, fitted on non-zero OOF; corrects the shrink-to-mean of the
regularised models (b > 1). Its effect is estimated honestly with an inner 5-fold CV.

Also reports the compulsory TRAINING RMSE: every component refitted on the full training
set, predicting the training set in-sample, blended with the same weights + stretch.

Usage: python src/final.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from ensemble import blend  # noqa: E402

FT_TAG = "ft_roberta-base-CoLA_e4"
FT_SEEDS = [0, 1, 2]
# Optional extra components (saved by other scripts) and output name, e.g.
#   python src/final.py --extra attnpool_wavlmL --out final_v2
_a = sys.argv[1:]
EXTRA = _a[_a.index("--extra") + 1].split(",") if "--extra" in _a else []
OUT = _a[_a.index("--out") + 1] if "--out" in _a else "final"


def design_matrices():
    train, _ = T.load_labels()
    y = train.label.values
    Htr, Hte, _ = T.hand_features()
    c = T.load_emb("cola")
    H = (np.hstack([Htr.values, c[0]]), np.hstack([Hte.values, c[1]]))
    _, W = T.audio_pool("wavlm_large", list(range(18, 24)))
    E = T.load_emb("whisper_enc")
    D = T.load_emb("deberta")
    return y, {
        "svr_all": (T.svr, (np.hstack([W[0], E[0], D[0], H[0]]),
                            np.hstack([W[1], E[1], D[1], H[1]])), False),
        "svr_wavlm_large": (T.svr, W, False),
        "ridge_deb_hand": (T.ridge, (np.hstack([D[0], H[0]]), np.hstack([D[1], H[1]])), True),
    }


def build_components():
    y, comps = design_matrices()
    for name, (make, (Xtr, Xte), dz) in comps.items():
        res, _, _ = T.run_cv(name, make, Xtr, y, Xte, drop_zero=dz)
    # fine-tuned RoBERTa: average the seeds that exist
    seeds = [s for s in FT_SEEDS if (T.PREDS / f"{FT_TAG}_s{s}.npz").exists()]
    P = [np.load(T.PREDS / f"{FT_TAG}_s{s}.npz") for s in seeds]
    oof = np.mean([p["oof"] for p in P], 0)
    test = np.mean([p["test"] for p in P], 0)
    np.savez(T.PREDS / "ft_roberta.npz", oof=oof, test=test, y=y)
    print(f"ft_roberta (seeds {seeds})", T.metrics(oof, y))
    names = ["svr_all", "svr_wavlm_large", "ridge_deb_hand_dz", "ft_roberta"] + EXTRA
    return y, comps, names, seeds


def stretch_cv(p, y):
    out = np.zeros_like(p)
    for tr, va in KFold(5, shuffle=True, random_state=T.SEED).split(p):
        out[va] = np.polyval(np.polyfit(p[tr], y[tr], 1), p[va])
    return np.clip(out, T.LO, T.HI)


def training_rmse(y, comps, names, w, coef, seeds):
    """In-sample predictions of each component refitted on the full train set."""
    ins = {}
    for name, (make, (Xtr, _), dz) in comps.items():
        idx = np.where(y > 0)[0] if dz else np.arange(len(y))
        m = make().fit(Xtr[idx], y[idx])
        ins[name + ("_dz" if dz else "")] = np.clip(m.predict(Xtr), T.LO, T.HI)
    ft = [T.PREDS / f"{FT_TAG}_s{s}_fulltrain.npy" for s in seeds]
    ft = [f for f in ft if f.exists()]
    if not ft:
        print("!! no full-train fine-tune predictions; run FT_FULL=1 finetune_text.py first")
        return None
    ins["ft_roberta"] = np.mean([np.load(f) for f in ft], 0)
    for name in EXTRA:  # extra neural components save their own full-train predictions
        ins[name] = np.load(T.PREDS / f"{name}_fulltrain.npy")
    P = np.column_stack([ins[n] for n in names])
    return np.clip(np.polyval(coef, np.clip(P @ w, T.LO, T.HI)), T.LO, T.HI)


def main():
    y, comps, names, seeds = build_components()
    print("\nBlend:")
    w, oof_blend, test_blend = blend(names)
    nz = y > 0

    coef = np.polyfit(oof_blend[nz], y[nz], 1)  # stretch fitted on non-zero OOF
    oof_final_cv = oof_blend.copy()
    oof_final_cv[nz] = stretch_cv(oof_blend[nz], y[nz])   # honest estimate of the stretch
    test_final = np.clip(np.polyval(coef, test_blend), T.LO, T.HI)
    print(f"stretch coef (slope, intercept) = {coef.round(4)}")
    m_blend, m_final = T.metrics(oof_blend, y), T.metrics(oof_final_cv, y)
    print("blend           OOF", {k: round(v, 4) for k, v in m_blend.items()})
    print("blend + stretch OOF", {k: round(v, 4) for k, v in m_final.items()})

    ins = training_rmse(y, comps, names, w, coef, seeds)
    report = {"weights": dict(zip(names, map(float, w))), "stretch": list(map(float, coef)),
              "oof_blend": m_blend, "oof_final": m_final}
    if ins is not None:
        report["train_insample"] = T.metrics(ins, y)
        np.save(T.PREDS / "final_insample.npy", ins)
        print("TRAINING (in-sample) ", {k: round(v, 4) for k, v in report["train_insample"].items()})

    np.savez(T.PREDS / f"{OUT}.npz", oof=oof_final_cv, test=test_final, y=y)
    (T.CACHE / f"{OUT}_report.json").write_text(json.dumps(report, indent=2, default=float))
    T.write_submission(OUT, test_final)
    print("test prediction mean/std:", test_final.mean().round(3), test_final.std().round(3))


if __name__ == "__main__":
    main()
