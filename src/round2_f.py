"""Local LLM judge (cache/features_llm_judge.csv from src/llm_judge_ollama.py, Qwen3.5-9B).

Judged against BOTH final models (final_stack_w: 6 components; final_stack_pw: + prosody
component) under random, prompt-held-out and speaker-grouped CV:
  L1 the raw judge score as a stack component (zero-shot: never saw our labels, so it can be
     used directly as an 'out-of-fold' prediction without any leakage)
  L2 Ridge on all judge outputs (score, error rates by type, complexity, confidence)
  L3 judge outputs merged into the prosody/timing Ridge (pw only)
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

LLM = T.CACHE / "features_llm_judge.csv"
BASES = {"final_stack_w": G.BASE, "final_stack_pw": G.BASE + ["ridge_r2w_dz"]}


def main():
    train, _ = T.load_labels()
    y = train.label.values
    la, lb, lcols = block(LLM)
    nz = y > 0
    print("LLM judge correlations with label (non-zero clips):")
    print(pd.DataFrame(la[nz], columns=lcols).corrwith(pd.Series(y[nz])).round(3).to_string())
    i = lcols.index("llm_score")
    L = wins(la, lb)
    ra, rb, _ = block(OUT)
    RL = wins(np.hstack([ra, la]), np.hstack([rb, lb]))
    for mode, folder in [("random", "preds"), ("group", "preds_g"), ("speaker", "preds_s")]:
        T.CV_MODE = mode
        T.PREDS = T.CACHE / folder
        np.savez(T.PREDS / "llm_raw.npz", oof=np.clip(la[:, i], 0, 5), test=np.clip(lb[:, i], 0, 5), y=y)
        np.save(T.PREDS / "llm_raw_fulltrain.npy", np.clip(la[:, i], 0, 5))
        fit = T.cv_and_full if mode == "random" else (lambda *a, **k: T.run_cv(*a, verbose=False, **k))
        fit("ridge_llm9b", T.ridge, L[0], y, L[1], drop_zero=True)
        fit("ridge_r2l", T.ridge, RL[0], y, RL[1], drop_zero=True)
        for bname, comps in BASES.items():
            base = G.per_split(comps, folder=T.PREDS)
            G.compare(G.per_split(comps + ["llm_raw"], folder=T.PREDS), base, f"[{mode}|{bname}] L1 + raw judge score")
            G.compare(G.per_split(comps + ["ridge_llm9b_dz"], folder=T.PREDS), base, f"[{mode}|{bname}] L2 + judge Ridge")
        base = G.per_split(BASES["final_stack_pw"], folder=T.PREDS)
        G.compare(G.per_split(G.BASE + ["ridge_r2l_dz"], folder=T.PREDS), base,
                  f"[{mode}|final_stack_pw] L3 judge merged into prosody Ridge")


if __name__ == "__main__":
    main()
