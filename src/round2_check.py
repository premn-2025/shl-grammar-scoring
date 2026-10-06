"""Leak check for the round-2 feature component: does the stack gain survive prompt-held-out
and speaker-grouped CV, and which feature group drives it?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import numpy as np  # noqa: E402
import gating as G  # noqa: E402
from round2_b import OUT, block  # noqa: E402

GROUPS = {
    "prosody": ["f0_mean_st", "f0_std_st", "f0_range_st", "f0_jump_std", "energy_mean", "energy_cv",
                "low_energy_frac"],
    "prosody_relative": ["f0_std_st", "f0_range_st", "f0_jump_std", "energy_cv", "low_energy_frac"],
    "timing": ["short_pause_pm", "short_pause_frac", "tempo_cv"],
    "text": ["mtld", "mean_dep_dist", "fragment_frac"],
}


def main():
    train, _ = T.load_labels()
    y = train.label.values
    for mode, folder in [("random", "preds"), ("group", "preds_g"), ("speaker", "preds_s")]:
        T.CV_MODE = mode
        T.PREDS = T.CACHE / folder
        base = G.per_split(folder=T.PREDS)
        for name, cols in [("all", None)] + list(GROUPS.items()):
            a, b, _ = block(OUT, cols)
            r = T.run_cv(f"ridge_r2_{name}", T.ridge, a, y, b, drop_zero=True, verbose=False)
            G.compare(G.per_split(G.BASE + [f"ridge_r2_{name}_dz"], folder=T.PREDS), base,
                      f"[{mode}] + Ridge({name}) alone {r[0]['rmse_nz']:.3f}")


if __name__ == "__main__":
    main()
