"""Writes MORNING_REPORT.md after the overnight queue: step status, every component's
training vs CV RMSE (over/underfitting check), and an honest NESTED estimate for each final
blend (blend weights + stretch fitted on 4/5 of clips, scored on the other 1/5). The
recommendation is based on the nested numbers, never on the public leaderboard alone.

Usage: python src/morning_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train as T  # noqa: E402
from ensemble import nested_blend_rmse  # noqa: E402

P = ROOT / "cache" / "preds"
LB = {"final": 0.3529, "final_v2": 0.3507}


def fit_label(train_rmse, cv_rmse):
    """Plain-language reading of the train/CV pair."""
    if np.isnan(train_rmse):
        return "n/a"
    gap = cv_rmse - train_rmse
    if train_rmse > 0.45:
        return "underfits (high error even on its own training data)"
    if gap > 0.35:
        return "memorises training data; CV is the honest number"
    return "balanced"


def main():
    lines = ["# Morning report", ""]
    st = ROOT / "cache" / "overnight" / "status.json"
    if st.exists():
        lines += ["## Overnight steps", "", "| step | result | minutes |", "|---|---|---|"]
        for s in json.loads(st.read_text()):
            lines.append(f"| {s['step']} | {s.get('rc', s.get('chosen', ''))} | {s.get('minutes', '')} |")
        lines.append("")

    comps = ["svr_all", "svr_wavlm_large", "ridge_deb_hand_dz", "ft_roberta", "attnpool_wavlmL",
             "ft_wavlmL_top8_avg", "ft_wavlmL_top8_e10_lr5e-05_avg", "ridge_llm_dz",
             "lgbm_handx_dz", "ridge_deb_handx_dz", "svr_allx"]
    lines += ["## Components: training vs cross-validated RMSE (non-zero clips)", "",
              "| component | train RMSE | CV RMSE | CV Pearson | reading |", "|---|---|---|---|---|"]
    for n in comps:
        if not (P / f"{n}.npz").exists():
            continue
        z = np.load(P / f"{n}.npz")
        m = T.metrics(z["oof"], z["y"])
        tr = float("nan")
        f = P / f"{n}_fulltrain.npy"
        if n == "ft_roberta":
            fs = sorted(P.glob("ft_roberta-base-CoLA_e4_s*_fulltrain.npy"))
            if fs:
                tr = T.metrics(np.mean([np.load(x) for x in fs], 0), z["y"])["rmse_nz"]
        elif f.exists():
            tr = T.metrics(np.load(f), z["y"])["rmse_nz"]
        lines.append(f"| {n} | {tr:.3f} | {m['rmse_nz']:.4f} | {m['pearson_nz']:.3f} | "
                     f"{fit_label(tr, m['rmse_nz'])} |")
    lines += ["", "_SVR/fine-tuned models fit their own training clips closely; what matters "
              "is the CV column and that it is stable across folds/seeds._", ""]

    lines += ["## Final blends", "", "| blend | CV RMSE | **nested CV RMSE** | train RMSE | public LB | "
              "non-zero weights |", "|---|---|---|---|---|---|"]
    nested = {}
    for out in ["final", "final_v2", "final_v3"]:
        p = ROOT / "cache" / f"{out}_report.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        names = list(r["weights"])
        if not all((P / f"{n}.npz").exists() for n in names):
            continue
        nested[out] = nested_blend_rmse(names)
        w = ", ".join(f"{k} {v:.2f}" for k, v in r["weights"].items() if v > 0.005)
        tr = r.get("train_insample", {}).get("rmse_nz", float("nan"))
        lines.append(f"| {out} | {r['oof_final']['rmse_nz']:.4f} | **{nested[out]:.4f}** | "
                     f"{tr:.3f} | {LB.get(out, '-')} | {w} |")
    lines.append("")

    lines += ["## Recommendation", ""]
    if "final_v3" in nested and "final_v2" in nested:
        gain = nested["final_v2"] - nested["final_v3"]
        if gain >= 0.005:
            lines.append(f"**Submit `submissions/final_v3.csv`.** Nested CV improves by {gain:.4f} "
                         "over v2, a real gain beyond blend noise (~0.002).")
        elif gain > 0:
            lines.append(f"v3 is better by only {gain:.4f} on nested CV (within noise). Optional "
                         "submission; expect a tiny leaderboard change either way.")
        else:
            lines.append("v3 does not beat v2 on nested CV: **keep final_v2** "
                         "(adding components did not generalise).")
    else:
        lines.append("final_v3 was not built; keep final_v2. Check the step table above.")
    lines.append("")
    lines.append("Reminder: the public leaderboard uses ~60% of test (~130 clips); the final "
                 "ranking uses the other 40%. Choose by CV, not by small public-LB moves.")
    (ROOT / "MORNING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
