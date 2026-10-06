"""Overnight LLM-judge upgrades, judged against final_stack_wj (public LB 0.3362).

Judge feature files (all from src/llm_judge_ollama.py):
  Z  features_llm_judge.csv        Qwen3.5-9B zero-shot (already in final_stack_wj)
  A  features_llm_judge_anch.csv   Qwen3.5-9B anchored with 8 human-scored training examples
  G  features_llm_judge_gemma.csv  Gemma2-9B zero-shot
Variants (Ridge on the judge block, winsorised to the training range, no 0.0 clips):
  V1 A replaces Z      V2 Z+A in one Ridge      V3 Z+A+G in one Ridge      V4 Z and A as two components
Decision rule (validated by the public LB on final_stack_wj): keep if prompt-held-out short-clip
gain >= 0.0015 with >= 6/8 wins, AND random-fold and speaker-grouped short-clip changes >= -0.001.
Writes candidate submissions and MORNING_REPORT_2.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import gating as G  # noqa: E402
from round2_b import block  # noqa: E402
from round2_d import wins  # noqa: E402

ROOT = T.CACHE.parent
FILES = {"Z": T.CACHE / "features_llm_judge.csv", "A": T.CACHE / "features_llm_judge_anch.csv",
         "G": T.CACHE / "features_llm_judge_gemma.csv"}
BASE6 = G.BASE
WJ = BASE6 + ["ridge_llm9b_dz"]
LINES = ["# Morning report 2 - LLM judge upgrades", ""]


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def main():
    train, _ = T.load_labels()
    y = train.label.values
    nz = y > 0
    avail = {k: block(p) for k, p in FILES.items() if p.exists()}
    say("## Judge quality (correlation of the judge's score with the human label, non-zero clips)")
    say("")
    for k, (a, b, cols) in avail.items():
        say(f"- {k}: r = {np.corrcoef(a[nz, cols.index('llm_score')], y[nz])[0, 1]:.3f}")
    if {"Z", "A"} <= avail.keys():
        za = np.corrcoef(avail["Z"][0][nz, avail["Z"][2].index("llm_score")],
                         avail["A"][0][nz, avail["A"][2].index("llm_score")])[0, 1]
        say(f"- correlation between Z and A scores: {za:.3f}")
    say("")

    def blk(keys):
        return wins(np.hstack([avail[k][0] for k in keys]), np.hstack([avail[k][1] for k in keys]))

    variants = {}
    if "A" in avail:
        variants["V1 anchored replaces zero-shot"] = (BASE6 + ["ridge_jA_dz"], {"ridge_jA": blk(["A"])})
        variants["V2 zero-shot+anchored in one Ridge"] = (BASE6 + ["ridge_jZA_dz"], {"ridge_jZA": blk(["Z", "A"])})
        variants["V4 zero-shot and anchored as two components"] = (WJ + ["ridge_jA_dz"], {"ridge_jA": blk(["A"])})
    if {"A", "G"} <= avail.keys():
        variants["V3 zero-shot+anchored+gemma in one Ridge"] = (BASE6 + ["ridge_jZAG_dz"], {"ridge_jZAG": blk(["Z", "A", "G"])})
    if "G" in avail:
        variants["V5 zero-shot+gemma in one Ridge"] = (BASE6 + ["ridge_jZG_dz"], {"ridge_jZG": blk(["Z", "G"])})

    results = {}
    for mode, folder in [("random", "preds"), ("group", "preds_g"), ("speaker", "preds_s")]:
        T.CV_MODE = mode
        T.PREDS = T.CACHE / folder
        base = G.per_split(WJ, folder=T.PREDS)
        for vname, (comps, builds) in variants.items():
            for cname, (Xa, Xb) in builds.items():
                if mode == "random":
                    T.cv_and_full(cname, T.ridge, Xa, y, Xb, drop_zero=True)
                else:
                    T.run_cv(cname, T.ridge, Xa, y, Xb, drop_zero=True, verbose=False)
            r = G.per_split(comps, folder=T.PREDS)
            d = base[:, 0] - r[:, 0]
            results.setdefault(vname, {})[mode] = (d.mean(), int((d > 0).sum()), r[:, 0].mean())
    T.CV_MODE, T.PREDS = "random", T.CACHE / "preds"

    say("## Paired short-clip CV vs final_stack_wj (gain = baseline RMSE - candidate RMSE)")
    say("")
    say("| variant | random | prompt-held-out | speaker-grouped | verdict |")
    say("|---|---|---|---|---|")
    keep = []
    for vname, res in results.items():
        g, s, r = res["group"], res["speaker"], res["random"]
        ok = g[0] >= 0.0015 and g[1] >= 6 and r[0] >= -0.001 and s[0] >= -0.001
        if ok:
            keep.append((g[0], vname))
        fmt = lambda t: f"{t[0]:+.4f} ({t[1]}/8)"
        say(f"| {vname} | {fmt(r)} | {fmt(g)} | {fmt(s)} | {'KEEP' if ok else 'reject'} |")
    say("")
    keep.sort(reverse=True)
    import subprocess
    for i, (gain, vname) in enumerate(keep[:2]):
        comps = variants[vname][0]
        out = f"final_stack_wj{i + 2}"
        subprocess.run([sys.executable, str(ROOT / "src" / "stack.py"), ",".join(comps), "--out", out],
                       cwd=ROOT, check=True, capture_output=True)
        a = pd.read_csv(ROOT / "submissions" / "final_stack_wj.csv")
        b = pd.read_csv(ROOT / "submissions" / f"{out}.csv")
        diff = float(np.sqrt(((a.label - b.label) ** 2).mean()))
        say(f"- **Candidate `submissions/{out}.csv`** = {vname} (prompt-held-out gain {gain:+.4f}); "
            f"test predictions differ from final_stack_wj by {diff:.3f} RMS.")
    if not keep:
        say("- No variant passed: keep **final_stack_wj** (0.3362) as the final submission.")
    say("")
    say("Keep final_stack_wj selected in any case; submit a candidate only as an extra check.")
    (ROOT / "MORNING_REPORT_2.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
