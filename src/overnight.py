"""Unattended overnight experiment queue. Each step runs as its own process (logs in
cache/overnight/), failures are recorded and the queue continues. At the end it builds
final_v3 (only components that exist) and writes MORNING_REPORT.md.

Usage: python src/overnight.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "cache" / "overnight"
LOG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
STATUS = []
PY = sys.executable


def step(name, args, env=None, timeout_h=4):
    t0 = time.time()
    e = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1", **(env or {}))
    with open(LOG / f"{name}.log", "w", encoding="utf-8") as f:
        try:
            rc = subprocess.run([PY] + args, cwd=ROOT, env=e, stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout_h * 3600).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    STATUS.append({"step": name, "rc": rc, "minutes": round((time.time() - t0) / 60, 1)})
    (LOG / "status.json").write_text(json.dumps(STATUS, indent=2))
    print(STATUS[-1], flush=True)
    return rc == 0


def preds(name):
    return ROOT / "cache" / "preds" / f"{name}.npz"


def combine_seeds(base):
    """Average all available seeds of an audio fine-tune variant -> <base>_avg(.npz/_fulltrain)."""
    import train as T
    P = ROOT / "cache" / "preds"
    seeds = sorted(p for p in P.glob(f"{base}_s*.npz"))
    if not seeds:
        return None
    oof = np.mean([np.load(s)["oof"] for s in seeds], 0)
    test = np.mean([np.load(s)["test"] for s in seeds], 0)
    y = np.load(seeds[0])["y"]
    fulls = []
    for s in seeds:
        f = P / f"{s.stem}_fulltrain.npy"
        if not f.exists() and s.stem.endswith("_s0") and (P / f"{base}_fulltrain.npy").exists():
            f = P / f"{base}_fulltrain.npy"   # seed-0 run predates per-seed saving
        if f.exists():
            fulls.append(np.load(f))
    np.savez(P / f"{base}_avg.npz", oof=oof, test=test, y=y)
    if fulls:
        np.save(P / f"{base}_avg_fulltrain.npy", np.mean(fulls, 0))
    m = T.metrics(oof, y)
    print(base, f"{len(seeds)} seeds", m, flush=True)
    return m["rmse_nz"], len(seeds), bool(fulls)


def wait_for_gpu_job(marker_file, needle, max_h=2):
    """The seed-0 audio fine-tune was started before this queue; wait for it to finish."""
    t0 = time.time()
    while time.time() - t0 < max_h * 3600:
        txt = marker_file.read_text(encoding="utf-8", errors="ignore") if marker_file.exists() else ""
        if needle in txt:
            return True
        if "Traceback" in txt:
            return False
        time.sleep(60)
    return False


def main():
    # 0) let the already-running option-1 seed-0 job finish (it uses the GPU)
    ok = wait_for_gpu_job(ROOT / "cache" / "ft_audio.log", "ft_wavlmL_top8 seeds")
    STATUS.append({"step": "wait_option1_seed0", "rc": 0 if ok else "failed/timeout"})
    P = ROOT / "cache" / "preds"
    if (P / "ft_wavlmL_top8_fulltrain.npy").exists() and not (P / "ft_wavlmL_top8_s0_fulltrain.npy").exists():
        # the seed-0 run predates per-seed saving; keep its copy before later runs overwrite it
        np.save(P / "ft_wavlmL_top8_s0_fulltrain.npy", np.load(P / "ft_wavlmL_top8_fulltrain.npy"))

    # 1) idea 2: clean (unprompted) Whisper pass -> diff features
    if step("whisper_clean", ["src/transcribe.py", "--clean"], timeout_h=2):
        step("whisper_diff", ["src/whisper_diff.py"], timeout_h=0.5)

    # 2) idea 3: LLM judge (needs the Qwen download to have finished)
    step("llm_judge", ["src/llm_judge.py"], timeout_h=1.5)

    # 3) idea 1: longer option-1 schedule, seed 0
    long_env = {"FA_EPOCHS": "10", "FA_LR": "5e-05"}
    step("ft_audio_long_s0", ["src/finetune_audio.py", "0"], env=long_env, timeout_h=3.5)

    # 4) CPU models with the new features
    step("train_extra", ["src/train.py", "extra"], timeout_h=1.5)

    # 5) more seeds for whichever option-1 variant is better
    short = combine_seeds("ft_wavlmL_top8")
    long = combine_seeds("ft_wavlmL_top8_e10_lr5e-05")
    if short and long:
        best_env = long_env if long[0] < short[0] else {}
    else:
        best_env = long_env if long else {}
    STATUS.append({"step": "choose_variant", "short": short, "long": long,
                   "chosen": "long" if best_env else "short"})
    step("ft_audio_more_seeds", ["src/finetune_audio.py", "1", "2"], env=best_env, timeout_h=5)
    short = combine_seeds("ft_wavlmL_top8")
    long = combine_seeds("ft_wavlmL_top8_e10_lr5e-05")

    # 6) final_v3 with every available extra component
    extras = []
    for n in ["attnpool_wavlmL", "ft_wavlmL_top8_avg", "ft_wavlmL_top8_e10_lr5e-05_avg",
              "ridge_llm_dz", "lgbm_handx_dz", "ridge_deb_handx_dz", "svr_allx"]:
        if preds(n).exists() and (ROOT / "cache" / "preds" / f"{n}_fulltrain.npy").exists():
            extras.append(n)
    step("final_v3", ["src/final.py", "--extra", ",".join(extras), "--out", "final_v3"], timeout_h=2)
    report(extras)


def report(extras):
    import train as T
    lines = ["# Morning report (overnight queue)", "",
             "## Steps", "", "| step | result | minutes |", "|---|---|---|"]
    for s in STATUS:
        lines.append(f"| {s['step']} | {s.get('rc', s.get('chosen', ''))} | {s.get('minutes', '')} |")
    lines += ["", "## Components (5-fold OOF, non-zero clips)", "",
              "| component | RMSE(nz) | Pearson(nz) |", "|---|---|---|"]
    base = ["svr_all", "svr_wavlm_large", "ridge_deb_hand_dz", "ft_roberta"]
    for n in base + extras:
        if preds(n).exists():
            z = np.load(preds(n))
            m = T.metrics(z["oof"], z["y"])
            lines.append(f"| {n} | {m['rmse_nz']:.4f} | {m['pearson_nz']:.4f} |")
    lines += ["", "## Final blends", ""]
    rows = {}
    for out in ["final", "final_v2", "final_v3"]:
        p = ROOT / "cache" / f"{out}_report.json"
        if p.exists():
            r = json.loads(p.read_text())
            rows[out] = r["oof_final"]["rmse_nz"]
            w = ", ".join(f"{k} {v:.2f}" for k, v in r["weights"].items() if v > 0.005)
            tr = r.get("train_insample", {}).get("rmse_nz", float("nan"))
            lines.append(f"- **{out}**: OOF RMSE(nz) {r['oof_final']['rmse_nz']:.4f}, "
                         f"training RMSE(nz) {tr:.4f}; weights: {w}")
    lines += ["", "Public LB so far: final 0.3529, final_v2 0.3507.", ""]
    if "final_v3" in rows and "final_v2" in rows:
        gain = rows["final_v2"] - rows["final_v3"]
        if gain > 0.005:
            rec = (f"**Submit `submissions/final_v3.csv`** - CV improves by {gain:.4f} over v2, "
                   "a real improvement (more than the ~0.002 noise level of the blend weights).")
        elif gain > 0:
            rec = (f"v3 improves CV by only {gain:.4f}: within noise. Submitting it is fine but "
                   "expect a leaderboard change of the same tiny size.")
        else:
            rec = "v3 does not improve CV - keep final_v2."
        lines += ["## Recommendation", "", rec]
    (ROOT / "MORNING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
