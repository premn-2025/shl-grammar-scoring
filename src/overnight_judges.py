"""Unattended overnight queue for the LLM-judge upgrades (each step logged in cache/overnight2/).
1. anchored Qwen3.5-9B judge   2. zero-shot Gemma2-9B judge   3. evaluation + MORNING_REPORT_2.md
Steps are independent: if one fails, the next still runs."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "cache" / "overnight2"
LOG.mkdir(parents=True, exist_ok=True)
STEPS = [
    ("judge_anchored", ["src/llm_judge_ollama.py", "qwen3.5:9b", "--tag", "anch", "--anchors"]),
    ("judge_gemma", ["src/llm_judge_ollama.py", "gemma2:9b", "--tag", "gemma"]),
    ("evaluate", ["src/round2_g.py"]),
]
with open(LOG / "status.txt", "a", encoding="utf-8") as st:
    for name, args in STEPS:
        t0 = time.time()
        with open(LOG / f"{name}.log", "w", encoding="utf-8") as f:
            rc = subprocess.run([sys.executable] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        st.write(f"{time.strftime('%H:%M')} {name}: rc={rc} ({(time.time() - t0) / 60:.0f} min)\n")
        st.flush()
