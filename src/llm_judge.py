"""LLM grammar judge (zero-shot): Qwen2.5-3B-Instruct reads the SHL rubric + transcript and
we read its next-token probabilities for "1".."5" (no text generation).

Features per clip: p1..p5 and the expected score sum(k * p_k). This is a pretrained judge
applied without any training on our labels, so it cannot overfit them; the downstream
Ridge/SVR learns how much to trust it inside the usual CV.

Output: cache/emb_llm.npz (filename, split, X = [p1..p5, expected]).
Usage: python src/llm_judge.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-3B-Instruct"

RUBRIC = """Grammar score rubric:
1 - The speaker struggles with sentence structure and relies on memorized patterns; many errors.
2 - Uses simple structures with consistent basic mistakes; incomplete sentences are common.
3 - Decent grasp of grammar, but makes noticeable errors in grammar or syntax.
4 - Strong control of grammar; only occasional minor errors, often self-corrected.
5 - High grammatical accuracy; complex structures handled well; self-corrects where needed."""

INSTR = ("You are an expert English-language examiner. Below is an automatic transcript of a "
         "candidate's spoken answer (about one minute). Fillers such as 'um'/'uh', repetitions "
         "and cut-off words come from the speech itself; punctuation was added by the "
         "transcription system and should be ignored.\n\n{rubric}\n\nTranscript:\n\"\"\"{text}\"\"\"\n\n"
         "Rate ONLY the speaker's grammar on the 1-5 scale. Answer with a single digit.")


@torch.no_grad()
def main():
    tr = pd.read_csv(ROOT / "cache" / "transcripts.csv").fillna({"text": ""})
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 6)]
    out = []
    for text in tqdm(tr.text.tolist()):
        text = " ".join(text.split()[:300]) or "(no speech)"
        msgs = [{"role": "user", "content": INSTR.format(rubric=RUBRIC, text=text)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        logits = m(**ids).logits[0, -1].float()
        p = torch.softmax(logits[digit_ids], 0).cpu().numpy()   # renormalised over 1..5
        out.append(np.concatenate([p, [(p * np.arange(1, 6)).sum()]]))
    X = np.array(out, dtype=np.float32)
    np.savez(ROOT / "cache" / "emb_llm.npz", filename=tr.filename.values, split=tr.split.values, X=X)
    lab = pd.read_csv(ROOT / "data" / "train.csv")
    d = pd.DataFrame({"filename": tr.filename, "split": tr.split, "llm": X[:, -1]})
    d = d[d.split == "train"].merge(lab, on="filename")
    d = d[d.label > 0]
    print("LLM expected score: corr with label (non-zero) =", round(d.llm.corr(d.label), 3),
          "| mean", round(d.llm.mean(), 3))


if __name__ == "__main__":
    main()
