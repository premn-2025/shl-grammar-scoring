"""Whisper decoder uncertainty: per-token entropy of the next-token distribution while
greedily transcribing each 30 s window (HF Whisper-large-v3, fp16, English, no prompt).

Intuition: when speech is ungrammatical or unclear, the decoder's language model is less sure
what comes next -> flatter distributions -> higher entropy. faster-whisper only gave us the
probability of the chosen word; this keeps the whole distribution's entropy.
Features per clip: mean / p90 / max entropy, share of tokens with entropy > 2 nats, mean top-1
probability. Output: cache/features_entropy.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parents[1]
SR, WIN = 16000, 30 * 16000


@torch.no_grad()
def main():
    clips = pd.concat([pd.read_csv(ROOT / "data" / f"{s}.csv").assign(split=s)
                       for s in ("train", "test")])[["filename", "split"]]
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()          # silence per-call generation warnings
    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16).to("cuda").eval()
    rows = []
    for k, r in enumerate(clips.itertuples()):
        if k % 50 == 0:
            print(f"progress {k}/{len(clips)}", flush=True)
        wav, _ = sf.read(ROOT / "data" / r.split / r.filename, dtype="float32")
        ents, top1 = [], []
        for s in range(0, len(wav), WIN):
            seg = wav[s:s + WIN]
            if len(seg) < SR:
                continue
            feats = proc(seg, sampling_rate=SR, return_tensors="pt").input_features.to("cuda", torch.float16)
            out = model.generate(feats, language="en", task="transcribe", max_new_tokens=220,
                                 return_dict_in_generate=True, output_scores=True)
            for sc in out.scores:                       # one (1, vocab) logit row per new token
                p = torch.softmax(sc[0].float(), -1)
                ents.append(float(-(p * torch.log(p + 1e-12)).sum()))
                top1.append(float(p.max()))
        e = np.array(ents) if ents else np.array([0.0])
        rows.append({"filename": r.filename, "split": r.split, "ent_mean": e.mean(),
                     "ent_p90": np.percentile(e, 90), "ent_max": e.max(), "ent_hi_frac": (e > 2).mean(),
                     "top1_mean": float(np.mean(top1)) if top1 else 0.0})
    pd.DataFrame(rows).to_csv(ROOT / "cache" / "features_entropy.csv", index=False)


if __name__ == "__main__":
    main()
