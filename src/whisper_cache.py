"""Cache Whisper-large-v3 encoder hidden state 24 (= input of the top 8 layers) per 30 s window.

Used by finetune_whisper.py, which LoRA-fine-tunes encoder layers 25-32 on top of this cache,
so the frozen bottom 24 layers (and the conv front-end) run only once.
Per clip: cache/whisper24/<split>__<name>.npy  float16 (n_windows, 1500, 1280), full padded
windows exactly as Whisper sees them; cache/whisper24/valid_frames.csv holds the number of
real (non-padding) frames per window (50 frames/s).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "whisper24"
SR, WIN, LAYER = 16000, 30 * 16000, 24


def path(split, name):
    return OUT / f"{split}__{Path(name).stem}.npy"


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clips = pd.concat([pd.read_csv(ROOT / "data" / f"{s}.csv").assign(split=s)
                       for s in ("train", "test")])[["filename", "split"]]
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    enc = WhisperModel.from_pretrained("openai/whisper-large-v3",
                                       torch_dtype=torch.float16).encoder.to("cuda").eval()
    # Keep LAYER+1 layers: Hugging Face applies the encoder's final layer_norm to the LAST
    # hidden state, so hidden_states[LAYER] must not be the last one or it gets normalised.
    enc.layers = enc.layers[:LAYER + 1]
    rows = []
    for r in tqdm(clips.itertuples(), total=len(clips)):
        wav, _ = sf.read(ROOT / "data" / r.split / r.filename, dtype="float32")
        wins, valid = [], []
        for s in range(0, len(wav), WIN):
            seg = wav[s:s + WIN]
            if len(seg) < SR and s > 0:      # skip <1 s tail (keep at least one window)
                continue
            feats = fe(seg, sampling_rate=SR, return_tensors="pt").input_features
            h = enc(feats.to("cuda", torch.float16), output_hidden_states=True).hidden_states[LAYER]
            wins.append(h[0].cpu().numpy().astype(np.float16))
            valid.append(min(1500, int(np.ceil(len(seg) / SR * 50))))
        np.save(path(r.split, r.filename), np.stack(wins))
        rows.append({"filename": r.filename, "split": r.split, "valid": " ".join(map(str, valid))})
    pd.DataFrame(rows).to_csv(OUT / "valid_frames.csv", index=False)


if __name__ == "__main__":
    main()
