"""Whisper-large-v3 encoder: mean+std pooled features of EVERY encoder layer (HF transformers).

Motivation: winning Speak & Improve 2025 spoken-assessment systems use *intermediate* Whisper
encoder representations; our whisper_enc.py only has the last layer (CTranslate2 exposes
nothing else). For WavLM, middle-upper layers beat the last one clearly - test the same here.

30 s windows (Whisper's input size), padding frames dropped before pooling (50 frames/s).
Output: cache/emb_whisper_layers.npz, X shape (N, 33, 2560) float16 = [mean | std] per layer
(layer 0 = conv/positional embeddings, 32 = last). Then:  python src/train.py wavlm_scan whisper_layers
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel

ROOT = Path(__file__).resolve().parents[1]
SR, WIN = 16000, 30 * 16000


@torch.no_grad()
def main():
    clips = pd.concat([pd.read_csv(ROOT / "data" / f"{s}.csv").assign(split=s)
                       for s in ("train", "test")])[["filename", "split"]]
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    enc = WhisperModel.from_pretrained("openai/whisper-large-v3",
                                       torch_dtype=torch.float16).encoder.to("cuda").eval()
    out = []
    for r in tqdm(clips.itertuples(), total=len(clips)):
        wav, _ = sf.read(ROOT / "data" / r.split / r.filename, dtype="float32")
        s1 = s2 = None
        n = 0
        for s in range(0, len(wav), WIN):
            seg = wav[s:s + WIN]
            if len(seg) < SR:
                continue
            feats = fe(seg, sampling_rate=SR, return_tensors="pt").input_features
            hs = enc(feats.to("cuda", torch.float16), output_hidden_states=True).hidden_states
            keep = int(np.ceil(len(seg) / SR * 50))
            h = torch.stack(hs)[:, 0, :keep].float()          # L x T x D
            s1 = h.sum(1) if s1 is None else s1 + h.sum(1)
            s2 = (h ** 2).sum(1) if s2 is None else s2 + (h ** 2).sum(1)
            n += keep
        mu = s1 / n
        sd = (s2 / n - mu ** 2).clamp(min=0).sqrt()
        out.append(torch.cat([mu, sd], -1).cpu().numpy().astype(np.float16))
    np.savez(ROOT / "cache" / "emb_whisper_layers.npz", filename=clips.filename.values,
             split=clips.split.values, X=np.stack(out))
    print("whisper_layers", np.stack(out).shape)


if __name__ == "__main__":
    main()
