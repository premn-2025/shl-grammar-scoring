"""Cache frame-level WavLM-large features (one pass, used by attnpool.py and finetune_audio.py).

Per clip, written to cache/frames/:
  <split>__<name>.grp.npy  float16 (3, T/2, 1024): averages of hidden states 12-15, 16-19,
                           20-23 (the useful upper-middle layers, see the layer scan),
                           average-pooled 2x in time (50 -> 25 frames/s) to save disk.
  <split>__<name>.l16.npy  float16 (T, 1024): hidden state 16 at full rate = the input of
                           transformer layers 17-24, which finetune_audio.py fine-tunes.
Audio is processed in the same 20 s chunks as embeddings.py; frames are concatenated.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "frames"
GROUPS = [(12, 16), (16, 20), (20, 24)]
CHUNK = 20 * 16000


def frame_path(split, name, kind):
    return OUT / f"{split}__{Path(name).stem}.{kind}.npy"


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clips = pd.concat([pd.read_csv(ROOT / "data" / f"{s}.csv").assign(split=s)
                       for s in ("train", "test")])
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    m = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().half()
    for r in tqdm(clips.itertuples(), total=len(clips)):
        if frame_path(r.split, r.filename, "l16").exists():
            continue  # resumable
        wav, sr = sf.read(ROOT / "data" / r.split / r.filename, dtype="float32")
        grp, l16 = [], []
        for s in range(0, len(wav), CHUNK):
            seg = wav[s:s + CHUNK]
            if len(seg) < 16000:
                continue
            x = fe(seg, sampling_rate=16000, return_tensors="pt").input_values.to("cuda").half()
            hs = m(x, output_hidden_states=True).hidden_states
            g = torch.stack([torch.stack(hs[a:b]).mean(0)[0] for a, b in GROUPS])  # 3,T,D
            T = g.shape[1] // 2 * 2
            grp.append(g[:, :T].reshape(3, T // 2, 2, -1).mean(2).cpu().numpy())
            l16.append(hs[16][0].cpu().numpy())
        np.save(frame_path(r.split, r.filename, "grp"), np.concatenate(grp, 1).astype(np.float16))
        np.save(frame_path(r.split, r.filename, "l16"), np.concatenate(l16, 0).astype(np.float16))


if __name__ == "__main__":
    main()
