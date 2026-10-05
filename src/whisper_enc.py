"""Whisper large-v3 *encoder* embeddings (last encoder layer), mean+std pooled over time.

The encoder was trained on 5M hours of speech to support transcription, so its states carry
both acoustic and linguistic (word/structure) information -- useful for proficiency scoring.
Uses the CTranslate2 model already downloaded for transcription (no extra download), so like
transcribe.py it runs in its own process (CUDA 12 DLLs, no torch).

Audio is split into 30 s windows (Whisper's fixed input size); frames that correspond to
zero-padding in the last window are dropped before pooling.
Output: cache/emb_whisper_enc.npz  (X = [mean(1280) | std(1280)]).
"""
import numpy as np
import pandas as pd

from transcribe import DATA, ROOT, add_nvidia_dlls

add_nvidia_dlls()
import ctranslate2  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402
from tqdm import tqdm  # noqa: E402

SR, WIN = 16000, 30 * 16000
FRAMES_PER_SEC = 50  # encoder output: 1500 frames per 30 s


def main():
    tr = pd.concat([pd.read_csv(DATA / f"{s}.csv").assign(split=s)
                    for s in ("train", "test")])[["filename", "split"]]
    wm = WhisperModel("large-v3", device="cuda", compute_type="float16")
    out = []
    for r in tqdm(tr.itertuples(), total=len(tr)):
        audio = decode_audio(str(DATA / r.split / r.filename), sampling_rate=SR)
        frames = []
        for s in range(0, len(audio), WIN):
            seg = audio[s:s + WIN]
            if len(seg) < SR:  # skip <1 s tail
                continue
            mel = wm.feature_extractor(np.pad(seg, (0, WIN - len(seg))))[:, :3000]
            sv = ctranslate2.StorageView.from_array(mel[None].astype(np.float32))
            enc = np.array(wm.model.encode(sv, to_cpu=True))[0]  # (1500, 1280)
            frames.append(enc[: int(np.ceil(len(seg) / SR * FRAMES_PER_SEC))])
        f = np.concatenate(frames).astype(np.float32)
        out.append(np.concatenate([f.mean(0), f.std(0)]))
    np.savez(ROOT / "cache" / "emb_whisper_enc.npz", filename=tr.filename.values,
             split=tr.split.values, X=np.stack(out))
    print("whisper_enc", np.stack(out).shape)


if __name__ == "__main__":
    main()
