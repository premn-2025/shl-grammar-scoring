"""Step 0 + Step 1: inspect the data and write a constant (train-mean) baseline submission.

The mean baseline is the RMSE-optimal constant predictor; it validates the submission
format end-to-end before any modelling work.
"""
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

print("train:", train.shape, list(train.columns))
print("test:", test.shape, list(test.columns))
print("sample_submission:", sample.shape, list(sample.columns))
print("test & sample have same filenames:", set(test.filename) == set(sample.filename))

print("\nLabel distribution:")
print(train.label.value_counts().sort_index().to_string())
print(f"mean={train.label.mean():.4f}  std={train.label.std():.4f}  "
      f"min={train.label.min()}  max={train.label.max()}")

# Audio metadata from file headers only (fast; no decoding needed).
def audio_info(split, names):
    rows = []
    for n in names:
        info = sf.info(DATA / split / n)
        rows.append((info.samplerate, info.channels, info.duration))
    return pd.DataFrame(rows, columns=["sr", "channels", "duration"])

for split, df in [("train", train), ("test", test)]:
    info = audio_info(split, df.filename)
    print(f"\n{split} audio: sample rates={info.sr.value_counts().to_dict()}, "
          f"channels={info.channels.value_counts().to_dict()}")
    print(info.duration.describe().round(2).to_string())

# Step 1: constant baseline. sample_submission.csv is inconsistent with test.csv (only 25 of
# its 204 filenames are test files), while test.csv matches the 216 WAVs in data/test exactly.
# So we keep sample_submission's format (filename,label) but use test.csv's rows/order.
sub = test[["filename"]].copy()
sub["label"] = train.label.mean()
out = ROOT / "submissions" / "00_mean_baseline.csv"
sub.to_csv(out, index=False)
print(f"\nWrote {out} ({len(sub)} rows)")
