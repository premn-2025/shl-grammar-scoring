"""Step 2: verbatim-ish transcription with faster-whisper large-v3 (run once, cached).

Run as its own process:  python src/transcribe.py
It deliberately never imports torch: torch here is built for CUDA 11.8, while CTranslate2
(faster-whisper's backend) needs CUDA 12 cuBLAS + cuDNN 9. Keeping them in separate
processes avoids a DLL clash on Windows.

Output: cache/transcripts.csv with one row per clip. Resumable: clips already present are
skipped, and each clip is appended as soon as it is done.
"""
import json
import os
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "cache" / "transcripts.csv"


def add_nvidia_dlls():
    """Make the pip-installed CUDA 12 DLLs (nvidia-cublas-cu12, nvidia-cudnn-cu12) findable.
    On Windows these live in site-packages/nvidia/<lib>/bin, which is not on PATH."""
    if sys.platform != "win32":
        return
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nv = Path(sp) / "nvidia"
        if nv.is_dir():
            for bin_dir in nv.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


add_nvidia_dlls()

import pandas as pd  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Whisper is trained to output clean, fluent text, so it tends to "fix" grammar mistakes and
# drop fillers -- exactly the signal we want to score. A disfluent, ungrammatical prompt nudges
# the decoder towards verbatim style (keeping "uh", repetitions, and errors).
INITIAL_PROMPT = ("Umm, so I, I was go to the market and, uh, he don't know what- "
                  "what I am saying.")


def decode(model, path, word_timestamps=True, condition_on_previous_text=True):
    segments, info = model.transcribe(
        str(path),
        language="en",
        beam_size=5,
        temperature=0.0,          # deterministic decoding (no sampling fallback)
        word_timestamps=word_timestamps,  # needed for pause / speech-rate features
        vad_filter=False,         # keep silences so pauses stay measurable
        initial_prompt=INITIAL_PROMPT,
        condition_on_previous_text=condition_on_previous_text,
    )
    return list(segments), info  # generator -> actually runs the decoding


def transcribe_one(model, path, decoded=None):
    # faster-whisper's word alignment occasionally crashes on an empty segment (IndexError in
    # find_alignment). Fallbacks: (1) don't condition on previous text, (2) no word timestamps.
    fallback = 0
    try:
        segments, info = decoded if decoded is not None else decode(model, path)
    except IndexError:
        fallback = 1
        try:
            segments, info = decode(model, path, condition_on_previous_text=False)
        except IndexError:
            fallback = 2
            segments, info = decode(model, path, word_timestamps=False)
    words, texts, logprobs, nospeech, seg_durs = [], [], [], [], []
    for s in segments:
        texts.append(s.text.strip())
        dur = max(s.end - s.start, 1e-3)
        logprobs.append(s.avg_logprob * dur)  # duration-weighted average below
        nospeech.append(s.no_speech_prob)
        seg_durs.append(dur)
        for w in s.words or []:
            words.append([round(w.start, 3), round(w.end, 3), w.word, round(w.probability, 4)])
    tot = sum(seg_durs) or 1.0
    return {
        "text": " ".join(texts),
        "duration": round(info.duration, 3),
        "words": json.dumps(words),
        "avg_logprob": sum(logprobs) / tot if segments else float("nan"),
        "no_speech_prob": sum(nospeech) / len(nospeech) if nospeech else float("nan"),
        "n_segments": len(segments),
        "fallback": fallback,
    }


def clean_pass():
    """Second transcription WITHOUT the disfluent prompt -> cache/transcripts_clean.csv.
    Unprompted, Whisper behaves like a copy-editor: it drops fillers and repetitions and
    often 'fixes' grammar. Comparing it with the verbatim pass (whisper_diff.py) therefore
    measures how much had to be fixed. Loop guard on (temperature fallback, no conditioning).
    Resumable."""
    out = ROOT / "cache" / "transcripts_clean.csv"
    clips = pd.concat([pd.read_csv(DATA / f"{s}.csv").assign(split=s)
                       for s in ("train", "test")])[["filename", "split"]]
    done = set()
    if out.exists():
        d = pd.read_csv(out)
        done = set(d.split + "/" + d.filename)
    clips = clips[~(clips.split + "/" + clips.filename).isin(done)]
    print(f"{len(done)} done, {len(clips)} to go")
    if clips.empty:
        return
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for r in tqdm(clips.itertuples(), total=len(clips)):
        segments, _ = model.transcribe(
            str(DATA / r.split / r.filename), language="en", beam_size=5, vad_filter=False,
            condition_on_previous_text=False, compression_ratio_threshold=2.4,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        text = " ".join(s.text.strip() for s in segments)
        pd.DataFrame([{"filename": r.filename, "split": r.split, "text": text}]).to_csv(
            out, mode="a", header=not out.exists(), index=False)


def flag_loops():
    """Write cache/redo_loops.csv: clips whose transcript looks like a repetition loop.
    Same rule for train and test: gzip compression ratio > 2.4 (Whisper's own threshold) or
    > 15 immediate word repetitions / phrase restarts per 100 words."""
    import zlib
    from features import disfluency_features
    df = pd.read_csv(OUT).fillna({"text": ""})
    cr = df.text.apply(lambda s: len(s.encode()) / max(len(zlib.compress(s.encode())), 1))
    dis = pd.DataFrame([disfluency_features(t) for t in df.text])
    flag = (cr > 2.4) | (dis.repetition_per100 > 15) | (dis.restart_per100 > 15)
    df[flag][["filename", "split"]].to_csv(ROOT / "cache" / "redo_loops.csv", index=False)
    print(f"flagged {flag.sum()} clips:", df[flag].split.value_counts().to_dict())


def redo_loops():
    """Re-transcribe clips whose first pass degenerated into a repetition loop
    ("so, so, so, ..."). Our main pass used temperature=0 only, which disables Whisper's
    built-in guard; here we turn it back on: if a window's gzip compression ratio > 2.4
    (i.e. highly repetitive text) it is re-decoded at a higher temperature, and we stop
    conditioning on the previous window (which is what propagates loops).
    The clip list (cache/redo_loops.csv) is chosen by the same rule for train and test:
    compression ratio > 2.4 or > 15 immediate repetitions/restarts per 100 words."""
    todo = pd.read_csv(ROOT / "cache" / "redo_loops.csv")
    df = pd.read_csv(OUT)
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for row in tqdm(todo.itertuples(), total=len(todo)):
        path = DATA / row.split / row.filename
        kw = dict(language="en", beam_size=5, vad_filter=False, initial_prompt=INITIAL_PROMPT,
                  condition_on_previous_text=False, compression_ratio_threshold=2.4,
                  temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        try:
            segments, info = model.transcribe(str(path), word_timestamps=True, **kw)
            decoded, fb = (list(segments), info), 3
        except IndexError:  # same alignment bug as above -> no word timestamps
            segments, info = model.transcribe(str(path), word_timestamps=False, **kw)
            decoded, fb = (list(segments), info), 4
        res = transcribe_one(model, path, decoded=decoded)
        res["fallback"] = fb
        idx = df.index[(df.filename == row.filename) & (df.split == row.split)]
        for k, v in res.items():
            df.loc[idx, k] = v
    df.to_csv(OUT, index=False)


def main():
    train = pd.read_csv(DATA / "train.csv").assign(split="train")
    test = pd.read_csv(DATA / "test.csv").assign(split="test")
    todo = pd.concat([train, test])[["filename", "split"]]

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, usecols=["filename", "split"])
                   .apply(lambda r: f"{r.split}/{r.filename}", axis=1))
    todo = todo[~(todo.split + "/" + todo.filename).isin(done)]
    print(f"{len(done)} already transcribed, {len(todo)} to go")
    if todo.empty:
        return

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for row in tqdm(todo.itertuples(), total=len(todo)):
        res = transcribe_one(model, DATA / row.split / row.filename)
        rec = pd.DataFrame([{"filename": row.filename, "split": row.split, **res}])
        rec.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean_pass()
    elif "--flag-loops" in sys.argv:
        flag_loops()
    elif "--redo-loops" in sys.argv:
        redo_loops()
    else:
        main()
