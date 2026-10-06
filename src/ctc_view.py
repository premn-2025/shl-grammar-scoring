"""Second ASR "view": literal CTC transcript with NO language model + cross-ASR features.

facebook/wav2vec2-large-960h-lv60-self decodes characters frame by frame (greedy CTC), so it
cannot 'repair' ungrammatical speech the way Whisper's language model does; where the two
systems disagree, the speech was hard to fit into well-formed English.
Features (cache/features_ctc.csv):
  ctc_sub/del/ins_per100, ctc_similarity  word-level alignment vs Whisper verbatim transcript
  ctc_lt_per100                            LanguageTool grammar errors per 100 words on the
                                           literal text (punctuation/casing rules excluded)
  ctc_conf_mean, ctc_lowconf_frac          mean max-softmax probability of non-blank frames
  ctc_wpm                                  words per minute in the literal transcript
Usage: python src/ctc_view.py          (GPU transcription, then CPU features)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTC_TXT = ROOT / "cache" / "transcripts_ctc.csv"
OUT = ROOT / "cache" / "features_ctc.csv"
MODEL = "facebook/wav2vec2-large-960h-lv60-self"


def transcribe():
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    m = Wav2Vec2ForCTC.from_pretrained(MODEL).to("cuda").eval().half()
    clips = pd.concat([pd.read_csv(ROOT / "data" / f"{s}.csv").assign(split=s)
                       for s in ("train", "test")])[["filename", "split"]]
    rows = []
    for k, r in enumerate(clips.itertuples()):
        if k % 100 == 0:
            print(f"ctc progress {k}/{len(clips)}", flush=True)
        wav, _ = sf.read(ROOT / "data" / r.split / r.filename, dtype="float32")
        texts, confs = [], []
        for s in range(0, len(wav), 20 * 16000):
            seg = wav[s:s + 20 * 16000]
            if len(seg) < 8000:
                continue
            x = proc(seg, sampling_rate=16000, return_tensors="pt").input_values.to("cuda").half()
            with torch.no_grad():
                logits = m(x).logits[0].float()
            p = torch.softmax(logits, -1)
            ids = logits.argmax(-1)
            nonblank = ids != proc.tokenizer.pad_token_id
            confs.extend(p.max(-1).values[nonblank].cpu().tolist())
            texts.append(proc.decode(ids))
        rows.append({"filename": r.filename, "split": r.split, "text": " ".join(texts).lower(),
                     "conf_mean": float(np.mean(confs)) if confs else 0.0,
                     "lowconf_frac": float(np.mean(np.array(confs) < 0.5)) if confs else 0.0})
    pd.DataFrame(rows).to_csv(CTC_TXT, index=False)


def features():
    import language_tool_python
    sys.path.insert(0, str(ROOT / "src"))
    from whisper_diff import diff_feats
    v = pd.read_csv(ROOT / "cache" / "transcripts.csv").fillna({"text": ""})
    c = pd.read_csv(CTC_TXT).fillna({"text": ""})
    m = v.merge(c, on=["filename", "split"], suffixes=("_w", "_c"))
    tool = language_tool_python.LanguageTool("en-US")
    ignored = {"PUNCTUATION", "CASING", "TYPOGRAPHY"}
    rows = []
    for r in m.itertuples():
        d = diff_feats(r.text_w, r.text_c)
        words = re.findall(r"[a-z']+", r.text_c)
        n = max(len(words), 1)
        errs = [x for x in tool.check(r.text_c) if (getattr(x, "category", "") or "") not in ignored] \
            if r.text_c.strip() else []
        rows.append({"filename": r.filename, "split": r.split,
                     "ctc_sub_per100": d["diff_sub_per100"], "ctc_del_per100": d["diff_del_per100"],
                     "ctc_ins_per100": d["diff_ins_per100"], "ctc_similarity": d["diff_similarity"],
                     "ctc_lt_per100": 100 * len(errs) / n, "ctc_conf_mean": r.conf_mean,
                     "ctc_lowconf_frac": r.lowconf_frac, "ctc_wpm": n / max(r.duration, 1) * 60})
    tool.close()
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    if not CTC_TXT.exists():
        transcribe()
    features()
