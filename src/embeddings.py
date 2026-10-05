"""Steps 5-6: pretrained embeddings, cached to cache/emb_<name>.npz (keys: filename, split, X).

Usage:  python src/embeddings.py mpnet | deberta | cola | wavlm

- mpnet   : sentence-transformers/all-mpnet-base-v2 sentence embedding of the transcript.
- deberta : microsoft/deberta-v3-base, mean-pooled last hidden state (masked mean).
- cola    : textattack/roberta-base-CoLA, a classifier fine-tuned on CoLA (grammatical
            acceptability judgements). We score each spaCy sentence and aggregate
            (mean/min/fraction unacceptable) -> a direct, interpretable grammar probe.
- wavlm   : microsoft/wavlm-base-plus on raw audio, mean-pooled hidden states of EVERY layer
            (shape N x 13 x 768) so we can compare middle vs last layer without recomputing.
            Audio is processed in 20 s chunks to fit 8 GB VRAM.
Runs in its own process (torch cu118); never imports faster-whisper.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def transcripts(crop_s=None):
    import json
    tr = pd.read_csv(CACHE / "transcripts.csv")
    tr["text"] = tr.text.fillna("")
    if crop_s is not None:
        def crop(r):
            if r.duration <= crop_s:
                return r.text
            return "".join(w[2] for w in json.loads(r.words) if w[1] <= crop_s).strip()
        tr["text"] = tr.apply(crop, axis=1)
    return tr


def save(name, tr, X):
    np.savez(CACHE / f"emb_{name}.npz", filename=tr.filename.values,
             split=tr.split.values, X=X.astype(np.float32))
    print(name, X.shape)


def mpnet(suffix="", crop_s=None):
    from sentence_transformers import SentenceTransformer
    tr = transcripts(crop_s)
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEV)
    m.max_seq_length = 512  # default 384 tokens could truncate long answers
    save("mpnet" + suffix, tr, m.encode(tr.text.tolist(), batch_size=16, show_progress_bar=True))


@torch.no_grad()
def deberta(suffix="", crop_s=None):
    from transformers import AutoModel, AutoTokenizer
    tr = transcripts(crop_s)
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    m = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(DEV).eval()
    out = []
    for i in tqdm(range(0, len(tr), 8)):
        b = tok(tr.text.iloc[i:i + 8].tolist(), padding=True, truncation=True,
                max_length=512, return_tensors="pt").to(DEV)
        h = m(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
    save("deberta" + suffix, tr, np.vstack(out))


@torch.no_grad()
def cola(suffix="", crop_s=None):
    import spacy
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tr = transcripts(crop_s)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    name = "textattack/roberta-base-CoLA"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).to(DEV).eval()
    feats = []
    for text in tqdm(tr.text.tolist()):
        sents = [s.text.strip() for s in nlp(text).sents if len(s.text.split()) >= 3]
        if not sents:
            feats.append([0.0, 0.0, 1.0, 0.0])
            continue
        b = tok(sents, padding=True, truncation=True, max_length=128,
                return_tensors="pt").to(DEV)
        p = torch.softmax(m(**b).logits, -1)[:, 1].cpu().numpy()  # P(acceptable)
        w = np.array([len(s.split()) for s in sents], dtype=float)
        feats.append([p.mean(), p.min(), (p < 0.5).mean(), np.average(p, weights=w)])
    save("cola" + suffix, tr, np.array(feats))


@torch.no_grad()
def wavlm(model_name="microsoft/wavlm-base-plus", out_name="wavlm"):
    """Saves emb_<out_name>.npz (mean pooled, N x L x D) and emb_<out_name>_std.npz (std)."""
    import soundfile as sf
    from transformers import AutoFeatureExtractor, WavLMModel
    # Ordering from the CSVs (not transcripts) so this can run while Whisper is still going.
    tr = pd.concat([pd.read_csv(ROOT / "data" / f"{s}.csv").assign(split=s)
                    for s in ("train", "test")])[["filename", "split"]]
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    m = WavLMModel.from_pretrained(model_name).to(DEV).eval().half()
    chunk = 20 * 16000
    means, stds = [], []
    for r in tqdm(tr.itertuples(), total=len(tr)):
        wav, sr = sf.read(ROOT / "data" / r.split / r.filename, dtype="float32")
        assert sr == 16000
        s1, s2, n = None, None, 0  # running sum and sum of squares over frames
        for s in range(0, len(wav), chunk):
            seg = wav[s:s + chunk]
            if len(seg) < 16000:  # skip <1 s tail
                continue
            x = fe(seg, sampling_rate=16000, return_tensors="pt").input_values.to(DEV).half()
            hs = m(x, output_hidden_states=True).hidden_states  # L x (1, T, D)
            h = torch.stack(hs, 0)[:, 0].float()                # L x T x D
            s1 = h.sum(1) if s1 is None else s1 + h.sum(1)
            s2 = (h ** 2).sum(1) if s2 is None else s2 + (h ** 2).sum(1)
            n += h.shape[1]
        mu = s1 / n
        means.append(mu.cpu().numpy())
        stds.append((s2 / n - mu ** 2).clamp(min=0).sqrt().cpu().numpy())
    save(out_name, tr, np.stack(means))
    save(out_name + "_std", tr, np.stack(stds))


if __name__ == "__main__":
    what = sys.argv[1]
    crop = "--crop45" in sys.argv
    if what == "wavlm":
        wavlm()
    elif what == "wavlm_large":
        wavlm("microsoft/wavlm-large", "wavlm_large")
    else:
        fn = {"mpnet": mpnet, "deberta": deberta, "cola": cola}[what]
        fn(suffix="_crop45" if crop else "", crop_s=45 if crop else None)
