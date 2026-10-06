"""Fine-tune a pretrained text encoder (default microsoft/deberta-v3-base) to regress the
grammar score from the transcript, inside the same 5-fold CV as everything else.

Why fine-tune rather than frozen embeddings? Frozen sentence embeddings mostly encode *topic*
/ meaning, which is unrelated to grammar; fine-tuning lets the encoder re-focus on form.

Choices (kept simple and defensible):
 - target scaled to [0,1] (y/5), MSE loss, linear head on mean-pooled tokens;
 - fixed number of epochs (no best-epoch selection on the validation fold -> no leakage);
 - lr 2e-5, batch 8, warmup 10%, bf16 autocast; several seeds can be averaged.
Saves cache/preds/ft_<tag>.npz like every other model.

Usage: python src/finetune_text.py [model_name] [epochs] [seed]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (CACHE, LO, HI, N_FOLDS, PREDS, SEED, get_splits, load_labels, metrics,  # noqa
                   rmse, strat_bins)

DEV = "cuda"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/deberta-v3-base"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
RUN_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
import os  # noqa: E402
# Batch size / learning rate can be overridden for large models (FT_BS=4, FT_LR=1e-5 fit 8 GB).
MAX_LEN = 256
BS = int(os.environ.get("FT_BS", 8))
LR = float(os.environ.get("FT_LR", 2e-5))
AMP = os.environ.get("FT_FP32") != "1"           # bf16 autocast unless FT_FP32=1
MAX_FOLDS = int(os.environ.get("FT_FOLDS", N_FOLDS))  # debug: run only the first k folds


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)
        # Start predicting the average scaled score (~0.66) so early training is stable.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.66)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1)).squeeze(-1)


def batches(enc, idx, y=None, shuffle=False, rng=None):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        out = {k: v[b].to(DEV) for k, v in enc.items()}
        yield out, (torch.tensor(y[b], dtype=torch.float32, device=DEV) if y is not None else None)


@torch.no_grad()
def predict(model, enc, idx):
    model.eval()
    out = []
    for b, _ in batches(enc, idx):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            out.append(model(**b).float().cpu().numpy())
    return np.concatenate(out) * 5


def train_model(enc_tr, tri, y, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Regressor().to(DEV)
    # eps=1e-6 (DeBERTa's own fine-tuning setting). With the default 1e-8, the first
    # AdamW step turned the embedding weights into NaN on this setup.
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, eps=1e-6)
    steps = EPOCHS * int(np.ceil(len(tri) / BS))
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for ep in range(EPOCHS):
        model.train()
        for b, yb in batches(enc_tr, tri, y / 5, shuffle=True, rng=rng):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                loss = nn.functional.mse_loss(model(**b).float(), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
    return model


def full_fit(enc_tr, y):
    """Train once on ALL non-zero training clips and predict the training set in-sample.
    Only used to report the compulsory *training* RMSE of the final model."""
    tri = np.where(y > 0)[0]
    model = train_model(enc_tr, tri, y, SEED + RUN_SEED * 100 + 99)
    p = np.clip(predict(model, enc_tr, np.arange(len(y))), LO, HI)
    tag = f"ft_{MODEL.split('/')[-1]}_e{EPOCHS}_s{RUN_SEED}"
    np.save(PREDS / f"{tag}_fulltrain.npy", p)
    print(tag, "in-sample", metrics(p, y))


def main():
    train, test = load_labels()
    tr = pd.read_csv(CACHE / "transcripts.csv")
    tr["text"] = tr.text.fillna("")
    txt = lambda split, names: tr[tr.split == split].set_index("filename").loc[names, "text"].tolist()
    texts_tr, texts_te = txt("train", train.filename), txt("test", test.filename)
    y = train.label.values

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc_tr = dict(tok(texts_tr, padding="max_length", truncation=True, max_length=MAX_LEN,
                      return_tensors="pt"))
    enc_te = dict(tok(texts_te, padding="max_length", truncation=True, max_length=MAX_LEN,
                      return_tensors="pt"))
    enc_tr = {k: enc_tr[k] for k in ("input_ids", "attention_mask")}
    enc_te = {k: enc_te[k] for k in ("input_ids", "attention_mask")}

    if os.environ.get("FT_FULL") == "1":
        return full_fit(enc_tr, y)

    oof, test_pred = np.zeros(len(y)), np.zeros(len(texts_te))
    for fold, (tri, vai) in enumerate(get_splits(y)):
        if fold >= MAX_FOLDS:
            break
        # Text models never see the label-0.0 batch: its transcripts read like normal 2-3
        # answers (the zero is a property of the recording batch, not the words), so training
        # on them only teaches the text model noise. Audio models handle that batch.
        tri = tri[y[tri] > 0]
        nzv = y[vai] > 0
        torch.manual_seed(SEED + RUN_SEED * 100 + fold)
        rng = np.random.default_rng(SEED + RUN_SEED * 100 + fold)
        model = Regressor().to(DEV)
        # eps=1e-6 (DeBERTa's own fine-tuning setting). With the default 1e-8, the first
        # AdamW step turned the embedding weights into NaN on this setup.
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, eps=1e-6)
        steps = EPOCHS * int(np.ceil(len(tri) / BS))
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        for ep in range(EPOCHS):
            model.train()
            losses = []
            for b, yb in batches(enc_tr, tri, y / 5, shuffle=True, rng=rng):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                    loss = nn.functional.mse_loss(model(**b).float(), yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                losses.append(loss.item())
            p = predict(model, enc_tr, vai)
            print(f"fold {fold} epoch {ep} train RMSE {np.sqrt(np.nanmean(losses)) * 5:.4f} "
                  f"(nan steps {int(np.isnan(losses).sum())})  val RMSE "
                  f"(non-zero) {rmse(np.clip(p, LO, HI)[nzv], y[vai][nzv]):.4f}  "
                  f"val pred std {p.std():.3f}", flush=True)
        oof[vai] = p
        test_pred += predict(model, enc_te, np.arange(len(texts_te))) / N_FOLDS
        del model, opt
        torch.cuda.empty_cache()

    oof, test_pred = np.clip(oof, LO, HI), np.clip(test_pred, LO, HI)
    tag = f"ft_{MODEL.split('/')[-1]}_e{EPOCHS}_s{RUN_SEED}"
    if MAX_FOLDS < N_FOLDS:  # debug run: don't save partial OOF
        return
    np.savez(PREDS / f"{tag}.npz", oof=oof, test=test_pred, y=y)
    print(tag, metrics(oof, y))


if __name__ == "__main__":
    main()
