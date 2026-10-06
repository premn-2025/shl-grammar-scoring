"""LoRA fine-tuning of the top 8 Whisper-large-v3 encoder layers (25-32) for grammar scoring.

Why the top layers: the layer scan showed Whisper's grammar-relevant information peaks at
layers 29-31, and caching hidden state 24 (whisper_cache.py) makes training cheap on 8 GB.

LoRA (low-rank adaptation): each frozen weight W of the attention query/value projections
gets a trainable update B @ A (rank r=16), so W x + (alpha/r) * B A x. Only ~1% of the
layer parameters train, which suits 769 labelled clips (less overfitting than full
fine-tuning) and keeps memory small. B starts at zero, so training starts from pretrained
Whisper exactly.

Head: softmax-weighted sum of the 8 layer outputs -> attention pooling (attnpool.AttnPool)
over the real (non-padding) frames of all 30 s windows of a clip -> score.
Training: one random window per clip per step, fixed epochs, same 5 folds as everything else,
0.0 clips kept (audio model). Saves cache/preds/<NAME>_s<seed>.npz (+ _fulltrain.npy).

Usage: python src/finetune_whisper.py [seeds]      e.g. python src/finetune_whisper.py 0
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attnpool import AttnPool  # noqa: E402
from train import LO, HI, PREDS, SEED, get_splits, load_labels, metrics  # noqa: E402
from whisper_cache import LAYER, OUT, path  # noqa: E402

DEV = "cuda"
EPOCHS = int(os.environ.get("FW_EPOCHS", 6))
BS = int(os.environ.get("FW_BS", 4))
LR_LORA, LR_HEAD, WD = float(os.environ.get("FW_LR", 2e-4)), 1e-3, 1e-2
RANK, ALPHA = 16, 32
NAME = "ft_whisper_lora"


class LoRALinear(nn.Module):
    """Frozen nn.Linear plus a trainable low-rank update (alpha/r) * B(A(x))."""

    def __init__(self, base: nn.Linear, r=RANK, alpha=ALPHA, dropout=0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)          # start exactly at the pretrained model
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(self.drop(x)))


class TopWhisper(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import WhisperModel
        # Load in fp16 (the full fp32 model would need ~6 GB RAM), keep only the top 8 layers
        # and convert those to fp32 for stable training (bf16 autocast is used for speed).
        enc = WhisperModel.from_pretrained("openai/whisper-large-v3",
                                           torch_dtype=torch.float16).encoder
        self.layers = nn.ModuleList(enc.layers[LAYER:]).float()
        self.final_norm = enc.layer_norm.float()
        for p in list(self.layers.parameters()) + list(self.final_norm.parameters()):
            p.requires_grad = False
        for layer in self.layers:               # LoRA on attention query and value
            layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj)
            layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj)
        self.layer_w = nn.Parameter(torch.zeros(len(self.layers)))
        self.pool = AttnPool(D=enc.config.d_model, G=1)
        del enc

    def encode(self, x):                         # x: (B, 1500, 1280) hidden state 24
        outs = []
        for i, layer in enumerate(self.layers):
            o = layer(x, attention_mask=None)
            x = o[0] if isinstance(o, tuple) else o
            outs.append(self.final_norm(x) if i == len(self.layers) - 1 else x)
        w = torch.softmax(self.layer_w, 0)
        return (w[:, None, None, None] * torch.stack(outs)).sum(0)   # (B, 1500, D)

    def forward(self, items):
        """items: list (per clip) of (windows tensor (n,1500,D), valid frame counts list)."""
        feats = []
        for wins, valid in items:
            h = self.encode(wins)
            feats.append(torch.cat([h[i, :v] for i, v in enumerate(valid)], 0))
        T = max(f.shape[0] for f in feats)
        x = torch.zeros(len(feats), 1, T, feats[0].shape[1], device=DEV, dtype=feats[0].dtype)
        mask = torch.zeros(len(feats), T, dtype=torch.bool, device=DEV)
        for i, f in enumerate(feats):
            x[i, 0, :f.shape[0]] = f
            mask[i, :f.shape[0]] = True
        return self.pool(x, mask)


VALID = None


def valid_frames(split, name):
    global VALID
    if VALID is None:
        v = pd.read_csv(OUT / "valid_frames.csv")
        VALID = {(r.split, r.filename): [int(t) for t in str(r.valid).split()] for r in v.itertuples()}
    return VALID[(split, name)]


def load(split, name, rng=None):
    w = np.load(path(split, name), mmap_mode="r")
    v = valid_frames(split, name)
    if rng is not None:                          # training: one random window (prefer >=10 s)
        ok = [i for i, n in enumerate(v) if n >= 500] or list(range(len(v)))
        i = ok[rng.integers(len(ok))]
        return torch.from_numpy(np.asarray(w[i:i + 1], dtype=np.float32)).to(DEV), [v[i]]
    return torch.from_numpy(np.asarray(w, dtype=np.float32)).to(DEV), v


@torch.no_grad()
def predict(model, split, names):
    model.eval()
    out = []
    for n in names:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model([load(split, n)]).float().item())
    return np.array(out)


def fit(names, y, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TopWhisper().to(DEV)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and (".A." in n or ".B." in n)]
    head = [model.layer_w] + list(model.pool.parameters())
    opt = torch.optim.AdamW([{"params": lora, "lr": LR_LORA}, {"params": head, "lr": LR_HEAD}],
                            weight_decay=WD, eps=1e-6)
    steps = EPOCHS * int(np.ceil(len(names) / BS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_LORA, LR_HEAD], total_steps=steps,
                                              pct_start=0.1)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for b in np.array_split(rng.permutation(len(names)), int(np.ceil(len(names) / BS))):
            batch = [load("train", names[i], rng) for i in b]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(model(batch).float(), yt[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sch.step()
            losses.append(loss.item())
        print(f"   epoch {ep} train RMSE {np.sqrt(np.mean(losses)):.4f}", flush=True)
    return model


def main(seeds):
    train, test = load_labels()
    y = train.label.values
    names, tnames = train.filename.values, test.filename.values
    for seed in seeds:
        oof, tp = np.zeros(len(y)), np.zeros(len(tnames))
        for fold, (tr, va) in enumerate(get_splits(y)):
            model = fit(names[tr], y[tr], SEED + 1000 * seed + fold)
            oof[va] = predict(model, "train", names[va])
            tp += predict(model, "test", tnames) / len(get_splits(y))
            print(f"seed {seed} fold {fold}",
                  {k: round(v, 4) for k, v in metrics(np.clip(oof[va], LO, HI), y[va]).items()}, flush=True)
            del model
            torch.cuda.empty_cache()
        o, t = np.clip(oof, LO, HI), np.clip(tp, LO, HI)
        np.savez(PREDS / f"{NAME}_s{seed}.npz", oof=o, test=t, y=y)
        print(f"{NAME}_s{seed}", metrics(o, y), flush=True)
        full = fit(names, y, SEED + 1000 * seed + 99)       # for the training-RMSE report
        np.save(PREDS / f"{NAME}_s{seed}_fulltrain.npy", np.clip(predict(full, "train", names), LO, HI))
        del full
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if os.environ.get("FW_FOLDS"):               # quick check: first fold only, no saving
        train, _ = load_labels()
        y = train.label.values
        tr, va = get_splits(y)[0]
        names = train.filename.values
        m = fit(names[tr], y[tr], SEED)
        p = predict(m, "train", names[va])
        print("fold 0 check", {k: round(v, 4) for k, v in metrics(np.clip(p, LO, HI), y[va]).items()})
    else:
        main([int(s) for s in sys.argv[1:]] or [0])
