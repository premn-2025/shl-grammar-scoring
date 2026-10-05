"""Option 2: learned attention pooling over frozen WavLM-large frames (cache/frames/*.grp.npy).

Why: mean/std pooling treats every 40 ms frame as equally informative. Grammar evidence is
concentrated in some stretches (an error, a restart, a long clause), so we let a small
network learn *which frames to listen to*.

Model (~0.3M parameters, the 316M-parameter WavLM stays frozen):
  softmax-weighted sum of the 3 layer groups -> LayerNorm -> Linear(1024->256)+GELU
  -> attention scores (Linear-tanh-Linear) -> attention-weighted mean and std -> Linear -> score
Training: MSE, AdamW, cosine schedule, fixed epochs (no early stopping on the validation
fold), random 20 s crops each epoch (augmentation), full clips for validation/test,
several seeds averaged. Same 5 folds as everything else; 0.0 clips kept (audio model).

Usage: python src/attnpool.py [seeds]      e.g. python src/attnpool.py 0 1 2
Saves cache/preds/attnpool_wavlmL.npz (OOF/test) and attnpool_wavlmL_fulltrain.npy.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import frame_path  # noqa: E402
from train import LO, HI, N_FOLDS, PREDS, SEED, load_labels, metrics, strat_bins  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

DEV = "cuda"
NAME = "attnpool_wavlmL"
EPOCHS, BS, LR, WD = 25, 16, 1e-3, 1e-2
CROP = 500  # frames at 25 fps = 20 s


class AttnPool(nn.Module):
    def __init__(self, D=1024, G=3, H=256, drop=0.3):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(G))
        self.norm = nn.LayerNorm(D)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(D, H), nn.GELU())
        self.att = nn.Sequential(nn.Linear(H, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * H, 1))
        nn.init.constant_(self.head[1].bias, 3.3)  # start at the mean score

    def forward(self, x, mask):  # x: B,G,T,D  mask: B,T (True = real frame)
        w = torch.softmax(self.layer_w, 0)
        h = self.proj(self.norm((w[None, :, None, None] * x).sum(1)))      # B,T,H
        a = self.att(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, -1).unsqueeze(-1)                            # B,T,1
        mu = (a * h).sum(1)
        sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-5).sqrt()
        return self.head(torch.cat([mu, sd], -1)).squeeze(-1)


def load(split, name, crop=None, rng=None):
    x = np.load(frame_path(split, name, "grp"), mmap_mode="r")
    if crop is not None and x.shape[1] > crop:
        s = rng.integers(0, x.shape[1] - crop + 1)
        x = x[:, s:s + crop]
    return np.asarray(x, dtype=np.float32)


def collate(arrs):
    T = max(a.shape[1] for a in arrs)
    x = np.zeros((len(arrs), arrs[0].shape[0], T, arrs[0].shape[2]), np.float32)
    mask = np.zeros((len(arrs), T), bool)
    for i, a in enumerate(arrs):
        x[i, :, :a.shape[1]] = a
        mask[i, :a.shape[1]] = True
    return torch.from_numpy(x).to(DEV), torch.from_numpy(mask).to(DEV)


@torch.no_grad()
def predict(model, split, names):
    model.eval()
    out = []
    for i in range(0, len(names), 8):
        x, m = collate([load(split, n) for n in names[i:i + 8]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x, m).float().cpu().numpy())
    return np.concatenate(out)


def fit(names, y, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AttnPool().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * int(np.ceil(len(names) / BS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    for ep in range(EPOCHS):
        model.train()
        for b in np.array_split(rng.permutation(len(names)), int(np.ceil(len(names) / BS))):
            x, m = collate([load("train", names[i], CROP, rng) for i in b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(model(x, m).float(), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    return model


def main(seeds):
    train, test = load_labels()
    y = train.label.values
    names, tnames = train.filename.values, test.filename.values
    oof = np.zeros((len(seeds), len(y)))
    tp = np.zeros((len(seeds), len(tnames)))
    ins = np.zeros((len(seeds), len(y)))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for si, seed in enumerate(seeds):
        for fold, (tr, va) in enumerate(skf.split(names, strat_bins(y))):
            model = fit(names[tr], y[tr], SEED + 1000 * seed + fold)
            oof[si, va] = predict(model, "train", names[va])
            tp[si] += predict(model, "test", tnames) / N_FOLDS
            print(f"seed {seed} fold {fold}", {k: round(v, 4) for k, v in
                                              metrics(np.clip(oof[si, va], LO, HI), y[va]).items()},
                  flush=True)
        print(f"seed {seed} OOF", metrics(np.clip(oof[si], LO, HI), y), flush=True)
        full = fit(names, y, SEED + 1000 * seed + 99)   # for the training-RMSE report
        ins[si] = predict(full, "train", names)
    o, t = np.clip(oof.mean(0), LO, HI), np.clip(tp.mean(0), LO, HI)
    np.savez(PREDS / f"{NAME}.npz", oof=o, test=t, y=y)
    np.save(PREDS / f"{NAME}_fulltrain.npy", np.clip(ins.mean(0), LO, HI))
    print(NAME, f"seeds {seeds}", metrics(o, y))


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
