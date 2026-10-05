"""Option 1: fine-tune the top 8 transformer layers (17-24) of WavLM-large for regression.

Trick that makes this feasible on an 8 GB laptop GPU: the frozen bottom of the network (CNN
feature encoder + layers 1-16) is run ONCE by frames.py and its output (hidden state 16) is
cached to disk. Training then only runs layers 17-24 + a small head, on 15 s random crops.

Details
 - WavLM's relative-position bias is computed by layer 1 and passed (ungated) to all later
   layers, which apply their own learned gate; we recompute it with layer 1's compute_bias.
 - Head: learned softmax-weighted sum of the 8 fine-tuned layer outputs (SUPERB-style) ->
   the attention-pooling head from attnpool.py.
 - Layer-wise learning rates: 2e-5 for pretrained layers, 1e-3 for the new head;
   fixed epochs, no early stopping on the validation fold; several seeds averaged.
 - Validation/test: each clip in the same 20 s chunks used for extraction (999 frames),
   chunk outputs concatenated, then pooled over the whole clip.
Saves cache/preds/ft_wavlmL_top8.npz and ft_wavlmL_top8_fulltrain.npy.

Usage: python src/finetune_audio.py [seeds]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import WavLMModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attnpool import AttnPool  # noqa: E402
from frames import frame_path  # noqa: E402
from train import LO, HI, N_FOLDS, PREDS, SEED, load_labels, metrics, strat_bins  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

DEV = "cuda"
NAME = "ft_wavlmL_top8"
FIRST = 16                 # cached hidden state index = input of encoder.layers[16]
EPOCHS, BS, CROP = 6, 4, 750
LR_ENC, LR_HEAD, WD = 2e-5, 1e-3, 1e-2
CHUNK_FRAMES = 999         # frames produced by one 20 s chunk (see frames.py)


class TopWavLM(nn.Module):
    def __init__(self):
        super().__init__()
        full = WavLMModel.from_pretrained("microsoft/wavlm-large")
        enc = full.encoder
        self.rel = enc.layers[0].attention          # owns rel_attn_embed / compute_bias
        self.layers = nn.ModuleList(enc.layers[FIRST:])
        self.heads = enc.layers[0].attention.num_heads
        n = len(self.layers)
        self.layer_w = nn.Parameter(torch.zeros(n))
        self.pool = AttnPool(G=1)
        for p in self.rel.parameters():
            p.requires_grad = False
        del full

    def encode(self, x):  # x: B,T,D -> B,T,D (weighted sum of top-layer outputs)
        B, T, _ = x.shape
        pb = self.rel.compute_bias(T, T)
        pb = pb.unsqueeze(0).repeat(B, 1, 1, 1).view(B * self.heads, T, T)
        outs = []
        for layer in self.layers:
            x, pb = layer(x, position_bias=pb)
            outs.append(x)
        w = torch.softmax(self.layer_w, 0)
        return (w[:, None, None, None] * torch.stack(outs)).sum(0)

    def forward(self, chunks_per_item):
        """chunks_per_item: list (per clip) of lists of tensors (T_i, D). Each chunk is
        encoded on its own, then frames of a clip are concatenated and pooled together."""
        feats = []
        for chunks in chunks_per_item:
            feats.append(torch.cat([self.encode(c[None])[0] for c in chunks], 0))
        T = max(f.shape[0] for f in feats)
        x = torch.zeros(len(feats), 1, T, feats[0].shape[1], device=DEV, dtype=feats[0].dtype)
        mask = torch.zeros(len(feats), T, dtype=torch.bool, device=DEV)
        for i, f in enumerate(feats):
            x[i, 0, :f.shape[0]] = f
            mask[i, :f.shape[0]] = True
        return self.pool(x, mask)


def load_chunks(split, name):
    x = np.load(frame_path(split, name, "l16"), mmap_mode="r")
    return [x[s:s + CHUNK_FRAMES] for s in range(0, x.shape[0], CHUNK_FRAMES)]


def random_crop(split, name, rng):
    chunks = load_chunks(split, name)
    long = [c for c in chunks if c.shape[0] >= CROP] or chunks
    c = long[rng.integers(len(long))]
    s = rng.integers(0, max(c.shape[0] - CROP, 0) + 1)
    return np.asarray(c[s:s + CROP], dtype=np.float32)


def to_dev(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(DEV)


@torch.no_grad()
def predict(model, split, names):
    model.eval()
    out = []
    for n in names:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model([[to_dev(c) for c in load_chunks(split, n)]]).float().item())
    return np.array(out)


def fit(names, y, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TopWavLM().to(DEV)
    enc_params = [p for p in model.layers.parameters() if p.requires_grad]
    head_params = [model.layer_w] + list(model.pool.parameters())
    opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENC},
                             {"params": head_params, "lr": LR_HEAD}], weight_decay=WD, eps=1e-6)
    steps = EPOCHS * int(np.ceil(len(names) / BS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_ENC, LR_HEAD], total_steps=steps,
                                              pct_start=0.1)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for b in np.array_split(rng.permutation(len(names)), int(np.ceil(len(names) / BS))):
            batch = [[to_dev(random_crop("train", names[i], rng))] for i in b]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(model(batch).float(), yt[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            losses.append(loss.item())
        print(f"   epoch {ep} train RMSE {np.sqrt(np.mean(losses)):.4f}", flush=True)
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
            del model
            torch.cuda.empty_cache()
        print(f"seed {seed} OOF", metrics(np.clip(oof[si], LO, HI), y), flush=True)
        np.savez(PREDS / f"{NAME}_s{seed}.npz", oof=np.clip(oof[si], LO, HI),
                 test=np.clip(tp[si], LO, HI), y=y)
        full = fit(names, y, SEED + 1000 * seed + 99)   # for the training-RMSE report
        ins[si] = predict(full, "train", names)
        del full
        torch.cuda.empty_cache()
    o, t = np.clip(oof.mean(0), LO, HI), np.clip(tp.mean(0), LO, HI)
    np.savez(PREDS / f"{NAME}.npz", oof=o, test=t, y=y)
    np.save(PREDS / f"{NAME}_fulltrain.npy", np.clip(ins.mean(0), LO, HI))
    print(NAME, f"seeds {seeds}", metrics(o, y))


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
