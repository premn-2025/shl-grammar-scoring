"""Round 2, C: small mid-fusion MLP (audio + text + hand features in one network).

Inputs (standardised on the training fold, then PCA-reduced on the fold to keep the net
small): WavLM-large L18-23 mean+std, DeBERTa embedding, hand + CoLA + round-2 features.
Net: 256-d input -> 32 -> 8 -> 1, ReLU, dropout 0.4/0.3, smooth-L1 loss, weight decay 1e-2,
fixed 60 epochs (no early stopping on the validation fold), 3 seeds averaged, CPU.
Trained on non-zero clips. Saved as a stack component: cache/<preds>/mlp_fusion.npz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from round2_b import OUT, block  # noqa: E402

torch.set_num_threads(6)


def net(d):
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.4),
                         nn.Linear(32, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 1))


def fit_predict(Xa, ya, Xbs, seed):
    torch.manual_seed(seed)
    m = net(Xa.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2)
    Xt = torch.tensor(Xa, dtype=torch.float32)
    yt = torch.tensor(ya, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for ep in range(60):
        m.train()
        for idx in torch.randperm(len(Xt), generator=g).split(32):
            loss = nn.functional.smooth_l1_loss(m(Xt[idx]).squeeze(-1), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return [m(torch.tensor(Xb, dtype=torch.float32)).squeeze(-1).numpy() for Xb in Xbs]


def main():
    train, _ = T.load_labels()
    y = train.label.values
    Htr, Hte, _ = T.hand_features()
    cola = T.load_emb("cola")
    D = T.load_emb("deberta")
    _, W = T.audio_pool("wavlm_large", list(range(18, 24)))
    a, b, _ = block(OUT)
    lo, hi = np.percentile(a, 1, 0), np.percentile(a, 99, 0)
    a, b = np.clip(a, lo, hi), np.clip(b, lo, hi)
    blocks_tr = [W[0], D[0], np.hstack([Htr.values, cola[0], a])]
    blocks_te = [W[1], D[1], np.hstack([Hte.values, cola[1], b])]
    ncomp = [128, 64, 64]
    oof, tp = np.zeros(len(y)), np.zeros(len(blocks_te[0]))
    for tr, va in T.get_splits(y):
        tr = tr[y[tr] > 0]
        Za, Zv, Zt = [], [], []
        for Btr, Bte, k in zip(blocks_tr, blocks_te, ncomp):
            sc = StandardScaler().fit(Btr[tr])
            pca = PCA(min(k, Btr.shape[1]), random_state=0).fit(sc.transform(Btr[tr]))
            f = lambda X: pca.transform(sc.transform(X)) / np.sqrt(pca.explained_variance_[:1])
            Za.append(f(Btr[tr])); Zv.append(f(Btr[va])); Zt.append(f(Bte))
        Za, Zv, Zt = np.hstack(Za), np.hstack(Zv), np.hstack(Zt)
        s2 = StandardScaler().fit(Za)
        Za, Zv, Zt = s2.transform(Za), s2.transform(Zv), s2.transform(Zt)
        pv, pt = 0, 0
        for seed in (0, 1, 2):
            v, t = fit_predict(Za, y[tr], [Zv, Zt], seed)
            pv, pt = pv + v / 3, pt + t / 3
        oof[va] = pv
        tp += pt / T.N_FOLDS
    oof, tp = np.clip(oof, T.LO, T.HI), np.clip(tp, T.LO, T.HI)
    np.savez(T.PREDS / "mlp_fusion.npz", oof=oof, test=tp, y=y)
    print(f"mlp_fusion ({T.CV_MODE}):", {k: round(v, 4) for k, v in T.metrics(oof, y).items()}, flush=True)


if __name__ == "__main__":
    main()
