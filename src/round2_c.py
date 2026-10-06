"""Round 2, quick CPU items on top of the new baseline (final_stack_pw components).

C1 prompt-frequency stack features: number of TRAINING clips in the clip's prompt cluster
   (log), rare flag. Counts are label-free, so they leak nothing; for train clips they are
   computed leave-one-out.
C2 KNN on text features (DeBERTa + hand + CoLA, standardised on the training fold),
   distance-weighted mean label of the 10 nearest training clips; out-of-fold, and checked
   under prompt-held-out folds as well.
Judged with the paired rule against the final_stack_pw component set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler, normalize  # noqa: E402
import gating as G  # noqa: E402

NEW = G.BASE + ["ridge_r2w_dz"]


def prompt_counts():
    train, test = T.load_labels()
    p = pd.read_csv(T.CACHE / "prompts.csv")
    ptr = train[["filename"]].merge(p[p.split == "train"], how="left").prompt.values
    cnt = pd.Series(ptr).value_counts()
    c_tr = np.array([cnt[k] - 1 for k in ptr])           # leave-one-out
    return c_tr


def main():
    train, test = T.load_labels()
    y = train.label.values
    base = G.per_split(NEW)
    np.save(T.CACHE / "baseline_pw_per_split.npy", base)
    print(f"new baseline (final_stack_pw) short {base[:, 0].mean():.4f} +/- {base[:, 0].std():.4f}")

    c = prompt_counts()
    extra = np.column_stack([np.log1p(c), (c < 30).astype(float)])
    extra = (extra - extra.mean(0)) / (extra.std(0) + 1e-9)
    G.compare(G.per_split(NEW, extra=extra), base, "C1 + prompt-frequency stack features")

    Htr, Hte, _ = T.hand_features()
    cola = T.load_emb("cola")
    D = T.load_emb("deberta")
    Xtr = np.hstack([D[0], Htr.values, cola[0]])
    Xte = np.hstack([D[1], Hte.values, cola[1]])
    nzmask = y > 0
    for mode, folder in [("random", "preds"), ("group", "preds_g")]:
        T.CV_MODE = mode
        oof, tp = np.zeros(len(y)), np.zeros(len(Xte))
        for tr, va in T.get_splits(y):
            tr = tr[nzmask[tr]]
            sc = StandardScaler().fit(Xtr[tr])
            A, Q, Z = normalize(sc.transform(Xtr[tr])), normalize(sc.transform(Xtr[va])), normalize(sc.transform(Xte))
            for idx_q, out, M in [(va, oof, Q), (None, None, Z)]:
                s = M @ A.T
                top = np.argsort(-s, 1)[:, :10]
                w = np.take_along_axis(s, top, 1).clip(min=0) + 1e-6
                pred = (w * y[tr][top]).sum(1) / w.sum(1)
                if idx_q is not None:
                    oof[va] = pred
                else:
                    tp += pred / T.N_FOLDS
        np.savez(T.CACHE / folder / "knn_text.npz", oof=oof, test=tp, y=y)
        print(f"knn_text alone ({mode}):", {k: round(v, 4) for k, v in T.metrics(oof, y).items()})
    T.CV_MODE = "random"
    G.compare(G.per_split(NEW + ["knn_text"]), base, "C2 + KNN on text features (random)")
    if (T.CACHE / "preds_g" / "ridge_r2w_dz.npz").exists():
        bg = G.per_split(NEW, folder=T.CACHE / "preds_g")
        G.compare(G.per_split(NEW + ["knn_text"], folder=T.CACHE / "preds_g"), bg,
                  "C2 + KNN on text features (prompt-held-out)")


if __name__ == "__main__":
    main()
