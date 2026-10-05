# Experiments log

All CV numbers: 5-fold StratifiedKFold (seed 42), out-of-fold (OOF).
**RMSE(nz)** = on the 732 non-zero clips (primary, test-like; see 2026-10-05 note on the
0.0 batch). RMSE(all) includes the 37 zero clips. `_dz` = trained without zero clips.

| Date | ID | Change | OOF RMSE(nz) | OOF Pearson(nz) | OOF RMSE(all) | Public LB |
|---|---|---|---|---|---|---|
| 2026-10-05 | 00 | Constant train mean (3.313) | 1.027 (expected) | — | 1.239 | **1.0050** |
| 2026-10-05 | — | Ridge, hand features | 0.790 | 0.641 | 0.848 | |
| 2026-10-05 | — | LightGBM, hand features | 0.793 | 0.633 | 0.826 | |
| 2026-10-05 | — | Ridge, hand features, train+test cropped to 45 s | 0.776 | 0.648 | 0.843 | |
| 2026-10-05 | — | LightGBM, hand + CoLA acceptability, `_dz` | 0.720 | 0.705 | 0.920 | |
| 2026-10-05 | — | SVR, mpnet sentence embedding | 0.773 | 0.652 | 1.038 | |
| 2026-10-05 | — | SVR, DeBERTa-v3 mean-pooled embedding, `_dz` | 0.617 | 0.794 | 0.898 | |
| 2026-10-05 | — | Ridge, DeBERTa + hand + CoLA, `_dz` | 0.602 | 0.805 | 0.841 | |
| 2026-10-05 | — | Fine-tuned roberta-base-CoLA (4 ep, seed 0), `_dz` | 0.687 | 0.781 | 1.006 | |
| 2026-10-05 | 02 | Ridge, WavLM-base layer 7 (mean pool) | 0.580* | — | 0.565 | |
| 2026-10-05 | — | SVR, WavLM-base L7-8 mean+std | 0.550 | 0.840 | 0.539 | |
| 2026-10-05 | 03 | SVR, WavLM-large L18-23 mean+std | 0.517 | 0.861 | 0.516 | |
| 2026-10-05 | — | SVR, Whisper-large-v3 encoder mean+std | 0.535 | 0.852 | 0.560 | |
| 2026-10-05 | 04 | Blend WavLM-large SVR + Whisper-enc SVR | 0.507 | 0.869 | — | |
| 2026-10-05 | — | SVR on all blocks (WavLM-L, Whisper-enc, DeBERTa, hand+CoLA) | 0.491 | 0.877 | 0.505 | |
| 2026-10-05 | — | Blend svr_all + WavLM-L + text models | 0.482 | 0.884 | — | |
| 2026-10-05 | — | ... + linear stretch (inner-CV estimate) | 0.474 | — | — | |
| 2026-10-05 | — | Blend svr_all + WavLM-L + ft-RoBERTa(s0) | 0.476 | 0.886 | — | |
| 2026-10-05 | — | Fine-tuned RoBERTa-CoLA, mean of 3 seeds, `_dz` | 0.684 | 0.782 | 1.008 | |
| 2026-10-05 | **final** | `src/final.py`: blend (svr_all .518, WavLM-L .335, ridge text .005, ft-RoBERTa .142) + stretch (slope 1.092) | **0.4705** | **0.886** | 0.504 | **0.3529** |

| 2026-10-05 | v2 | final + attention-pooling WavLM-large (option 2, 1 seed; alone 0.563, blend weight .066) | 0.4694 | 0.886 | 0.501 | **0.3507** |

Final model training (in-sample) RMSE: 0.101 (non-zero) / 0.118 (all).

Submission files: `00_mean_baseline` (LB 1.0050), `02_ridge_wavlm_L7`, `03_svr_wavlmlarge_L18-23`,
`04_blend_audio`, `final`.

\* approximate (recomputed under the non-zero metric later).

## Notes / findings

- **2026-10-05 — the 0.0 batch.** All 37 zero labels are ids 5037–5073 (other train ids
  0–784, test 0–215), louder recordings. Constant-mean LB = 1.005 matches the RMSE expected
  if test ~ non-zero train (1.027), not full train (1.238) → test has (almost) no zeros.
  Primary metric switched to RMSE(nz). Dropping zeros from training: no help for audio
  models (SVR WavLM-L 0.517 keep vs 0.533 drop), clear help for text models
  (DeBERTa+hand 0.632 → 0.602) → audio keeps them, text drops them.
- **Whisper repetition loops.** temperature=0 disables Whisper's compression-ratio guard;
  71 clips (67 train / 4 test) looped ("so, so, so…"). Re-decoded with temperature fallback
  + no conditioning on previous text. Fixed train/test drift in repetition features
  (restarts/100w: 3.9 vs 0.7 before → 0.48 vs 0.29 after).
- **WavLM layers.** Best layers are upper-middle (base 7–8/12, large 18–23/24); std
  pooling adds ~0.01–0.02.
- **DeBERTa-v3 fine-tuning collapsed** (NaN with AdamW eps=1e-8; constant predictions with
  eps=1e-6) on this torch 2.7/cu118 setup → switched to roberta-base-CoLA, which trains fine.
- **LightGBM crash** (access violation) when sklearn is imported first → import lightgbm first.
- Sample submission is stale; submissions use test.csv's 216 rows (accepted by Kaggle).
- **2026-10-05 — final public LB 0.3529** (submitted as submission.csv, identical to final.csv).
  The public LB is lower than CV (0.471). Relative to the constant baseline the test looks easier
  (0.353/1.005 = 0.35 vs CV 0.471/1.027 = 0.46), and the public split is small, so LB moves are
  noisy — keep choosing models by CV.
- **v2 LB 0.3507** (v1 0.3529). Public LB = ~60% of test (~130 clips); private = other 40%. v1/v2 test predictions differ by only 0.018 RMS, so the gain is within noise.
