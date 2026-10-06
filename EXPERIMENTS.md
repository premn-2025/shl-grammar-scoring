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

| 2026-10-05 | stack | Duration-aware Ridge stack (5 components + duration/wpm/word-prob + pred x duration, duration clamped to 20-61 s) | 0.4674 (nested) | - | - | **0.3440** |
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
- **2026-10-05 late: ideas tried.** Clip to [1,5]: no-op (min test pred 1.83). Crop augmentation + TTA: 0.522 alone, no blend gain. Nested SVR tuning: worse (audio SVR 0.526 vs 0.517 default; per-fold choices unstable). Duration-aware stacking: nested 0.4674 vs 0.4713, and 0.518 vs 0.525 on short (<50 s, test-like) clips -> **LB 0.3440** (from 0.3507).
- Why duration matters: 40-50 s train clips have mean label 3.06 vs 3.58 for 60 s clips, and 141/216 test clips are 40-50 s. Predictions are calibrated (true-on-pred slope 1.01-1.04 in every duration bin), so no extra stretch.
- Public-LB noise: ~130 clips at RMSE ~0.35 -> roughly +/-0.02 standard error.
- **2026-10-06: prompt shift found.** KMeans(12) on transcript embeddings (labels unused) -> prompt groups. The test is dominated by prompts rare/absent in train (one group: 22 test / 0 train; another 57 test / 14 train). Random-fold CV mixes prompts, so it over-rates topic-dependent text models.
- Prompt-held-out CV (StratifiedGroupKFold by prompt, CV_MODE=group, preds in cache/preds_g): audio SVR 0.517 -> 0.520 (robust); joint SVR 0.491 -> 0.507; DeBERTa+hand Ridge 0.602 -> 0.664 (topic shortcut); fine-tuned RoBERTa-CoLA 0.684 -> ~0.69 (learned grammar, not topic).
- Stacker weights learned on prompt-held-out OOF vs random OOF, both scored on held-out prompts: new 0.4836 / short 0.5739, old 0.4886 / short 0.5648, **average 0.4838 / short 0.5663** -> inal_stack_avg.csv (avg of final_stack and final_stack_g).
- Overnight queue: longer audio fine-tune fixed underfitting (0.559 -> 0.533 alone) but adds nothing to the stack; diff (verbatim vs clean Whisper) and LLM-judge features add nothing to the joint SVR (0.4912 vs 0.4906). Not used.
- **LB results 2026-10-06:** final_stack_avg 0.3472, final_stack_g 0.3522, final_v3 0.3519; final_stack (0.3440) stays best. The LB order matches the order of nested CV on SHORT (<50 s) clips (0.5648 < 0.5663 < 0.5739), not the all-clip or unseen-prompt order -> **select models by short-clip CV** (the test is mostly 45 s answers).
- **Importance weighting** short clips in the stacker (x2/x3/x5): no gain (short 0.5227/0.5227/0.5218 vs 0.5220) - duration interactions already handle the shift.
- **RoBERTa-large-CoLA** (cointegrated/roberta-large-cola-krishna2020, bs 4, lr 1e-5, 4 ep, w/o 0.0 clips): alone 0.661 random-CV / 0.682 prompt-held-out (base 3-seed: 0.684 / ~0.69); short clips alone 0.633 vs 0.688. But in the stack (1 seed) short-clip CV gets worse (0.5349 vs 0.5214).
- **Seed noise dominates:** swapping the 3-seed base RoBERTa for one single base seed moves stack short-clip CV between 0.5155 and 0.5292. Single-seed comparisons are inconclusive; never pick the best seed (that overfits CV). Running 2 more large seeds for a fair 3-vs-3 comparison.
- **RoBERTa-large, 3 seeds (fair test):** alone 0.640 all / **0.621 short** (base 3-seed: 0.684 / 0.688). In the stack (8 fold splits): replace base -> short 0.5299 vs 0.5212; add alongside -> 0.5220; 6-model text average -> 0.5233. Consistently worse. Cause: the large model is more correlated with svr_all (0.859 vs 0.847), so it adds less new information - ensemble diversity beats individual strength. **final_stack (LB 0.3440) remains the final model.**
- **Whisper-large-v3 all-layer scan** (HF encoder, mean+std per layer, src/whisper_layers.py): unlike WavLM, the best layers are near the top (Ridge: L31 0.543, L29 0.551, last L32 0.550; early layers 0.66-0.84). SVR on mean of L29-31: **0.527** alone (last-layer CTranslate2 SVR: 0.535).
- In the stack (8 fold splits, short-clip CV): + svr_whisperL29_31 -> all 0.4652 / **short 0.5086** (current 0.4686 / 0.5212); replacing whisper_enc inside svr_all -> 0.4647 / 0.5169; both -> 0.4636 / 0.5092. Chose the simplest: add one component -> inal_stack_w.csv (nested 0.4637 / short 0.5052, train RMSE 0.137).
- **final_stack_w LB 0.3432** (new best; final_stack 0.3440). Short-clip CV ranking again matched the LB direction. Final model = duration-aware stack of 6 components incl. Whisper L29-31 SVR.
## 2026-10-06 - structured experiment list (judged on 8-split short-clip CV; current 0.5086)
- **EXP1 gating** (prompt novelty = 1 - max cosine sim of mpnet transcript embedding to train clips; model disagreement = std of component preds; interactions with preds): all variants 0.5083-0.5092 -> no gain. Reason: 42% of test clips are more novel than 95% of train clips, so trust-for-new-prompts cannot be learned from train (it would extrapolate).
- **EXP2 speakers:** low-layer WavLM voice embeddings show repeated speakers inside train (18% have a near-duplicate voice, sim>0.9; their labels differ by only 0.27 vs 1.16 for random pairs) but NO test speaker appears in train (max sim 0.847). Speaker-grouped CV (CV_MODE=speaker, cache/speakers.csv, threshold 0.85): audio models inflated (WavLM-L SVR 0.517 -> 0.554, Whisper L29-31 0.527 -> 0.556, svr_all 0.491 -> 0.505), text barely (Ridge 0.602 -> 0.606, RoBERTa 0.684 -> ~0.69). But scored on unseen speakers, the current stack weights are still best (short 0.5183 vs 0.5207 speaker-grouped weights, 0.5193 average) -> final_stack_w is robust to new speakers. Honest new-speaker estimate: all 0.490 / short 0.518.
- **EXP3 error types** (LanguageTool rule IDs -> top-25 rules + groups: tense/verb-form r=-0.18, agreement -0.18, article -0.13, plural -0.11, pronoun -0.10): in svr_all 0.5087, in text Ridge 0.5079, as own component 0.5094 -> no gain.
- **EXP4 ordinal regression** (8 cumulative logistic classifiers P(y>=k), k=1.5..5, on PCA-128 of joint features): alone 0.530; in stack 0.5085 -> no gain.
- Conclusion: final_stack_w (LB 0.3432) stays. Remaining gap is within public-LB noise.
## 2026-10-06 - final round (paired rule: same 8 stacker splits as baseline; keep only if wins >= 6/8 AND mean gain > sd of differences)
- **Baseline** final_stack_w per split (short): 0.5052 0.5075 0.5152 0.5061 0.5114 0.5079 0.5067 0.5089 -> 0.5086 +/- 0.0031 (all 0.4652).
- **1. Repeated-CV bagging** (SVR/Ridge components over fold seeds 42,1,2,3,4; OOF averaged, test = 25 fold models): 0/8 wins (short 0.5128). Caveat: seed 42 was the luckiest split for the audio SVRs (WavLM-L 0.517 vs 0.528-0.543 on seeds 1-4), so the baseline CV is ~0.004 optimistic and CV cannot judge bagging fairly. Bagged test preds differ from final_stack_w by only 0.009 RMS (max 0.037) -> below LB noise, not submitted. Confirms the submission is stable, not a split fluke.
- **2. HuBERT-large** (facebook/hubert-large-ll60k): layer scan peaks upper-middle (best L21 0.559). SVR L16-21 0.5335 alone; joint SVR with HuBERT 0.4865 (vs 0.4906). Stack: +component 5/8 wins (+0.0010, sd 0.0014), in joint SVR 7/8 (+0.0002, sd 0.0002), both 5/8 -> reject.
- **3a. Transductive scaling** (standardise on train+test features, no labels; no per-fold scaler): all variants within +/-0.0002 -> reject.
- **3b. Leak-free pseudo-labelling** (per fold: fit, pseudo-label test, refit with weight 0.5): joint SVR 0.4911; stack 0/8 and 1/8 wins -> reject.
- **Outcome:** final_stack_w (LB 0.3432) is final. 0 of 3 allowed LB submissions used this round.
- **Whisper-large-v3 LoRA** (src/whisper_cache.py + src/finetune_whisper.py): cache hidden state 24 per 30 s window (bug caught by a reconstruction check: HF applies the final layer_norm to the LAST hidden state, so the cache must keep LAYER+1 layers); LoRA r=16 on q/v of layers 25-32 (1.0M of 158M params trainable) + layer weighting + attention pooling; 6 epochs, 5 folds, 3 seeds. Alone: 0.5315 / 0.5315 / 0.5327 per seed, **0.5258 3-seed avg** (fold 0: 0.535 vs 0.551 for the frozen Whisper SVR). Correlation with the audio SVRs ~0.93.
- Paired stack test: seed 0 6/8 (+0.0008, sd 0.0010); 2-seed 7/8 (+0.0017, sd 0.0013, passes); **3-seed 7/8 (+0.0013, sd 0.0015) -> borderline reject** (not cherry-picking the 2-seed result). Candidate file inal_stack_wl.csv built (nested 0.4639 / short 0.5073, train 0.156); test preds differ from final_stack_w by 0.033 RMS (max 0.10).
- **final_stack_wl LB 0.3447** (final_stack_w 0.3432): within public-LB noise, consistent with the borderline paired test. final_stack_w stays the primary submission.
## 2026-10-06 - round 2 (stricter rule: gain >= 0.0015, >= 6/8 wins, prompt-held-out must not degrade, no speaker-leak dependence)
- **Group A, all rejected:** A1 shrinking novel-prompt clips to the mean (fitted lambda = 0); A2 duration-bucket clipping (0.5114); A3 quantile alignment to label distribution (0.5443, as theory predicts); A4 Huber / BayesianRidge / LightGBM(depth 2) stackers (0.5151 / 0.5096 / 0.5115); A5 KNN on WavLM embeddings (0.675 alone; stack 0.5098, speaker-grouped 0.5240 vs 0.5215).
- **Group B features** (src/round2_b.py): YIN pitch (semitones) + RMS energy contours, short pauses > 0.25 s, tempo CV over 5 s windows, MTLD, mean dependency distance, fragment share. Correlations up to 0.35 (MTLD), 0.22 (short pauses/min), 0.21 (pitch-jump std).
- As features inside the joint SVR / text Ridge: +0.0000 / +0.0016; as stacker meta-features: worse. **As their own Ridge component (trained w/o 0.0 clips): +0.0110 short, 8/8 wins**, despite only 0.869 alone.
- **Leak checks (src/round2_check.py):** gain holds under prompt-held-out CV (+0.0105, 8/8) and speaker-grouped CV (+0.0156, 8/8). Mostly from prosody (pitch/energy), sub-groups add up.
- **Drift:** energy/pause features shift by 0.3-0.4 SD on test (quieter, fewer pauses); 8 test clips had a feature beyond 4 SD. Fix: winsorise every feature to the TRAIN 1st-99th percentiles (67 of 2808 test values clipped). Gains: random +0.0082, prompt-held-out +0.0081, speaker-grouped +0.0135, all 8/8.
- **Candidate inal_stack_pw.csv** = final_stack_w components + ridge_r2w_dz: nested 0.4611 / short 0.4973 (final_stack_w 0.4637 / 0.5052), train RMSE 0.138; test preds differ by 0.059 RMS.
- New baseline (final_stack_pw components) over the 8 splits: short 0.5004 +/- 0.0026.
- **C1 prompt-frequency stack features** (log count of training clips in the prompt cluster, rare flag; label-free): 0.5033, 0/8 -> reject.
- **C2 KNN on text features** (DeBERTa + hand + CoLA, out-of-fold): 0.666 alone; random-split stack 0.5022 (1/8) but prompt-held-out +0.0045 (8/8) -> mixed, reject under the rule.
- **C mid-fusion MLP** (PCA'd WavLM + DeBERTa + hand/round-2 features -> 32 -> 8 -> 1, dropout, 3 seeds): 0.639 alone; stack 0.5005 (3/8) -> reject.
- python src/round2_b.py --component rebuilds ridge_r2w_dz exactly.
- **B2 Whisper decoder entropy** (src/whisper_entropy.py; HF whisper-large-v3 greedy decode per 30 s window, per-token entropy): ent_mean r=-0.515, top1_mean r=+0.531 (as strong as Whisper word confidence, r=0.52). E1 own component: random 0.5009 (2/8), prompt-held-out -0.0019, speaker -0.0015 -> reject (redundant with ASR-confidence features + Whisper embeddings). E2 merged into the prosody Ridge: -0.0072 / -0.0054 / -0.0090 -> reject (it makes the prosody component less complementary).
- **Round-2 outcome:** one component kept (ridge_r2w_dz) -> final_stack_pw is the new primary candidate.
