# SHL Grammar Scoring Engine

Predicts a continuous grammar score (0–5) for 6–61 s spoken-English answers
(Kaggle: `shl-hiring-assessment-2026`). Final deliverable:
[`notebooks/SHL_Grammar_Scoring.ipynb`](notebooks/SHL_Grammar_Scoring.ipynb).

## Results

| Submission | What changed | Public LB RMSE |
|---|---|---|
| Constant mean | baseline | 1.0050 |
| v1 | weighted blend of 4 models + linear stretch | 0.3529 |
| v2 | + attention pooling over WavLM frames | 0.3507 |
| Stack | duration-aware Ridge stack | 0.3440 |
| **Final** (`final_stack_w`) | **+ SVR on Whisper-large-v3 encoder layers 29–31** | **0.3432** |
| Alternative (`final_stack_pw`) | + Ridge on prosody, timing, lexical and syntax features | lower than 0.3432 |

Both final files are selected on Kaggle for the private ranking:

| On the 732 non-zero training clips | `final_stack_w` (final) | `final_stack_pw` (alternative) |
|---|---|---|
| **Training (in-sample) RMSE** | 0.137 | 0.138 |
| **Nested cross-validated RMSE** | 0.464 | 0.461 |
| Nested CV, short clips (< 50 s) | 0.505 | 0.497 |
| Public leaderboard | **0.3432** | lower |

`final_stack_w` has the best public score; `final_stack_pw` has the best cross-validation (its
gain held under random, prompt-held-out and unseen-speaker CV). The public leaderboard has only
~130 clips (about ±0.02 of noise), so the two can swap order on the hidden 40%, which is why
both are selected.

The training/CV gap is expected: SVRs on thousands of embedding dimensions nearly memorise
769 clips. Every model choice was made on cross-validation, never on training error.

The **primary metric** is RMSE on non-zero clips. The 37 training clips labelled 0.0 are a
separate recording batch that the test set doesn't contain (notebook, section 2). Model choices
were made on **short clips (< 50 s)**, which resemble the test set; that score predicted the
leaderboard order of every submission. Full history, including everything that didn't work:
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Approach

1. **Transcription:** faster-whisper large-v3 with a disfluent prompt (to keep fillers and
   errors) and word timestamps. Clips that fell into repetition loops are re-decoded with
   Whisper's loop guard on.
2. **Interpretable features:** LanguageTool error rates (punctuation and casing rules
   excluded), fluency and pauses, disfluencies, syntactic complexity (spaCy), Whisper
   confidence, and CoLA grammatical-acceptability scores.
3. **Six components** (seven in the alternative), each cross-validated with the same 5 folds:

   | Component | Input | Model | CV RMSE |
   |---|---|---|---|
   | `svr_all` | all blocks below + hand features | SVR | 0.491 |
   | `svr_wavlm_large` | WavLM-large layers 18–23, mean+std | SVR | 0.517 |
   | `svr_whisperL29_31` | Whisper-large-v3 encoder layers 29–31, mean+std | SVR | 0.527 |
   | `attnpool_wavlmL` | attention pooling over WavLM frames | small neural net | 0.563 |
   | `ridge_deb_hand_dz` | DeBERTa-v3 embedding + hand features (no 0.0 clips) | Ridge | 0.602 |
   | `ft_roberta` | fine-tuned RoBERTa-CoLA, 3 seeds (no 0.0 clips) | transformer | 0.684 |
   | `ridge_r2w_dz` (alternative only) | pitch, energy, short pauses, tempo variation, MTLD, dependency distance, fragments; each clipped to its training 1st–99th percentile (no 0.0 clips) | Ridge | 0.867 |

   The last component is weak alone, but its information differs from the others, so it adds
   the most to the stack in cross-validation.

4. **Duration-aware Ridge stack:** the out-of-fold predictions, plus duration (clamped to
   20–61 s), speech rate and ASR confidence, plus prediction × duration interactions. On short
   answers it trusts the fine-tuned text model more and the pooled audio SVRs less.

## Key findings

- **0.0 batch:** 37 clips (IDs 5037–5073) with an off-rubric label, absent from test.
- **Short clips:** 40–50 s answers score lower (mean 3.06 vs 3.58), and most test clips are 45 s.
- **Prompt shift:** about half the test answers prompts that are rare or absent in training.
  Frozen text embeddings partly learn topic; the fine-tuned grammar model doesn't.
- **Repeated speakers:** they occur in training but never in test. This inflated audio models'
  CV by 0.03–0.04, but the stack's weights stay best on unseen speakers.
- **Layer choice:** WavLM's best layers are upper-middle; Whisper's are near the top.
- **Diversity beats strength:** RoBERTa-large was better alone but made the stack worse, while
  the weak (0.87) prosody/timing component gave the largest late gain.
- **Many ideas were tested and rejected** under a strict paired rule (same 8 splits, ≥ 6/8
  wins, gain above noise): HuBERT, bagging, pseudo-labels, Whisper LoRA, post-processing,
  alternative stackers, KNN features, mid-fusion MLP, decoder entropy. See `EXPERIMENTS.md`.

## Setup (Windows 11, RTX 4060 8 GB, Python 3.11)

```powershell
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m spacy download en_core_web_sm
java -version   # LanguageTool needs Java >= 17 (e.g. Temurin 17)
```

**faster-whisper on Windows:**
- **CUDA 12 libraries.** CTranslate2 needs CUDA 12 cuBLAS and cuDNN 9. The
  `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels provide them, and `src/transcribe.py`
  adds their `site-packages/nvidia/*/bin` folders to the DLL search path, so you don't need
  to install CUDA system-wide.
- **Separate processes.** torch here is built for CUDA 11.8, so the CTranslate2 Whisper
  scripts never import torch and run as their own processes, which avoids a DLL clash.
- **Downloads.** Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence the symlink warning. On an
  unstable connection, try toggling `HF_HUB_DISABLE_XET=1`; different files downloaded with
  different settings.

**Data:** download it into `data/` (gitignored, since competition data must not be shared):
```
kaggle competitions download -c shl-hiring-assessment-2026
# unzip so that data/train.csv, data/test.csv, data/train/*.wav, data/test/*.wav exist
```

## How to reproduce the final submission

Each step caches its output in `cache/`. Times are for an RTX 4060.

```powershell
python src/explore_and_baseline.py                   # data checks + mean baseline
python src/transcribe.py                             # ~50 min, resumable
python src/transcribe.py --flag-loops; python src/transcribe.py --redo-loops
python src/features.py                               # LanguageTool + spaCy features
python src/embeddings.py wavlm_large                 # ~8 min
python src/whisper_enc.py                            # ~8 min (last encoder layer)
python src/whisper_layers.py                         # ~5 min (all encoder layers)
python src/embeddings.py deberta; python src/embeddings.py cola
python src/frames.py                                 # frame cache for attention pooling
python src/attnpool.py 0
foreach ($s in 0,1,2) { python src/finetune_text.py textattack/roberta-base-CoLA 4 $s }
$env:FT_FULL="1"; foreach ($s in 0,1,2) { python src/finetune_text.py textattack/roberta-base-CoLA 4 $s }; $env:FT_FULL=""
python src/final.py                                  # SVR/Ridge components + full-fit predictions
python src/train.py wavlm_scan whisper_layers        # layer scan (choose layers 29-31)
python src/train.py whisper_upper                    # SVR on Whisper layers 29-31
# final submission (public LB 0.3432):
python src/stack.py svr_all,svr_wavlm_large,ridge_deb_hand_dz,ft_roberta,attnpool_wavlmL,svr_whisperL29_31 --out final_stack_w
# alternative submission (best CV): prosody/timing component + 7-component stack
python src/round2_b.py --component
python src/stack.py svr_all,svr_wavlm_large,ridge_deb_hand_dz,ft_roberta,attnpool_wavlmL,svr_whisperL29_31,ridge_r2w_dz --out final_stack_pw
python src/make_notebook.py
jupyter nbconvert --execute --to notebook --inplace notebooks/SHL_Grammar_Scoring.ipynb
```

**Evaluation modes:**
- `CV_MODE=group` uses prompt-held-out folds; `CV_MODE=speaker` uses speaker-grouped folds.
  Their predictions go to separate cache folders.
- `src/gating.py` holds the paired 8-split short-clip evaluation used to accept or reject changes.

## Repository layout

```
src/explore_and_baseline.py  data checks, constant baseline
src/transcribe.py            Whisper transcription, loop re-decoding, clean pass
src/features.py              interpretable features, correlation report
src/embeddings.py            WavLM / HuBERT / DeBERTa / mpnet / CoLA embeddings
src/whisper_enc.py           Whisper encoder (last layer, CTranslate2)
src/whisper_layers.py        Whisper encoder, all layers (transformers)
src/frames.py                frame-level WavLM cache
src/attnpool.py              attention-pooling model
src/finetune_text.py         fine-tuned text regressor (folds, seeds, full fit)
src/finetune_audio.py        top-layer WavLM fine-tuning (experiment)
src/train.py                 shared CV (random / prompt / speaker folds), Ridge/SVR/LightGBM
src/final.py                 component matrices + blend (v1-v3)
src/stack.py                 duration-aware Ridge stack (final model)
src/gating.py                paired evaluation + gating experiments
src/ensemble.py, bagging.py, pseudo.py, tune_svr.py, crop_aug.py, lt_rules.py,
whisper_diff.py, llm_judge.py, overnight.py, morning_report.py   experiments (see EXPERIMENTS.md)
src/make_notebook.py         builds the final notebook
notebooks/                   final notebook (transcript outputs cleared: competition data)
plots/                       figures used in the notebook
```

## Known issues on this setup

- **LightGBM import order:** LightGBM crashes (access violation) if scikit-learn is imported
  first, so `train.py` imports `lightgbm` first.
- **DeBERTa-v3 fine-tuning:** it diverged with torch 2.7 cu118 (NaN, then collapse), so the
  fine-tuned text model is RoBERTa-CoLA instead (AdamW eps=1e-6).
- **Stale sample submission:** `sample_submission.csv` doesn't match `test.csv`, so
  submissions use `test.csv`'s 216 rows (accepted by Kaggle).
