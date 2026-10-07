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
| Stack + Whisper layers (`final_stack_w`) | + SVR on Whisper-large-v3 encoder layers 29–31 | 0.3432 |
| Alternative (`final_stack_pw`) | + Ridge on prosody, timing, lexical and syntax features | 0.3493 |
| **Final** (`final_stack_wj`) | **+ Ridge on a local LLM judge's grammar ratings (Qwen3.5-9B via Ollama)** | **0.3362** |
| Second selection (`final_stack_wj2`) | judge anchored with 8 human-scored training answers | 0.3387 |

The two final selections on Kaggle:

| On the 732 non-zero training clips | `final_stack_wj` (best public score) | `final_stack_wj2` (best CV) |
|---|---|---|
| **Training (in-sample) RMSE** | 0.149 | 0.150 |
| **Nested cross-validated RMSE** | 0.463 | 0.464 |
| Judge's correlation with human score | 0.572 (zero-shot) | 0.592 (anchored) |
| Paired CV vs the other (24 comparisons) | — | wins 24/24 (+0.002) |
| Public leaderboard | **0.3362** | 0.3387 |

The LLM judge was neutral in random-fold CV but improved **prompt-held-out CV**, and the public
leaderboard confirmed it (0.3432 → 0.3362). A zero-shot judge rates grammar without having seen
the prompt, which matters because about half the test answers prompts that are rare or absent in
training. The anchored judge is better by every CV scheme but 0.0025 lower on the ~130 public
clips (within noise), so both are selected for the private ranking.

The training/CV gap is expected: SVRs on thousands of embedding dimensions nearly memorise
769 clips. Every model choice was made on cross-validation, never on training error.

The **primary metric** is RMSE on non-zero clips. The 37 training clips labelled 0.0 are a
separate recording batch that the test set doesn't contain (notebook, section 2). Model choices
were made on **short clips (< 50 s)**, which resemble the test set, with **prompt-held-out CV**
as the guide for prompt-independent signals such as the LLM judge. Full history, including
everything that didn't work:
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Approach

1. **Transcription:** faster-whisper large-v3 with a disfluent prompt (to keep fillers and
   errors) and word timestamps. Clips that fell into repetition loops are re-decoded with
   Whisper's loop guard on.
2. **Interpretable features:** LanguageTool error rates (punctuation and casing rules
   excluded), fluency and pauses, disfluencies, syntactic complexity (spaCy), Whisper
   confidence, and CoLA grammatical-acceptability scores.
3. **Seven components**, each cross-validated with the same 5 folds:

   | Component | Input | Model | CV RMSE |
   |---|---|---|---|
   | `svr_all` | all blocks below + hand features | SVR | 0.491 |
   | `svr_wavlm_large` | WavLM-large layers 18–23, mean+std | SVR | 0.517 |
   | `svr_whisperL29_31` | Whisper-large-v3 encoder layers 29–31, mean+std | SVR | 0.527 |
   | `attnpool_wavlmL` | attention pooling over WavLM frames | small neural net | 0.563 |
   | `ridge_deb_hand_dz` | DeBERTa-v3 embedding + hand features (no 0.0 clips) | Ridge | 0.602 |
   | `ft_roberta` | fine-tuned RoBERTa-CoLA, 3 seeds (no 0.0 clips) | transformer | 0.684 |
   | `ridge_llm9b_dz` | local LLM judge (Qwen3.5-9B, Ollama, temperature 0, seed 42): rubric score, errors per 100 words by type, complex structures, confidence; clipped to the training range (no 0.0 clips) | Ridge | 0.783 |

   The judge's score alone correlates 0.57 with the human grade. Everything runs locally:
   transcripts never leave the machine. The alternative `final_stack_pw` instead adds
   `ridge_r2w_dz` (pitch, energy, pauses, tempo, MTLD, dependency distance, fragments).

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
- **Diversity beats strength:** RoBERTa-large was better alone but made the stack worse.
- **Choose the validation scheme by the signal:** the LLM judge looked neutral in random-fold CV
  but helped on prompt-held-out CV, and the leaderboard agreed (0.3432 → 0.3362).
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
# fallback submission (public LB 0.3432):
python src/stack.py svr_all,svr_wavlm_large,ridge_deb_hand_dz,ft_roberta,attnpool_wavlmL,svr_whisperL29_31 --out final_stack_w
# final submission (public LB 0.3362): local LLM judge via Ollama (https://ollama.com)
ollama pull qwen3.5:9b                               # once; ~6.6 GB, runs on an 8 GB GPU
python src/llm_judge_ollama.py qwen3.5:9b            # ~100 min, cached + resumable
python src/round2_f.py                               # judge component (+ paired CV report)
python src/stack.py svr_all,svr_wavlm_large,ridge_deb_hand_dz,ft_roberta,attnpool_wavlmL,svr_whisperL29_31,ridge_llm9b_dz --out final_stack_wj
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
src/llm_judge_ollama.py      local LLM grammar judge (Qwen3.5-9B via Ollama)
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
