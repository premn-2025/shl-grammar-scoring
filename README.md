# SHL Grammar Scoring Engine

Predicts a continuous grammar score (0–5) for 45–60 s spoken-English answers
(Kaggle: `shl-hiring-assessment-2026`). Final deliverable:
[`notebooks/SHL_Grammar_Scoring.ipynb`](notebooks/SHL_Grammar_Scoring.ipynb).

## Results

| | RMSE (non-zero clips) | Pearson | RMSE (all 769) |
|---|---|---|---|
| Constant mean baseline (public LB 1.005) | 1.027 | — | 1.239 |
| Best hand-feature model | 0.776 | 0.65 | 0.84 |
| Best text model (DeBERTa emb. + hand + CoLA, Ridge) | 0.602 | 0.80 | 0.84 |
| Best single audio model (WavLM-large L18–23, SVR) | 0.517 | 0.86 | 0.52 |
| **Final blend + stretch (5-fold OOF)** | **0.471** | **0.886** | **0.504** |
| Final model, **training (in-sample)** RMSE | 0.101 | 0.996 | 0.118 |

Non-zero clips are the primary metric: the 37 train clips labelled 0.0 are a separate
recording batch that the test set does not contain (see the notebook, section 2).
Full experiment history: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Approach

1. **Transcription:** faster-whisper large-v3 with a disfluent prompt (to keep fillers and
   errors), word timestamps, and re-decoding of clips that fell into repetition loops.
2. **Interpretable features:**
   - LanguageTool error rates (punctuation and casing rules excluded)
   - fluency and pauses
   - disfluencies
   - syntactic complexity (spaCy)
   - Whisper confidence
   - CoLA grammatical-acceptability scores
3. **Representations:**
   - WavLM-large (upper-middle layers, mean+std pooling)
   - Whisper-large-v3 encoder
   - DeBERTa-v3 transcript embedding
   - fine-tuned RoBERTa-CoLA regressor
4. **Models:** Ridge / SVR / LightGBM, all in the same 5-fold stratified CV (seed 42), with
   scaling and tuning inside the folds.
5. **Final:** a non-negative weighted blend fitted on OOF predictions, plus a linear
   "stretch" that counters shrinkage toward the mean, clipped to [0, 5].

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
- **Separate processes.** torch here is built for CUDA 11.8, so the Whisper scripts never
  import torch and run as their own processes, which avoids a DLL clash.
- **Symlink warning.** If Hugging Face warns about symlinks, set
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- **Flaky downloads.** `HF_HUB_DISABLE_XET=1` helped on an unstable connection.

**Data:** download it and put it in `data/` (gitignored, since competition data must not be
shared):
```
kaggle competitions download -c shl-hiring-assessment-2026
# unzip so that data/train.csv, data/test.csv, data/train/*.wav, data/test/*.wav exist
```

## How to run

Each step caches its output in `cache/`, so steps can be re-run independently. Times are
for an RTX 4060.

```powershell
python src/explore_and_baseline.py                 # data summary + mean baseline
python src/transcribe.py                           # ~50 min, resumable
python src/transcribe.py --flag-loops              # find repetition-loop transcripts
python src/transcribe.py --redo-loops              # re-decode them with Whisper's guard on
python src/features.py                             # LanguageTool + spaCy features
python src/embeddings.py wavlm_large               # ~8 min
python src/whisper_enc.py                          # ~8 min
python src/embeddings.py deberta; python src/embeddings.py cola
foreach ($s in 0,1,2) { python src/finetune_text.py textattack/roberta-base-CoLA 4 $s }
$env:FT_FULL="1"; foreach ($s in 0,1,2) { python src/finetune_text.py textattack/roberta-base-CoLA 4 $s }; $env:FT_FULL=""
python src/final.py                                # blend + stretch -> submissions/final.csv
jupyter nbconvert --execute --to notebook --inplace notebooks/SHL_Grammar_Scoring.ipynb
```

Optional exploration scripts:
- `python src/train.py hand` runs the hand-feature models.
- `python src/train.py emb` runs the embedding models.
- `python src/train.py wavlm_scan wavlm_large` scores each WavLM layer.
- `python src/train.py audio wavlm_large 18 23` compares pooling choices.
- `python src/ensemble.py <models...>` blends any saved predictions.

## Repository layout

```
src/explore_and_baseline.py  data checks, constant baseline
src/transcribe.py            Whisper transcription (+ loop re-decoding)
src/features.py              interpretable features, correlation report
src/embeddings.py            WavLM / DeBERTa / mpnet / CoLA embeddings
src/whisper_enc.py           Whisper encoder embeddings
src/finetune_text.py         fine-tuned text regressor (5-fold, seeds, full fit)
src/train.py                 shared CV + Ridge/SVR/LightGBM experiments
src/ensemble.py              OOF-weighted blending
src/final.py                 final model, training RMSE, submission
notebooks/                   final notebook
plots/                       figures used in the notebook
```

## Known issues on this setup

- **LightGBM import order:** LightGBM crashes (access violation) if scikit-learn is
  imported first, so `train.py` imports `lightgbm` first.
- **DeBERTa-v3 fine-tuning:** it diverged with torch 2.7 cu118 (NaN, then collapse), so the
  fine-tuned text model is RoBERTa-CoLA instead.
- **Stale sample submission:** `sample_submission.csv` doesn't match `test.csv`, so
  submissions use `test.csv`'s 216 rows (accepted by Kaggle).
