# LMSYS Chatbot Arena – Minimal Pipeline Guide

This repo contains a lean end-to-end pipeline to produce a valid submission for the LMSYS Chatbot Arena competition using a distilled student and optional LoRA/GPTQ steps. It’s tailored for a Slurm environment.

## Prerequisites
- Python (see `requirements.txt`)
- Slurm cluster with CUDA GPUs for training/quantization
- Data files present:
  - Train: `data/train.csv`
  - Test:  `data/test.csv`

Install (Windows PowerShell example for local prep):
```
pip install -r requirements.txt
```

## Data schema
- Train must include: `prompt, response_a, response_b`, plus either `winner` or probability columns `winner_model_a(_prob), winner_model_b(_prob), winner_tie(_prob)`.
- Test must include: `id` and text columns consistent with training (typically prompt/response pairs).

## What you run (Slurm)
The core workflow is Steps 2–9. Steps 1/3 are optional if you already have teacher outputs; defaults assume LLaMA-only OOF for student distillation.

### Step 2 — Create 5-fold splits
```
sbatch step2_make_folds.sh
```
- Output: `data/processed_data/folds_5_seed42.json`

### Step 4 — Infer teacher distributions (OOF)
Compute per-fold teacher probabilities and write a single OOF parquet used by the student.
```
# One fold
FOLD=0 sbatch step4_infer_fold.sh
# Or multi-fold driver
sbatch step4_infer_teacher_logits.sh
```
- Output: `model_save/teacher_logits/oof_probs.parquet`

### Step 5 — Distill student from OOF (array over 5 folds)
```
sbatch --array=0-4 step5_distill_student.sh
```
- Output per fold: `model_save/distilled_gemma2-9b_fold_<k>/`

### Step 6 — Average LoRA and merge
```
sbatch step6_lora_ensemble.sh
```
- Outputs:
  - `model_save/avg_lora/adapter_model.safetensors` (+ adapter_config)
  - `model_save/final_merged_model/`

### Step 7 — Quantize merged model (GPTQ 8-bit, with fallback)
```
sbatch step7_quantize_gptq.sh
```
- Output: `model_save/final_quantized_model/` (+ tokenizer). Falls back to BNB int8 for unsupported types.

### Step 8 — TTA inference (quantized CausalLM + head)
```
sbatch step8_infer_tta.sh
```
- Output: `sub/final_submission.csv`

### Step 9 — Ensemble to final
```
# Equal-mean ensemble (default)
RUN=my_run sbatch step9_eval.sh
# CV-weighted
RUN=my_run USE_WEIGHTED=1 sbatch step9_eval.sh
```
- Output: `sub/<RUN>_final_ensemble.csv`
- If no per-fold files found, the script falls back to `sub/final_submission.csv` or `sub/student_submission.csv`.

## Orchestration helpers (optional)
- `job_distill.sh`: single-fold pipeline (distill → calibrate → infer). Env: `FOLD=0..4`, optional `RUN`.
- `distill_all_folds.sh`: run all 5 folds and ensemble at the end.
- `job_distill_all_in_one.sh`: run all folds sequentially inside one job, then ensemble.

## Submission format
- Columns: `id, winner_model_a, winner_model_b, winner_tie`
- Each row must sum to 1. Final files live in `./sub/`.

## Notes
- If you re-run Step 5 into the same OUTDIR, it won’t auto-delete; use a new `STUDENT_OUTDIR` or add overwrite/resume flags.
- Quantization (Step 7) writes tokenizer files into the quantized dir and records fallback metadata if GPTQ is unsupported.

## End-to-end Slurm pipeline: Steps 1–9
These scripts live in the repo root and are designed for a Slurm cluster. They compose the full workflow from post-pretraining to final submission. Adjust SBATCH headers and paths to match your cluster.

### Step 1 — Post-pretrain large models (LoRA/QLoRA)
- Script: `step1_post_pretrain.sh`
- Purpose: Post-pretrain base CausalLMs (LLaMA/Qwen/Gemma) on UltraFeedback via LoRA/QLoRA, then merge adapters to produce a stronger base for later stages.
- Key env/inputs: 
  - `SCRATCH_BASE` for HF caches (auto-resolved; must be writable)
  - Base/tokenizer repos: `LLAMA_TOK`, `QWEN_TOK`, `GEMMA_TOK`, and `LLAMA_BASE`, `QWEN_BASE`, `GEMMA_BASE`
  - `AUTO_DOWNGRADE=1` auto-selects smaller bases if GPU memory is limited
- Outputs: Post-pretrained adapters and merged full model under `model_save/`
- Run:
```bash
sbatch step1_post_pretrain.sh
```

### Step 2 — Make 5-fold splits
- Script: `step2_make_folds.sh`
- Purpose: Create and persist 5-fold stratified splits for reuse by teachers and student.
- Key env/inputs:
  - Reads `data/train.csv`
  - `SCRATCH_BASE` and optional `FOLDS_OUT_DIR` for a scratch copy
- Outputs:
  - Local: `data/processed_data/folds_5_seed42.json`
  - Scratch: `${SCRATCH_BASE}/folds/folds_5_seed42.json`
- Run:
```bash
sbatch step2_make_folds.sh
```

### Step 3 — Train teachers per fold
- Script: `step3_train_teachers.sh`
- Purpose: Train teacher models per fold with 4/5 Kaggle train + external 33k augmentation; merge adapters after training.
- Key env/inputs:
  - `HF_TOKEN` (optional) for gated models
  - `FOLDS_JSON` (defaults to `data/processed_data/folds_5_seed42.json`)
  - Uses UltraFeedback CSVs in `data/`
- Outputs: Per-fold teacher artifacts under `model_save/`, ready for inference in Step 4.
- Run:
```bash
sbatch step3_train_teachers.sh
```

### Step 4 — Infer teacher distributions
- Scripts: `step4_infer_teacher_logits.sh` (multi-fold) and `step4_infer_fold.sh` (single fold wrapper)
- Purpose: For each fold teacher, compute log-likelihoods and produce a 3-way probability distribution over {A, B, Tie}. Optionally save last-token logits. Aggregate into an OOF table.
- Key env/inputs:
  - `INFER_FOLDS` (e.g., `all` or `0,1,2`)
  - `INFER_MODELS` to select teacher families
  - `INFER_SAVE_LASTTOK` (0/1), `INFER_FORCE_REGEN` (0/1)
- Outputs:
  - Per-fold files under `model_save/...`
  - OOF parquet used later by the student: `model_save/teacher_logits/oof_probs.parquet`
- Run:
```bash
# One fold
FOLD=0 sbatch step4_infer_fold.sh
# All (per Step 4 driver)
sbatch step4_infer_teacher_logits.sh
```

### Step 5 — Distill student from teacher OOF
- Script: `step5_distill_student.sh`
- Purpose: Distill Gemma-2 9B (sequence classifier) from LLaMA-only OOF probabilities with multi-loss KD: alpha*KL + (1-alpha)*CE + optional MSE on prob dists.
- Key env/inputs:
  - `INFER_OOF_TABLE` (default `model_save/teacher_logits/oof_probs.parquet`)
  - Fold CSV: `data/fold_data/fold_${FOLD}_train.csv`
  - Student cfg: `STUDENT_MODEL_NAME` (default `google/gemma-2-9b-it`), `STUDENT_LR`, `STUDENT_EPOCHS`, `STUDENT_MAX_STEPS`, `STUDENT_PER_DEVICE_BS`, `STUDENT_GRAD_ACCUM`, `STUDENT_MAXLEN`
  - KD cfg: `STUDENT_T_SOFT`, `STUDENT_ALPHA`, `STUDENT_MSE_WEIGHT`, `STUDENT_LABEL_SMOOTH`
- Outputs: `model_save/distilled_gemma2-9b_fold_${FOLD}` with checkpoints, tokenizer, and head.
- Notes: The script writes into `OUTDIR` but does not delete it; re-runs may append new checkpoints. Prefer a new `STUDENT_OUTDIR` or add overwrite/resume flags if desired.
- Run (array over all folds):
```bash
sbatch --array=0-4 step5_distill_student.sh
```

### Step 6 — LoRA averaging and merge
- Script: `step6_lora_ensemble.sh`
- Purpose: Directly average LoRA adapters from 5 student folds, ensuring `adapter_config.json` is present, then merge into a final base.
- Key env/inputs:
  - `BASE_MODEL` (default `google/gemma-2-9b-it`)
  - `FOLDS` (e.g., `0,1,2,3,4`)
  - `LORA_DIR_PREFIX` (e.g., `model_save/distilled_gemma2-9b_fold_`)
- Outputs:
  - Averaged adapter: `model_save/avg_lora/adapter_model.safetensors` (+ `adapter_config.json`)
  - Merged full model: `model_save/final_merged_model`
- Run:
```bash
sbatch step6_lora_ensemble.sh
```

### Step 7 — GPTQ quantization (8-bit)
- Script: `step7_quantize_gptq.sh`
- Purpose: Quantize the merged model to 8-bit GPTQ with calibration; save tokenizer alongside. Falls back to BitsAndBytes int8 for unsupported model types (e.g., Gemma2) with metadata for inference.
- Key env/inputs:
  - `MODEL_DIR` (default `model_save/final_merged_model`)
  - `OUT_DIR` (default `model_save/final_quantized_model`)
  - `CALIB_CSV` (default `data/train.csv`), `TOKENIZER_DIR` for tokenizer resolution
- Outputs: `OUT_DIR` with quantized weights, `quantization_config.json`, and tokenizer files. On fallback, writes BNB-int8 metadata and tokenizer.
- Run:
```bash
sbatch step7_quantize_gptq.sh
```

### Step 8 — TTA inference (quantized CausalLM + classifier head)
- Script: `step8_infer_tta.sh`
- Purpose: Run test-time augmentation (up to length 2000) using the quantized CausalLM and a lightweight classification head. Uses calibration temperature if present.
- Key env/inputs:
  - `HEAD_DIR` (student directory to export/derive head), `HEAD_OUT` (optional explicit head path)
  - `MODEL_DIR` (quantized dir), `TEST_CSV` (default `./data/test.csv`)
  - `CALIB_JSON` path (defaults under head dir)
- Outputs: `./sub/final_submission.csv`
- Run:
```bash
sbatch step8_infer_tta.sh
```

### Step 9 — Evaluate and ensemble to final
- Script: `step9_eval.sh`
- Purpose: Create final leaderboard submission by ensembling per-fold submissions or falling back to an existing single submission.
- Modes:
  - Equal-mean (default): averages `sub/${RUN}_student_submission_fold_*.csv` if present; else copies `sub/final_submission.csv` or `sub/student_submission.csv`.
  - CV-weighted: `USE_WEIGHTED=1` uses fold-wise log-loss metrics to weight files.
- Outputs: `sub/${RUN}_final_ensemble.csv`
- Run:
```bash
RUN=my_run sbatch step9_eval.sh               # equal-mean
RUN=my_run USE_WEIGHTED=1 sbatch step9_eval.sh  # CV-weighted
```

## Orchestration helpers
- `job.sh`: Quick baseline student training without KD/ensemble; good for sanity checks.
  - Run: `sbatch job.sh`

- `job_distill.sh`: Single fold pipeline (distill → calibrate → infer) with namespaced outputs.
  - Env: `FOLD` (0..4), `RUN` (tag)
  - Run: `FOLD=0 RUN=myrun sbatch job_distill.sh`

- `distill_all_folds.sh`: Submits all five folds sequentially, then ensembles.
  - Env: `RUN` tag, `USE_WEIGHTED=1` for CV-weighted
  - Run: `RUN=myrun sbatch distill_all_folds.sh`

- `job_distill_all_in_one.sh`: Runs all folds sequentially inside a single Slurm job (no nested sbatch), then ensembles.
  - Env: `RUN` tag, `USE_WEIGHTED=1` for CV-weighted
  - Run: `RUN=myrun sbatch job_distill_all_in_one.sh`

## Submission format
All submissions follow the competition’s 3-class probability format with row-wise sums equal to 1:
- Columns: `id,winner_model_a,winner_model_b,winner_tie`
- Final files are written under `./sub/`, e.g., `final_submission.csv` or `${RUN}_final_ensemble.csv`.

