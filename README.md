# LMSYS Chatbot Arena First Place Solution (Mock + Student Baseline)

This workspace contains:
- A lightweight, runnable mock reproduction of the first-place pipeline phases (post-pretrain, CV, distillation, LoRA merge, GPTQ, TTA).
- A real small-model student training path using Hugging Face (DistilBERT/BERT) that generates a valid probabilities submission and supports calibration.

What you can do locally:
- Run mock train/inference to validate structure and output a submission (`./sub/final_submission.csv`).
- Train a real student model (DistilBERT or BERT), calibrate probabilities with temperature scaling, evaluate on a dedicated holdout, and produce a calibrated submission (`./sub/student_submission.csv`).

## Data layout
- Train CSV (one of):
  - `./data/train.csv`
  - `./data/lmsys-chatbot-arena/train.csv`
  - Columns required: `prompt,response_a,response_b` and either `winner` or probability columns: `winner_model_a(_prob), winner_model_b(_prob), winner_tie(_prob)`.
- Test CSV: `./data/test.csv` (already present).

## Quick start (Windows PowerShell)
1) Install dependencies
```
pip install -r requirements.txt
pip install torch transformers datasets accelerate
```

2) Mock pipeline (optional)
```
python main.py --mode train
python main.py --mode inference
```
Artifacts: `training_summary.json`, `./sub/final_submission.csv`.

## Real student training (HF)
Train DistilBERT (default):
```
python main.py --mode student-train --student-epochs 1 --student-max-samples 2000
```

Try BERT base with label smoothing and early stopping:
```
python main.py --mode student-train \
  --student-model bert-base-uncased \
  --student-epochs 3 \
  --student-max-samples 6000 \
  --student-label-smoothing 0.1 \
  --student-early-stopping 2
```

Trainer details:
- Label smoothing: configurable via `--student-label-smoothing` (default 0.05)
- Early stopping: `--student-early-stopping` patience (default 2), monitors eval_loss
- Best-checkpoint restore: `load_best_model_at_end=True` using Kaggle-style `log_loss`
- Checkpoints: `save_strategy='epoch'`, `save_total_limit=1`

Artifacts:
- Model + tokenizer: `./model_save/student_distilbert` (path reused for simplicity)
- Calibration: `./model_save/student_distilbert/calibration.json`
- Submission: `./sub/student_submission.csv`

## Probability calibration + holdout eval
Fit temperature on a calibration split and evaluate on a distinct holdout (no leakage):
```
python main.py --mode student-calibrate
python main.py --mode student-eval-holdout
```
The console logs and `calibration.json` include:
- temperature
- nll_cal_before / nll_cal_after / nll_cal_improvement
- nll_holdout_before / nll_holdout_after / nll_holdout_improvement
- logloss_cal_before / logloss_cal_after / logloss_cal_improvement (Kaggle-style)
- logloss_holdout_before / logloss_holdout_after / logloss_holdout_improvement (Kaggle-style)

Notes:
- In this small baseline, calibration may slightly help or hurt on holdout; expect better gains as the student improves (more epochs, stronger backbone, soft-label distillation).

## Calibrated inference
`student_infer_hf.py` automatically loads `calibration.json` (if present) and divides logits by T before softmax.
```
python main.py --mode student-infer
```
Outputs `./sub/student_submission.csv` with probability columns:
- `winner_model_a`
- `winner_model_b`
- `winner_tie`

## Tips for better Log Loss
- Train longer (5–8 epochs) with early stopping; we select best by Kaggle log loss.
- Increase `--student-max-samples` or use the full dataset.
- Use a stronger model (e.g., `bert-base-uncased`, `deberta-v3-base`).
- Add soft-label distillation with teacher logits (KL + CE).
- Keep calibration on a separate split; report holdout metrics.

## Knowledge Distillation (KL + CE)
Train the student on soft labels from one or more teacher models. Provide .npy files containing raw logits with shape [N, 3], aligned to rows of `train.csv` after minimal preprocessing (dropping NaNs in prompt/response fields). We automatically validate shapes and length before training.

Example (Windows PowerShell):
```
python main.py --mode student-distill-train \
  --student-model bert-base-uncased \
  --student-epochs 3 \
  --student-max-samples 12000 \
  --student-label-smoothing 0.05 \
  --student-early-stopping 2 \
  --student-train-batch-size 8 \
  --student-eval-batch-size 16 \
  --student-grad-accum 2 \
  --student-learning-rate 3e-5 \
  --student-warmup-ratio 0.06 \
  --student-gradient-checkpointing \
  --distill-alpha 0.7 \
  --distill-temp 3.0 \
  --distill-teachers "model_save/llama3-70b_fold_0/teacher_logits_fold_0.npy,model_save/qwen2-72b_fold_1/teacher_logits_fold_1.npy"
```

Notes:
- Teacher files should be raw logits, not probabilities. We'll apply softmax at temperature `--distill-temp` and average across teachers.
- Shapes must be [N, 3] and N must match the effective train rows (after dropping NaNs). The CLI validates this and will abort early with a helpful message if mismatched.
- For stronger students, consider `microsoft/deberta-v3-base` or `-large`. Enable `--student-bf16` or `--student-fp16` on suitable GPUs and `--student-gradient-checkpointing` for memory savings.

## Adding UltraFeedback datasets (external data)
Two CSVs are provided in `./data`: `ultrafeedback.csv` and `ultrafeedback_ties.csv`. They follow the LMSYS train schema (prompt/response_a/response_b plus either `winner` or `winner_*` probability columns).

You can include them in training (and distillation) via `--student-extra-csvs` as a comma-separated list. Optionally use `--student-shuffle-ab` to randomly swap A/B for a subset of the extra data to reduce position bias (labels are flipped automatically on swapped rows):

Example (baseline training with UltraFeedback merged):
```
python main.py --mode student-train \
  --student-model microsoft/deberta-v3-base \
  --student-epochs 3 \
  --student-max-samples 0 \
  --student-label-smoothing 0.1 \
  --student-early-stopping 2 \
  --student-extra-csvs "data/ultrafeedback.csv,data/ultrafeedback_ties.csv" \
  --student-shuffle-ab
```

Example (distillation with UltraFeedback merged):
```
python main.py --mode student-distill-train \
  --student-model microsoft/deberta-v3-large \
  --student-epochs 5 \
  --student-max-samples 0 \
  --student-label-smoothing 0.05 \
  --student-early-stopping 2 \
  --student-gradient-checkpointing \
  --student-extra-csvs "data/ultrafeedback.csv,data/ultrafeedback_ties.csv" \
  --student-shuffle-ab \
  --distill-alpha 0.7 \
  --distill-temp 3.0 \
  --distill-teachers "model_save/llama3-70b_fold_*/*.npy,model_save/qwen2-72b_fold_*/*.npy"
```

Implementation details:
- Base `train.csv` rows keep their original A/B order and carry an internal `base_idx` used to align teacher logits.
- Extra CSV rows are marked with `base_idx=-1` and don’t consume teacher logits. During distillation their KL term is masked (loss reduces to CE for those rows).
- A/B shuffling is applied only to extra rows to avoid mismatches between base rows and teacher logits.

## Throughput preview and stats-only mode
Before long runs, you can sanity-check dataset merging/deduping and estimate throughput without training by using a stats-only pass:

```
python main.py --mode student-train \
  --student-model microsoft/deberta-v3-large \
  --student-epochs 1 \
  --student-max-samples 0 \
  --student-max-length 320 \
  --student-extra-csvs "data/ultrafeedback.csv,data/ultrafeedback_ties.csv,data/lmsys-33k-deduplicated.csv" \
  --student-dedup-by-prompt \
  --student-shuffle-ab \
  --student-fp16 \
  --student-gradient-checkpointing \
  --student-train-batch-size 8 \
  --student-eval-batch-size 8 \
  --student-grad-accum 2 \
  --student-num-workers 8 \
  --student-stats-only
```

This prints merged/deduped/used counts and logs per-epoch throughput (examples/sec, tokens/sec) in a quick dry-run.

## Inference options: 8-bit and TTA
`student_infer_hf.py` supports:
- 8-bit loading (bitsandbytes) to reduce memory.
- Simple TTA by averaging probabilities across multiple `max_length` values.

Direct invocation example:

```
python student_infer_hf.py
```

Or via main (if flags are wired in your `main.py`):

```
python main.py --mode student-infer --student-infer-8bit --student-infer-tta-lengths "512,1024"
```

## Slurm pipeline (.sh): which to run and in what order
These scripts are designed for a Slurm cluster (Linux). Submit them from your cluster shell (not Windows PowerShell). Adjust SBATCH headers to match your environment.

- `job.sh`
  - Purpose: quick baseline student training (no distillation/ensemble). Good for sanity and throughput checks.
  - Run: `sbatch job.sh`

- `job_distill.sh`
  - Purpose: run a single fold end-to-end: distill-train → calibrate → infer.
  - Inputs: `FOLD` (0..4), optional `RUN` to namespace outputs.
  - Run: `sbatch job_distill.sh` (defaults to fold 0) or `FOLD=3 RUN=my_run sbatch job_distill.sh`.

- `distill_all_folds.sh` (recommended)
  - Purpose: orchestrates all 5 folds by submitting `job_distill.sh` for `FOLD=0..4` and then ensembles.
  - Control: `USE_WEIGHTED=1` enables CV-weighted ensemble; default is equal-mean.
  - Run: `sbatch distill_all_folds.sh` or `RUN=my_run USE_WEIGHTED=1 sbatch distill_all_folds.sh`.

Outputs:
- Per fold: `model_save/${RUN}/student_distilbert_fold_${FOLD}` and `sub/${RUN}_student_submission_fold_${FOLD}.csv`
- Final ensemble: `sub/${RUN}_final_ensemble.csv`

If your cluster doesn’t permit nested `sbatch` from within a job, submit `job_distill.sh` manually for each fold and then run the ensembling Python script locally.

## Calibration-aware GPTQ quantization (CausalLM)
Use a held-out calibration set to guide GPTQ quantization for better accuracy.

```
python quantize_gptq_calibrated.py \
  --model-dir model_save/final_merged_model \
  --out-dir model_save/final_quantized_model \
  --calib-csv ./data/train.csv \
  --bits 4 \
  --group-size 128 \
  --max-calib-samples 1024 \
  --max-length 512
```

Notes:
- The script builds calibration examples from the A/B text format used by the student.
- `auto-gptq` must be installed in the environment and is typically best supported on Linux/CUDA.

## GPTQ inference with a classification head adapter
Run inference on a GPTQ-quantized CausalLM using a lightweight classifier head (3-way) and optional temperature/TTA.

1) Export a classifier head from your trained seq-classifier directory:

```
python export_classifier_head.py \
  --model-dir ./model_save/student_distilbert \
  --out ./model_save/student_distilbert/classifier_head.pt
```

2) Inference (Option A: provide head explicitly and temperature path):

```
python student_gptq_infer.py \
  --model-dir ./model_save/final_quantized_model \
  --test-csv ./data/test.csv \
  --out ./sub/student_submission.csv \
  --head-path ./model_save/student_distilbert/classifier_head.pt \
  --tta-lengths 512,1024 \
  --batch-size 8 \
  --max-length 1024 \
  --temperature-json ./model_save/student_distilbert/calibration.json
```

Option B: derive head (and temperature) directly from a seq-classifier directory:

```
python student_gptq_infer.py \
  --model-dir ./model_save/final_quantized_model \
  --test-csv ./data/test.csv \
  --out ./sub/student_submission.csv \
  --classifier-from-dir ./model_save/student_distilbert \
  --tta-lengths 512,1024 \
  --batch-size 8 \
  --max-length 1024
```

## Optional: LoRA/QLoRA post-pretrain and merge
You can nudge a base CausalLM toward your domain with LoRA/QLoRA, then merge adapters to a full model for KD or quantization.

```
# Train LoRA/QLoRA adapters
python lora_train.py \
  --base-model model_save/post_pretrain_llama3-70b \
  --out-dir model_save/post_pretrain_llama3-70b_lora \
  --bf16

# Merge adapters
python lora_merge.py \
  --base-model model_save/post_pretrain_llama3-70b \
  --lora-dir model_save/post_pretrain_llama3-70b_lora \
  --out-dir model_save/final_merged_model
```

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

