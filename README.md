# LMSYS Chatbot Arena First Place Solution (Mock + Student Baseline)

This workspace contains:
- A lightweight, runnable mock reproduction of the first-place pipeline phases (post-pretrain, CV, distillation, LoRA merge, GPTQ, TTA).
- A real small-model student training path using Hugging Face (DistilBERT/BERT) that generates a valid probabilities submission and supports calibration.

What you can do locally:
- Run mock train/inference to validate structure and output a submission (`./sub/final_submission.csv`).
- Train a real student model (DistilBERT or BERT), calibrate probabilities with temperature scaling, evaluate on a dedicated holdout, and produce a calibrated submission (`./sub/student_submission.csv`).

## Modes
- google/gemma-2-9b [https://huggingface.co/google/gemma-2-9b]
- Qwen/Qwen2.5-72B [https://huggingface.co/Qwen/Qwen2.5-72B]
- meta-llama/Llama-3.1-70B [https://huggingface.co/meta-llama/Llama-3.1-70B]

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

