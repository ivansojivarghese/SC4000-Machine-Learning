#!/bin/bash
# Step 5: Distill a 3-class student from teacher probabilities (LLaMA-only) per fold.
# This script now uses student_train_distill_hf.py directly and aligns via OOF parquet from step4.
# Usage:
#   sbatch --array=0-4 step5_distill_student.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --cpus-per-task=8
#SBATCH --job-name=S5_Distill
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

FOLD=${FOLDS:-${SLURM_ARRAY_TASK_ID:-0}}

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${PWD}/.hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


echo "[Step5] Distilling fold ${FOLD} using LLaMA-only OOF probs"

OOF_PATH=${INFER_OOF_TABLE:-model_save/teacher_logits/oof_probs.parquet}
FOLD_TRAIN_CSV=data/fold_data/fold_${FOLD}_train.csv
# RESUME_CHECKPOINT=model_save/distilled_gemma2-9b_fold_${FOLD}/checkpoint-1000
STUDENT_MODEL=${STUDENT_MODEL_NAME:-google/gemma-2-9b-it}
OUTDIR=${STUDENT_OUTDIR:-model_save/distilled_gemma2-9b_fold_${FOLD}}
LR=${STUDENT_LR:-5e-5}
EPOCHS=${STUDENT_EPOCHS:-2}
BATCH=${STUDENT_PER_DEVICE_BS:-8}
ACCUM=${STUDENT_GRAD_ACCUM:-4}
T_SOFT=${STUDENT_T_SOFT:-2.0}
ALPHA=${STUDENT_ALPHA:-0.7}
MSE_W=${STUDENT_MSE_WEIGHT:-0.05}
LABEL_SMOOTH=${STUDENT_LABEL_SMOOTH:-0.05}
MAXLEN=${STUDENT_MAXLEN:-384}
# Leave STUDENT_MAX_STEPS unset to run purely by epochs; set to a positive int to cap by steps
MAX_STEPS=${STUDENT_MAX_STEPS:-2000}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
OVERWRITE=${STUDENT_OVERWRITE:-0}
# Early stopping patience (in evaluation events). With evaluation_strategy=epoch, this is epochs without improvement.
ES_PATIENCE=${STUDENT_ES_PATIENCE:-1}

# Evaluation cadence (steps-based by default for live logs). Override via env if you prefer epoch-based.
EVAL_STRATEGY=${STUDENT_EVAL_STRATEGY:-steps}   # steps | epoch | no
EVAL_STEPS=${STUDENT_EVAL_STEPS:-200}           # only used when steps strategy
# Saving cadence (align saves with evals by default when using steps)
if [ "$EVAL_STRATEGY" = "steps" ]; then
  SAVE_STRATEGY=${STUDENT_SAVE_STRATEGY:-steps}
  SAVE_STEPS=${STUDENT_SAVE_STEPS:-$EVAL_STEPS}
else
  SAVE_STRATEGY=${STUDENT_SAVE_STRATEGY:-epoch}
  SAVE_STEPS=${STUDENT_SAVE_STEPS:-}
fi
# Logging frequency (print more often during steps-based evals)
if [ "$EVAL_STRATEGY" = "steps" ]; then
  LOGGING_STEPS=$(( EVAL_STEPS/2 ))
  if [ "$LOGGING_STEPS" -lt 10 ]; then LOGGING_STEPS=10; fi
else
  LOGGING_STEPS=${STUDENT_LOGGING_STEPS:-50}
fi

# Calibration file (update path if needed)
CALIBRATION_FILE="calibration/vector_scaling_params.json"

# Optional fresh start: wipe OUTDIR when OVERWRITE=1 and not resuming
if [ "$OVERWRITE" = "1" ] && [ -d "$OUTDIR" ] && [ -z "${RESUME_CHECKPOINT}" ]; then
  echo "[Step5] OVERWRITE=1 -> removing existing OUTDIR: $OUTDIR"
  rm -rf "$OUTDIR"
fi

if [ ! -f "$FOLD_TRAIN_CSV" ]; then
  echo "[Step5][Error] Missing fold train CSV: $FOLD_TRAIN_CSV. Run step3 first." >&2
  exit 3
fi
if [ ! -f "$OOF_PATH" ]; then
  echo "[Step5][Error] Missing OOF table produced by step4: $OOF_PATH" >&2
  exit 4
fi

# Informative log: epoch-driven vs step-capped
if [ -z "$MAX_STEPS" ]; then
  echo "[Step5] Epoch-driven training: num_epochs=$EPOCHS (no --max_steps passed)"
else
  echo "[Step5] Step-capped training: --max_steps=$MAX_STEPS will limit total steps (overrides epochs)"
fi

echo "[Step5] Eval strategy=$EVAL_STRATEGY ${EVAL_STRATEGY=steps:+(eval_steps=$EVAL_STEPS)} | Save strategy=$SAVE_STRATEGY ${SAVE_STEPS:+(save_steps=$SAVE_STEPS)} | logging_steps=$LOGGING_STEPS"

python -u student_train_distill_hf.py \
  --fold_train_csv "$FOLD_TRAIN_CSV" \
  --teacher_oof_table "$OOF_PATH" \
  --teacher_model_name llama \
  --output_dir "$OUTDIR" \
  --model_name "$STUDENT_MODEL" \
  --learning_rate $LR \
  --num_epochs $EPOCHS \
  ${MAX_STEPS:+--max_steps $MAX_STEPS} \
  --per_device_train_batch_size $BATCH \
  --per_device_eval_batch_size $BATCH \
  --gradient_accumulation_steps $ACCUM \
  --T_soft $T_SOFT \
  --alpha $ALPHA \
  --mse_weight $MSE_W \
  --label_smoothing $LABEL_SMOOTH \
  --max_length $MAXLEN \
  --early_stopping_patience $ES_PATIENCE \
  --fp16 --gradient_checkpointing --use_fast_tokenizer \
  --load_in_4bit --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --dataloader_num_workers 4 \
  --num_folds 3 --fold_idx $FOLD \
  --evaluation_strategy "$EVAL_STRATEGY" \
  ${EVAL_STEPS:+--eval_steps $EVAL_STEPS} \
  --save_strategy "$SAVE_STRATEGY" \
  ${SAVE_STEPS:+--save_steps $SAVE_STEPS} \
  --save_total_limit 1 \
  --logging_steps $LOGGING_STEPS \
  # --calibration "$CALIBRATION_FILE" \
  ${RESUME_CHECKPOINT:+--resume_from_checkpoint "$RESUME_CHECKPOINT"}

# Export a standalone classifier head for this fold (useful for inference scripts)
HEAD_PT="$OUTDIR/classifier_head.pt"
if [ ! -f "$HEAD_PT" ]; then
  echo "[Step5] Exporting classifier head to $HEAD_PT"
  if ! python export_classifier_head.py --model-dir "$OUTDIR" --out "$HEAD_PT"; then
    echo "[Step5][Warn] Failed to export classifier head for fold ${FOLD}. Continuing without standalone head."
  fi
fi

echo "[Step5] Done fold $FOLD -> $OUTDIR"
