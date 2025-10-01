#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --cpus-per-task=8
#SBATCH --job-name=DistillJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Usage:
#   sbatch job_distill.sh           # runs fold 0 by default
#   FOLD=3 sbatch job_distill.sh    # override to run fold 3

# Select which fold to run (0..4) and a run name to avoid overwrites
FOLD=${FOLD:-0}
# Provide RUN in environment to group outputs (e.g., RUN=myexp); defaults to timestamp
RUN=${RUN:-$(date +%Y%m%d_%H%M%S)}

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

# Throughput-oriented env tuning
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${PWD}/.hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[DistillJob] RUN=${RUN} FOLD=${FOLD}"

# Compose teacher logits list for this fold (llama3 + qwen2)
TEACHERS="model_save/llama3-70b_fold_${FOLD}/teacher_logits_fold_${FOLD}.npy,model_save/qwen2-72b_fold_${FOLD}/teacher_logits_fold_${FOLD}.npy"

# Per-fold model output directory and submission file (namespaced by RUN)
MODEL_DIR="model_save/${RUN}/student_distilbert_fold_${FOLD}"
SUB_FILE="sub/${RUN}_student_submission_fold_${FOLD}.csv"
echo "[DistillJob] MODEL_DIR=${MODEL_DIR}"
echo "[DistillJob] SUB_FILE=${SUB_FILE}"

# 1) Distill one fold: KL + CE (+ optional MSE), fast student config (<6h target)
python ./main.py --mode student-distill-train \
  --student-model microsoft/deberta-v3-base \
  --student-epochs 2 \
  --student-label-smoothing 0.05 \
  --student-early-stopping 1 \
  --student-max-length 256 \
  --student-train-batch-size 12 \
  --student-eval-batch-size 12 \
  --student-grad-accum 2 \
  --student-fp16 \
  --student-gradient-checkpointing \
  --student-num-workers 8 \
  --student-extra-csvs "data/ultrafeedback.csv,data/ultrafeedback_ties.csv,data/lmsys-33k-deduplicated.csv" \
  --student-dedup-by-prompt \
  --student-shuffle-ab \
  --student-output-model-dir "${MODEL_DIR}" \
  --distill-alpha 0.7 \
  --distill-temp 3.0 \
  --distill-mse-weight 0.1 \
  --cv-num-folds 5 \
  --cv-fold-idx ${FOLD} \
  --distill-teachers "${TEACHERS}"

# 2) Calibrate probabilities with temperature scaling
python ./main.py --mode student-calibrate --student-output-model-dir "${MODEL_DIR}"

# 3) Inference to produce submission CSV
python ./main.py --mode student-infer --student-output-model-dir "${MODEL_DIR}" --student-submission-path "${SUB_FILE}"

# After all folds are run (FOLD=0..4), ensemble with:
# python ensemble_submissions.py sub/${RUN}_final_ensemble.csv \
#   sub/${RUN}_student_submission_fold_0.csv sub/${RUN}_student_submission_fold_1.csv \
#   sub/${RUN}_student_submission_fold_2.csv sub/${RUN}_student_submission_fold_3.csv \
#   sub/${RUN}_student_submission_fold_4.csv
