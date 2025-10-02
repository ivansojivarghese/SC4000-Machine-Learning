#!/bin/bash
# Step 5: Distill teacher logits into a student per fold, sequentially (train -> calibrate -> infer).
# Usage:
#   RUN=my_run sbatch step5_distill_student.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=1440
#SBATCH --cpus-per-task=8
#SBATCH --job-name=S5_Distill
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

RUN=${RUN:-$(date +%Y%m%d_%H%M%S)}

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${PWD}/.hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for FOLD in 0 1 2 3 4; do
  echo "[Step5] RUN=${RUN} FOLD=${FOLD}"
  TEACHERS="model_save/llama3-70b_fold_${FOLD}/teacher_logits_fold_${FOLD}.npy,model_save/qwen2-72b_fold_${FOLD}/teacher_logits_fold_${FOLD}.npy"
  MODEL_DIR="model_save/${RUN}/student_distilbert_fold_${FOLD}"
  SUB_FILE="sub/${RUN}_student_submission_fold_${FOLD}.csv"

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

  python ./main.py --mode student-calibrate --student-output-model-dir "${MODEL_DIR}"
  python ./main.py --mode student-infer --student-output-model-dir "${MODEL_DIR}" --student-submission-path "${SUB_FILE}"
done

echo "[Step5] Done"
