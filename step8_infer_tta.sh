#!/bin/bash
# Step 8: Apply TTA during inference (GPTQ 8-bit CausalLM + classification head)
# Usage:
#   sbatch step8_infer_tta.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --time=120
#SBATCH --cpus-per-task=4
#SBATCH --job-name=S8_TTA
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

HEAD_DIR=${HEAD_DIR:-model_save/distilled_gemma2-9b_fold_0}
HEAD_OUT=${HEAD_OUT:-${HEAD_DIR}/classifier_head.pt}
if [ ! -f "$HEAD_OUT" ]; then
  python export_classifier_head.py \
    --model-dir "$HEAD_DIR" \
    --out "$HEAD_OUT" || true
fi

MODEL_DIR=${MODEL_DIR:-./model_save/final_quantized_model}
TEST_CSV=${TEST_CSV:-./data/test.csv}
OUT_SUB=${OUT_SUB:-./sub/final_submission.csv}
CALIB_JSON=${CALIB_JSON:-${HEAD_DIR}/calibration.json}
python student_gptq_infer.py \
  --model-dir "$MODEL_DIR" \
  --test-csv "$TEST_CSV" \
  --out "$OUT_SUB" \
  --head-path "$HEAD_OUT" \
  --tta-lengths 2000 \
  --batch-size 4 \
  --max-length 2000 \
  --temperature-json "$CALIB_JSON"

echo "[Step8] Wrote $OUT_SUB"
