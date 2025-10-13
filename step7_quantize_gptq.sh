#!/bin/bash
# Step 7: Quantize final merged model to 8-bit (GPTQ with calibration)
# Usage:
#   sbatch step7_quantize_gptq.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --time=180
#SBATCH --cpus-per-task=4
#SBATCH --job-name=S7_GPTQ
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

MODEL_DIR=${MODEL_DIR:-model_save/final_merged_model}
OUT_DIR=${OUT_DIR:-model_save/final_quantized_model}
CALIB_CSV=${CALIB_CSV:-./data/train.csv}
TOKENIZER_DIR=${TOKENIZER_DIR:-google/gemma-2-9b-it}
# Use 8-bit per the summary, keep group size reasonable
python quantize_gptq_calibrated.py \
  --model-dir "$MODEL_DIR" \
  --out-dir "$OUT_DIR" \
  --calib-csv "$CALIB_CSV" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --bits 8 \
  --group-size 128 \
  --max-calib-samples 2048 \
  --max-length 1024

echo "[Step7] Done"
