#!/bin/bash
# Step 7: Quantize final model to 8-bit (GPTQ with calibration)
# Usage:
#   sbatch step7_quantize_gptq.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
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

python quantize_gptq_calibrated.py \
  --model-dir model_save/final_merged_model \
  --out-dir model_save/final_quantized_model \
  --calib-csv ./data/train.csv \
  --bits 4 \
  --group-size 128 \
  --max-calib-samples 1024 \
  --max-length 512

echo "[Step7] Done"
