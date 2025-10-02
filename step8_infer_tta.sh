#!/bin/bash
# Step 8: Apply TTA during inference (GPTQ CausalLM + classification head adapter)
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

# Export a classification head from your best seq-classifier (if not already saved)
if [ ! -f model_save/student_distilbert/classifier_head.pt ]; then
  python export_classifier_head.py \
    --model-dir ./model_save/student_distilbert \
    --out ./model_save/student_distilbert/classifier_head.pt || true
fi

python student_gptq_infer.py \
  --model-dir ./model_save/final_quantized_model \
  --test-csv ./data/test.csv \
  --out ./sub/final_submission.csv \
  --head-path ./model_save/student_distilbert/classifier_head.pt \
  --tta-lengths 512,1024 \
  --batch-size 8 \
  --max-length 1024 \
  --temperature-json ./model_save/student_distilbert/calibration.json

echo "[Step8] Wrote ./sub/final_submission.csv"
