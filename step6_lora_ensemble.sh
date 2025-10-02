#!/bin/bash
# Step 6: Ensemble LoRA layers from 5 folds (placeholder logic).
# Usage:
#   sbatch step6_lora_ensemble.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --time=30
#SBATCH --cpus-per-task=2
#SBATCH --job-name=S6_LoRAEnsemble
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

cd ~/exported-assets_sc4000

echo "[Step6] Placeholder: average LoRA adapters across folds and merge into base"
echo "[Step6] For now, pick the best fold adapter and merge as your final model:"
echo "  python lora_merge.py --base-model model_save/post_pretrain_llama3-70b --lora-dir model_save/llama3-70b_fold_0_lora --out-dir model_save/final_merged_model"
echo "[Step6] If you want a script to actually average adapters across folds, ask me to add lora_average.py."
