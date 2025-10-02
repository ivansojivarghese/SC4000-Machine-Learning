#!/bin/bash
# Step 3: Train LLaMA3-70B & Qwen2-72B per fold (LoRA/QLoRA), sequentially.
# Usage:
#   sbatch step3_train_teachers.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=2160
#SBATCH --cpus-per-task=8
#SBATCH --job-name=S3_Teachers
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

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
  echo "[Step3] LLaMA3 fold ${FOLD}"
  python lora_train.py \
    --base-model model_save/post_pretrain_llama3-70b_merged \
    --out-dir model_save/llama3-70b_fold_${FOLD}_lora \
    --bf16
  python lora_merge.py \
    --base-model model_save/post_pretrain_llama3-70b_merged \
    --lora-dir model_save/llama3-70b_fold_${FOLD}_lora \
    --out-dir model_save/llama3-70b_fold_${FOLD}

  echo "[Step3] Qwen2 fold ${FOLD}"
  python lora_train.py \
    --base-model model_save/post_pretrain_qwen2-72b_merged \
    --out-dir model_save/qwen2-72b_fold_${FOLD}_lora \
    --bf16
  python lora_merge.py \
    --base-model model_save/post_pretrain_qwen2-72b_merged \
    --lora-dir model_save/qwen2-72b_fold_${FOLD}_lora \
    --out-dir model_save/qwen2-72b_fold_${FOLD}
done

echo "[Step3] Done"
