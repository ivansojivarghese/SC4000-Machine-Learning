#!/bin/bash
# Step 4: Infer logits for all training data for each teacher fold; save as [N,3] .npy.
# Usage:
#   sbatch step4_infer_teacher_logits.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=720
#SBATCH --cpus-per-task=8
#SBATCH --job-name=S4_TeacherInfer
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
  echo "[Step4] Inferring logits for LLaMA3 fold ${FOLD}"
  # TODO: replace with your actual teacher inference that writes [N,3] raw logits aligned to train rows
  # python teacher_logits_infer.py --model-dir model_save/llama3-70b_fold_${FOLD} --out model_save/llama3-70b_fold_${FOLD}/teacher_logits_fold_${FOLD}.npy
  python - <<PY
import numpy as np, os
os.makedirs('model_save/llama3-70b_fold_${FOLD}', exist_ok=True)
# Placeholder: write a random logits file of the correct shape; replace this with real inference.
N = len(open('data/train.csv','r').read().strip().splitlines()) - 1
np.save('model_save/llama3-70b_fold_${FOLD}/teacher_logits_fold_${FOLD}.npy', np.random.randn(N,3).astype('float32'))
print('[Step4] wrote llama3 logits for fold ${FOLD}', (N,3))
PY

  echo "[Step4] Inferring logits for Qwen2 fold ${FOLD}"
  # TODO: replace with your actual teacher inference
  # python teacher_logits_infer.py --model-dir model_save/qwen2-72b_fold_${FOLD} --out model_save/qwen2-72b_fold_${FOLD}/teacher_logits_fold_${FOLD}.npy
  python - <<PY
import numpy as np, os
os.makedirs('model_save/qwen2-72b_fold_${FOLD}', exist_ok=True)
N = len(open('data/train.csv','r').read().strip().splitlines()) - 1
np.save('model_save/qwen2-72b_fold_${FOLD}/teacher_logits_fold_${FOLD}.npy', np.random.randn(N,3).astype('float32'))
print('[Step4] wrote qwen2 logits for fold ${FOLD}', (N,3))
PY
done

echo "[Step4] Validating shapes"
python teacher_logits_validator.py

echo "[Step4] Done"
