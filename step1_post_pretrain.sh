#!/bin/bash
# Step 1: Post-pretrain large models on UltraFeedback via LoRA/QLoRA, then merge adapters.
# Usage:
#   sbatch step1_post_pretrain.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --cpus-per-task=8
#SBATCH --job-name=S1_PostPretrain
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

# Tokenizer sources (set these to LOCAL directories if your cluster has no HF access or the repos are gated):
# Use local path if set, otherwise fall back
# Default tokenizer repos
LLAMA_TOK=${LLAMA_TOK:-meta-llama/Meta-Llama-3.1-70B}
QWEN_TOK=${QWEN_TOK:-Qwen/Qwen2.5-72B}
GEMMA_TOK=${GEMMA_TOK:-google/gemma-2-9b}

# Base model repos for LoRA training (use remote IDs, not local dirs)
LLAMA_BASE=${LLAMA_BASE:-meta-llama/Meta-Llama-3.1-70B}
QWEN_BASE=${QWEN_BASE:-Qwen/Qwen2.5-72B}
GEMMA_BASE=${GEMMA_BASE:-google/gemma-2-9b}

export LLAMA_TOK
export QWEN_TOK
export GEMMA_TOK

# Basic dependency check; install if needed
python - <<'PY'
import importlib, sys
for m in ['torch','transformers','peft','datasets','pandas','numpy','huggingface_hub']:
  try:
    importlib.import_module(m)
  except Exception as e:
    print('[Step1] Missing', m, '-> will install')
    sys.exit(1)
print('[Step1] Deps OK')
PY
if [ $? -ne 0 ]; then
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r requirements.txt
fi

# Ensure online mode and login if token provided; prefetch Qwen tokenizer using snapshot_download (no hf_transfer)
export TRANSFORMERS_OFFLINE=0
# Hard-disable accelerated transports (hf-transfer and xet) in all known env var spellings
export HF_HUB_ENABLE_HF_TRANSFER=0
export HUGGINGFACE_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ENABLE_HF_XET=0
export HUGGINGFACE_HUB_ENABLE_HF_XET=0
export HF_HUB_DISABLE_XET=1
export HF_TOKEN="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"

python - <<'PY'
import os, sys
print("[Step1] Online mode enabled (TRANSFORMERS_OFFLINE=0)")
from huggingface_hub import login
# Hardcoded token per user request (do not read env). Avoid adding to Git credential store to prevent git/xet path.
token = "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"
login(token=token, add_to_git_credential=False)
print("[Step1] Logged in to Hugging Face (hardcoded token)")

qwen_tok = os.environ.get("QWEN_TOK", "Qwen/Qwen2.5-72B")
from transformers import AutoTokenizer
from pathlib import Path

def is_local_dir(p:str)->bool:
  try:
    return Path(p).exists() and Path(p).is_dir()
  except Exception:
    return False

local_tok_dir = None
if is_local_dir(qwen_tok):
  local_tok_dir = qwen_tok
  print(f"[Step1] Using local tokenizer dir: {local_tok_dir}")
else:
  try:
    from huggingface_hub import snapshot_download
    allow = [
      'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
      'vocab.json', 'merges.txt', 'tokenizer.model', '*.model', '*.txt'
    ]
    print(f"[Step1] Downloading tokenizer files for {qwen_tok} (no hf_transfer)...")
    token = "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"
    snapshot_path = snapshot_download(repo_id=qwen_tok, allow_patterns=allow, token=token)
    local_tok_dir = snapshot_path
    print(f"[Step1] Tokenizer snapshot at: {snapshot_path}")
  except Exception as e:
    print(f"[Step1] snapshot_download failed for {qwen_tok}: {e}")
    sys.exit(2)

try:
  tok = AutoTokenizer.from_pretrained(local_tok_dir, trust_remote_code=True, use_fast=True)
  print(f"[Step1] Prefetched tokenizer from {local_tok_dir}: vocab size {len(tok)}")
except Exception as e:
  print(f"[Step1] AutoTokenizer load failed from {local_tok_dir}: {e}")
  sys.exit(2)
PY

echo "[Step1] Train LoRA adapters on UT datasets and merge"

echo "[Step1] Gemma2-9B LoRA: starting"
# Gemma2-9B (LoRA)
python lora_train.py \
  --base-model "${GEMMA_BASE}" \
  --output-dir model_save/post_pretrain_gemma2-9b_lora \
  --data-path data/ultrafeedback.csv \
  --tokenizer-path "${GEMMA_TOK}" \
  --bf16 \
  --epochs 1 \
  --lr 1e-5 \
  --max-length 1024 \
  --r 64 \
  --lora-alpha 128

python lora_merge.py \
  --base-model "${GEMMA_BASE}" \
  --lora-dir model_save/post_pretrain_gemma2-9b_lora \
  --out-dir model_save/post_pretrain_gemma2-9b_merged

echo "[Step1] Qwen2.5-72B QLoRA: starting"
# Qwen2-72B (QLoRA)
python lora_train.py \
  --base-model "${QWEN_BASE}" \
  --output-dir model_save/post_pretrain_qwen2-72b_lora \
  --data-path data/ultrafeedback.csv \
  --tokenizer-path "${QWEN_TOK}" \
  --bf16 \
  --qlora \
  --epochs 1 \
  --lr 1e-5 \
  --max-length 1024 \
  --r 64 \
  --lora-alpha 128

python lora_merge.py \
  --base-model "${QWEN_BASE}" \
  --lora-dir model_save/post_pretrain_qwen2-72b_lora \
  --out-dir model_save/post_pretrain_qwen2-72b_merged

echo "[Step1] LLaMA3.1-70B QLoRA: starting"
# LLaMA3-70B (assumes HF_TOKEN is available)
LLAMA_TOK_RESOLVED="${LLAMA_TOK}"
if [ -d "${LLAMA_TOK}" ]; then
  echo "[Step1] Using local LLaMA tokenizer dir: ${LLAMA_TOK}"
else
  LLAMA_TOK_RESOLVED=$(python - <<'PY'
from huggingface_hub import snapshot_download
import os
tok=os.environ['LLAMA_TOK']
allow=['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','tokenizer.model','*.model','*.txt']
token="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"
print(snapshot_download(repo_id=tok, allow_patterns=allow, token=token))
PY
)
  echo "[Step1] LLaMA tokenizer snapshot at: ${LLAMA_TOK_RESOLVED}"
fi

python lora_train.py \
  --base-model "${LLAMA_BASE}" \
  --output-dir model_save/post_pretrain_llama3-70b_lora \
  --data-path data/ultrafeedback.csv \
  --tokenizer-path "${LLAMA_TOK_RESOLVED}" \
  --bf16 \
  --qlora \
  --epochs 1 \
  --lr 1e-5 \
  --max-length 1024 \
  --r 64 \
  --lora-alpha 128

python lora_merge.py \
  --base-model "${LLAMA_BASE}" \
  --lora-dir model_save/post_pretrain_llama3-70b_lora \
  --out-dir model_save/post_pretrain_llama3-70b_merged

echo "[Step1] Done"
