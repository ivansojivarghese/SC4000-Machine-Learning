#!/bin/bash
# Minimal: Post-pretrain and merge for Gemma, Qwen, or LLaMA. Usage:
#   RUN_STAGE=gemma sbatch step1_post_pretrain.sh
#   RUN_STAGE=qwen sbatch step1_post_pretrain.sh
#   RUN_STAGE=llama sbatch step1_post_pretrain.sh

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

# Prefetch all models/tokenizers to /scratch-shared/tc1proj005
export HF_HOME="/scratch-shared/tc1proj005"
export HUGGINGFACE_HUB_CACHE="/scratch-shared/tc1proj005"
export HF_DATASETS_CACHE="/scratch-shared/tc1proj005"
export TRANSFORMERS_CACHE="/scratch-shared/tc1proj005"

echo "[Step1] Downloading/caching all models and tokenizers to $HF_HOME ..."
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
  "google/gemma-2-9b",
  "Qwen/Qwen2.5-7B-Instruct",
  "meta-llama/Llama-3.1-8B-Instruct"
]:
  print(f"[Step1] Downloading {repo} ...")
  snapshot_download(repo_id=repo, local_dir="/scratch-shared/tc1proj005", local_dir_use_symlinks=False, resume_download=True)
PY

# Common hyperparameters
UT_DATA="data/ultrafeedback.csv"
LR=1e-5
MAXLEN=1024
PER_DEVICE_BS=1
LORA_R=64
LORA_ALPHA=128

RUN_STAGE=${RUN_STAGE:-gemma}

if [ "$RUN_STAGE" = "gemma" ]; then
  BASE="google/gemma-2-9b"
  TOK="google/gemma-2-9b"
  OUT="post_pretrain_gemma2-9b"
  EPOCHS=2
  GRAD_ACCUM=64
  echo "[Step1] Starting Gemma2-9B LoRA post-pretraining"
  python lora_train.py \
    --base-model "$BASE" \
    --output-dir "$OUT" \
    --data-path "$UT_DATA" \
    --tokenizer-path "$TOK" \
    --bf16 \
    --attn-impl eager \
    --load-8bit \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-length "$MAXLEN" \
    --r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --per-device-batch "$PER_DEVICE_BS" \
    --grad-accum "$GRAD_ACCUM"
  echo "[Step1] Merging LoRA adapters into base model"
  python lora_merge.py \
    --base-model "$BASE" \
    --lora-dir "$OUT" \
    --out-dir "${OUT}_merged"
  echo "[Step1] Gemma done."

elif [ "$RUN_STAGE" = "qwen" ]; then
  BASE="Qwen/Qwen2.5-7B-Instruct"
  TOK="Qwen/Qwen2.5-7B-Instruct"
  OUT="post_pretrain_qwen2-7b-instruct"
  EPOCHS=2
  GRAD_ACCUM=64
  echo "[Step1] Starting Qwen2.5-7B-Instruct QLoRA post-pretraining"
  python lora_train.py \
    --base-model "$BASE" \
    --output-dir "$OUT" \
    --data-path "$UT_DATA" \
    --tokenizer-path "$TOK" \
    --bf16 \
    --qlora \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-length "$MAXLEN" \
    --r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --per-device-batch "$PER_DEVICE_BS" \
    --grad-accum "$GRAD_ACCUM"
  echo "[Step1] Merging QLoRA adapters into base model"
  python lora_merge.py \
    --base-model "$BASE" \
    --lora-dir "$OUT" \
    --out-dir "${OUT}_merged"
  echo "[Step1] Qwen done."

elif [ "$RUN_STAGE" = "llama" ]; then
  BASE="meta-llama/Llama-3.1-8B-Instruct"
  TOK="meta-llama/Llama-3.1-8B-Instruct"
  OUT="post_pretrain_llama3-8b-instruct"
  EPOCHS=2
  GRAD_ACCUM=64
  echo "[Step1] Starting LLaMA-3.1-8B-Instruct QLoRA post-pretraining"
  python lora_train.py \
    --base-model "$BASE" \
    --output-dir "$OUT" \
    --data-path "$UT_DATA" \
    --tokenizer-path "$TOK" \
    --bf16 \
    --qlora \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-length "$MAXLEN" \
    --r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --per-device-batch "$PER_DEVICE_BS" \
    --grad-accum "$GRAD_ACCUM"
  echo "[Step1] Merging QLoRA adapters into base model"
  python lora_merge.py \
    --base-model "$BASE" \
    --lora-dir "$OUT" \
    --out-dir "${OUT}_merged"
  echo "[Step1] LLaMA done."

else
  echo "[Step1] Unknown RUN_STAGE: $RUN_STAGE. Use one of: gemma | qwen | llama" >&2
  exit 2
fi

echo "[Step1] All done."
def is_local_dir(p:str)->bool:
echo "[Step1] Stage '${RUN_STAGE}' completed"

#!/bin/bash
# Simplified: Post-pretrain Gemma2-9B on UltraFeedback via LoRA, then merge adapters.
# Usage: sbatch step1_post_pretrain.sh

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

# Basic config (edit as needed)
GEMMA_BASE="google/gemma-2-9b"
GEMMA_TOK="google/gemma-2-9b"
UT_DATA="data/ultrafeedback.csv"
OUT_BASE="post_pretrain_gemma2-9b"
EPOCHS=1
LR=1e-5
MAXLEN=1024
PER_DEVICE_BS=1
LORA_R=64
LORA_ALPHA=128

echo "[Step1] Starting Gemma2-9B LoRA post-pretraining"
#!/bin/bash
# Minimal: Post-pretrain and merge for Gemma, Qwen, or LLaMA. Usage:
#   RUN_STAGE=gemma sbatch step1_post_pretrain.sh
#   RUN_STAGE=qwen sbatch step1_post_pretrain.sh
#   RUN_STAGE=llama sbatch step1_post_pretrain.sh

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

UT_DATA="data/ultrafeedback.csv"
EPOCHS=1
LR=1e-5
MAXLEN=1024
PER_DEVICE_BS=1
LORA_R=64
LORA_ALPHA=128

RUN_STAGE=${RUN_STAGE:-gemma}


if [ "$RUN_STAGE" = "gemma" ]; then
  BASE="google/gemma-2-9b"
  TOK="google/gemma-2-9b"
  OUT="post_pretrain_gemma2-9b"
  EPOCHS=2
  GRAD_ACCUM=64
  echo "[Step1] Starting Gemma2-9B LoRA post-pretraining"
  python lora_train.py \
    --base-model "$BASE" \
    --output-dir "$OUT" \
    --data-path "$UT_DATA" \
    --tokenizer-path "$TOK" \
    --bf16 \
    --attn-impl eager \
    --load-8bit \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-length "$MAXLEN" \
    --r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --per-device-batch "$PER_DEVICE_BS" \
    --grad-accum "$GRAD_ACCUM"
  echo "[Step1] Merging LoRA adapters into base model"
  python lora_merge.py \
    --base-model "$BASE" \
    --lora-dir "$OUT" \
    --out-dir "${OUT}_merged"
  echo "[Step1] Gemma done."

elif [ "$RUN_STAGE" = "qwen" ]; then
  BASE="Qwen/Qwen2.5-7B-Instruct"
  TOK="Qwen/Qwen2.5-7B-Instruct"
  OUT="post_pretrain_qwen2-7b-instruct"
  EPOCHS=2
  GRAD_ACCUM=64
  echo "[Step1] Starting Qwen2.5-7B-Instruct QLoRA post-pretraining"
  python lora_train.py \
    --base-model "$BASE" \
    --output-dir "$OUT" \
    --data-path "$UT_DATA" \
    --tokenizer-path "$TOK" \
    --bf16 \
    --qlora \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-length "$MAXLEN" \
    --r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --per-device-batch "$PER_DEVICE_BS" \
    --grad-accum "$GRAD_ACCUM"
  echo "[Step1] Merging QLoRA adapters into base model"
  python lora_merge.py \
    --base-model "$BASE" \
    --lora-dir "$OUT" \
    --out-dir "${OUT}_merged"
  echo "[Step1] Qwen done."

elif [ "$RUN_STAGE" = "llama" ]; then
  BASE="meta-llama/Llama-3.1-8B-Instruct"
  TOK="meta-llama/Llama-3.1-8B-Instruct"
  OUT="post_pretrain_llama3-8b-instruct"
  EPOCHS=2
  GRAD_ACCUM=64
  echo "[Step1] Starting LLaMA-3.1-8B-Instruct QLoRA post-pretraining"
  python lora_train.py \
    --base-model "$BASE" \
    --output-dir "$OUT" \
    --data-path "$UT_DATA" \
    --tokenizer-path "$TOK" \
    --bf16 \
    --qlora \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-length "$MAXLEN" \
    --r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --per-device-batch "$PER_DEVICE_BS" \
    --grad-accum "$GRAD_ACCUM"
  echo "[Step1] Merging QLoRA adapters into base model"
  python lora_merge.py \
    --base-model "$BASE" \
    --lora-dir "$OUT" \
    --out-dir "${OUT}_merged"
  echo "[Step1] LLaMA done."

else
  echo "[Step1] Unknown RUN_STAGE: $RUN_STAGE. Use one of: gemma | qwen | llama" >&2
  exit 2
fi
