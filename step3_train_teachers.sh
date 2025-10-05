#!/bin/bash
# Step 3: Train teacher models per fold with 4/5 Kaggle train + external 33k augmentation, then merge adapters.
# Usage:
#   sbatch step3_train_teachers.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=360   # 6 hours wall-clock limit
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

# Allow user override of scratch root where step1 may have written merged post-pretrained models
SCRATCH_BASE=${SCRATCH_BASE:-/scratch-shared/tc1proj005}

echo "[Step3] SCRATCH_BASE candidate: ${SCRATCH_BASE}"

# Hugging Face gated model login (only if token available). You can override HF_TOKEN externally.
export HF_TOKEN="${HF_TOKEN:-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}"
python - <<'PY'
import os
tok=os.environ.get('HF_TOKEN')
if tok:
  try:
    from huggingface_hub import login
    login(token=tok, add_to_git_credential=False)
    print('[Step3] Hugging Face login succeeded (token provided).')
  except Exception as e:
    print('[Step3][Warn] HF login failed:', e)
else:
  print('[Step3] No HF_TOKEN set; proceeding without login.')
PY

KF=data/train.csv
EXT=data/ultrafeedback.csv
FOLDS_JSON=data/processed_data/folds_5_seed42.json
if [ ! -f "$FOLDS_JSON" ]; then
  echo "[Step3] Missing folds file $FOLDS_JSON. Run step2_make_folds.sh first." >&2
  exit 3
fi

echo "[Step3] Using Kaggle train: $KF | External: $EXT"
python - <<'PY'
import json, pandas as pd, os
KF='data/train.csv'
EXT='data/ultrafeedback.csv'
FOLDS_JSON='data/processed_data/folds_5_seed42.json'
folds=json.load(open(FOLDS_JSON))
train_df=pd.read_csv(KF)
ext_df=pd.read_csv(EXT)
# Basic sanity: ensure ext has compatible columns; we'll try to align prompt/response style
def normalize(df):
    # If original dataset columns differ, create 'prompt' and 'response' fallbacks
    cols=df.columns
    out=df.copy()
    if 'prompt' not in cols:
        # use first textual column heuristically
        text_cols=[c for c in cols if df[c].dtype==object]
        if text_cols:
            out['prompt']=df[text_cols[0]]
    if 'response' not in cols:
        cand=[c for c in ['chosen','answer','response','completion'] if c in cols]
        if cand:
            out['response']=df[cand[0]]
    return out
train_df=normalize(train_df)
ext_df=normalize(ext_df)
os.makedirs('data/fold_data', exist_ok=True)
for k, val_idx in folds.items():
    val_set=train_df.iloc[val_idx]
    train_idx=[i for i in range(len(train_df)) if i not in val_idx]
    tr_part=train_df.iloc[train_idx]
    combined=pd.concat([tr_part, ext_df], ignore_index=True)
    fold_train_path=f'data/fold_data/fold_{k}_train.csv'
    val_path=f'data/fold_data/fold_{k}_val.csv'
    combined.to_csv(fold_train_path, index=False)
    val_set.to_csv(val_path, index=False)
    print(f'[Step3][Prep] Fold {k}: train {combined.shape} (incl ext {len(ext_df)}) | val {val_set.shape}')
print('[Step3][Prep] Wrote per-fold train/val CSVs to data/fold_data/')
PY

# Robust resolution of local post-pretrained base model directories.
# We search in (1) explicit override, (2) model_save/, (3) repo root, (4) SCRATCH_BASE.
resolve_base_dir() {
  local override="$1"; shift
  local -a names=("$@")
  if [ -n "$override" ] && [ -d "$override" ]; then
    echo "$override"; return 0
  fi
  for n in "${names[@]}"; do
    for base in "." "model_save" "${SCRATCH_BASE}"; do
      if [ -d "${base}/${n}" ]; then
        echo "${base}/${n}"; return 0
      fi
    done
  done
  echo ""  # not found
}

LLAMA_CANDIDATES=(post_pretrain_llama3-8b_merged post_pretrain_llama3-70b_merged)
QWEN_CANDIDATES=(post_pretrain_qwen2-14b_merged post_pretrain_qwen2-72b_merged)

LLAMA_BASE_DIR=$(resolve_base_dir "${LLAMA_FT_BASE:-}" "${LLAMA_CANDIDATES[@]}")
QWEN_BASE_DIR=$(resolve_base_dir "${QWEN_FT_BASE:-}" "${QWEN_CANDIDATES[@]}")

SKIP_LLAMA=${SKIP_LLAMA:-0}
SKIP_QWEN=${SKIP_QWEN:-0}

if [ -z "$LLAMA_BASE_DIR" ]; then
  if [ "$SKIP_LLAMA" = "1" ]; then
    echo "[Step3][Info] SKIP_LLAMA=1 and no local LLaMA base found; LLaMA folds will be skipped." >&2
  else
    echo "[Step3][Error] No local LLaMA post-pretrained directory found. Set LLAMA_FT_BASE=/path/to/post_pretrain_llama*_merged or export SKIP_LLAMA=1 to bypass." >&2
    exit 41
  fi
fi
if [ -z "$QWEN_BASE_DIR" ]; then
  if [ "$SKIP_QWEN" = "1" ]; then
    echo "[Step3][Info] SKIP_QWEN=1 and no local Qwen base found; Qwen folds will be skipped." >&2
  else
    echo "[Step3][Error] No local Qwen post-pretrained directory found. Set QWEN_FT_BASE=/path/to/post_pretrain_qwen*_merged or export SKIP_QWEN=1 to bypass." >&2
    exit 42
  fi
fi

# Tokenizer paths (local merged dirs typically lack tokenizer files)
LLAMA_TOKENIZER=${LLAMA_TOKENIZER_PATH:-meta-llama/Meta-Llama-3.1-8B}
QWEN_TOKENIZER=${QWEN_TOKENIZER_PATH:-Qwen/Qwen2.5-14B}

echo "[Step3] LLaMA base: ${LLAMA_BASE_DIR:-<skipped>} | tokenizer: $LLAMA_TOKENIZER | skip=$SKIP_LLAMA"
echo "[Step3] Qwen base:  ${QWEN_BASE_DIR:-<skipped>} | tokenizer: $QWEN_TOKENIZER | skip=$SKIP_QWEN"

LORA_R=${TEACHER_LORA_R:-16}
LORA_ALPHA=${TEACHER_LORA_ALPHA:-32}
MAXLEN=${TEACHER_MAXLEN:-512}
GRAD_ACCUM=${TEACHER_GRAD_ACCUM:-8}
BS=${TEACHER_PER_DEVICE_BS:-1}
EPOCHS=${TEACHER_EPOCHS:-1}
LR=${TEACHER_LR:-1e-5}
SUBSET=${TEACHER_SUBSET_SIZE:--1}
MAX_STEPS=${TEACHER_MAX_STEPS:-300}
TIME_BUDGET_HOURS=${TEACHER_TIME_BUDGET_HOURS:-6}

# Auto shrink MAX_STEPS if naive estimate exceeds time budget.
# Rough per-step seconds heuristic (smaller post-pretrained base, QLoRA): 45s; adjust via TEACHER_EST_STEP_SEC
EST_STEP_SEC=${TEACHER_EST_STEP_SEC:-45}
TOTAL_FOLDS=5
MODELS_PER_FOLD=2
if [ -z "${TEACHER_MAX_STEPS:-}" ] || [ "${TEACHER_MAX_STEPS}" = "" ]; then
  TEACHER_MAX_STEPS=${MAX_STEPS}
fi
RAW_PROJECTED_SEC=$(( MAX_STEPS * EST_STEP_SEC * TOTAL_FOLDS * MODELS_PER_FOLD ))
BUDGET_SEC=$(( TIME_BUDGET_HOURS * 3600 ))
if [ ${RAW_PROJECTED_SEC} -gt ${BUDGET_SEC} ]; then
  # distribute budget evenly
  PER_MODEL_ALLOWED=$(( BUDGET_SEC / (TOTAL_FOLDS * MODELS_PER_FOLD) ))
  NEW_MAX=$(( PER_MODEL_ALLOWED / EST_STEP_SEC ))
  if [ ${NEW_MAX} -lt 10 ]; then NEW_MAX=10; fi
  echo "[Step3][AutoCap] Reducing MAX_STEPS from ${MAX_STEPS} to ${NEW_MAX} to fit ${TIME_BUDGET_HOURS}h budget (est step ${EST_STEP_SEC}s)" >&2
  MAX_STEPS=${NEW_MAX}
fi

for FOLD in 0 1 2 3 4; do
  TRAIN_CSV=data/fold_data/fold_${FOLD}_train.csv
  if [ "$SKIP_LLAMA" != "1" ]; then
    echo "[Step3] LLaMA fold ${FOLD} train csv: ${TRAIN_CSV}"
    python lora_train.py \
      --base-model "${LLAMA_BASE_DIR}" \
      --tokenizer-path "${LLAMA_TOKENIZER}" \
      --output-dir "model_save/llama_fold_${FOLD}_lora" \
      --data-path "${TRAIN_CSV}" \
      --bf16 \
      --qlora \
      --r "${LORA_R}" \
      --lora-alpha "${LORA_ALPHA}" \
      --max-length "${MAXLEN}" \
      --grad-accum "${GRAD_ACCUM}" \
      --per-device-batch "${BS}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --max-steps "${MAX_STEPS}" || { echo "[Step3][Error] LLaMA fold ${FOLD} failed"; exit 11; }
    python lora_merge.py \
      --base-model "${LLAMA_BASE_DIR}" \
      --lora-dir "model_save/llama_fold_${FOLD}_lora" \
      --out-dir "model_save/llama_fold_${FOLD}" || { echo "[Step3][Error] LLaMA merge fold ${FOLD} failed"; exit 12; }
  else
    echo "[Step3][Skip] LLaMA fold ${FOLD} due to SKIP_LLAMA=1"
  fi

  if [ "$SKIP_QWEN" != "1" ]; then
    echo "[Step3] Qwen fold ${FOLD} train csv: ${TRAIN_CSV}"
    python lora_train.py \
      --base-model "${QWEN_BASE_DIR}" \
      --tokenizer-path "${QWEN_TOKENIZER}" \
      --output-dir "model_save/qwen_fold_${FOLD}_lora" \
      --data-path "${TRAIN_CSV}" \
      --bf16 \
      --qlora \
      --r "${LORA_R}" \
      --lora-alpha "${LORA_ALPHA}" \
      --max-length "${MAXLEN}" \
      --grad-accum "${GRAD_ACCUM}" \
      --per-device-batch "${BS}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --max-steps "${MAX_STEPS}" || { echo "[Step3][Error] Qwen fold ${FOLD} failed"; exit 21; }
    python lora_merge.py \
      --base-model "${QWEN_BASE_DIR}" \
      --lora-dir "model_save/qwen_fold_${FOLD}_lora" \
      --out-dir "model_save/qwen_fold_${FOLD}" || { echo "[Step3][Error] Qwen merge fold ${FOLD} failed"; exit 22; }
  else
    echo "[Step3][Skip] Qwen fold ${FOLD} due to SKIP_QWEN=1"
  fi
done

echo "[Step3] Done"
