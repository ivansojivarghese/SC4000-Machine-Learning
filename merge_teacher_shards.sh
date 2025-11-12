#!/bin/bash
# Merge sharded teacher outputs for a given fold.
# Usage examples:
#   FOLD=0 MODELS=llama SHARDS=5 RECOMPUTE_ENSEMBLE=1 sbatch merge_teacher_shards.sh
#   sbatch --array=0-2 merge_teacher_shards.sh  # uses FOLD from SLURM_ARRAY_TASK_ID

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --time=30
#SBATCH --cpus-per-task=2
#SBATCH --job-name=MergeTeacherShards
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

FOLD=${FOLD:-${SLURM_ARRAY_TASK_ID:-0}}
MODELS=${MODELS:-llama}
SHARDS=${SHARDS:-5}
DIR=${DIR:-model_save/teacher_logits}
RECOMPUTE_ENSEMBLE=${RECOMPUTE_ENSEMBLE:-1}
PREFER_PARQUET=${PREFER_PARQUET:-1}

echo "[MergeJob] fold=$FOLD models=$MODELS shards=$SHARDS dir=$DIR recompute=$RECOMPUTE_ENSEMBLE parquet=$PREFER_PARQUET"

ARGS=(--fold "$FOLD" --models "$MODELS" --dir "$DIR" --shards "$SHARDS")
if [[ "$RECOMPUTE_ENSEMBLE" == "1" ]]; then
  ARGS+=(--recompute-ensemble)
fi
if [[ "$PREFER_PARQUET" == "1" ]]; then
  ARGS+=(--prefer-parquet)
fi

python merge_teacher_shards.py "${ARGS[@]}"

echo "[MergeJob] Done"
