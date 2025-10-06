#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --job-name=TestJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

python ./main.py --mode student-train \
  --student-model microsoft/deberta-v3-large \
  --student-epochs 5 \
  --student-max-samples 0 \
  --student-label-smoothing 0.1 \
  --student-early-stopping 2 \
  --student-max-length 512 \
  --student-extra-csvs "data/ultrafeedback.csv,data/ultrafeedback_ties.csv,data/lmsys-33k-deduplicated.csv" \
  --student-dedup-by-prompt \
  --student-shuffle-ab \
  --student-fp16 \
  --student-gradient-checkpointing