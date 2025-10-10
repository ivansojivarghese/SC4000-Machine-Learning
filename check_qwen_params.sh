#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=10
#SBATCH --cpus-per-task=1
#SBATCH --job-name=CheckQwen
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv
cd ~/exported-assets_sc4000

python - <<'PY'
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

path = "/scratch-shared/tc1proj005/post_pretrain_qwen2-72b_merged"  # adjust if needed
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
with init_empty_weights():
    mdl = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
print("num_hidden_layers =", getattr(cfg, "num_hidden_layers", None))
print("hidden_size       =", getattr(cfg, "hidden_size", None))
print("n_heads           =", getattr(cfg, "num_attention_heads", None))
print("approx_params     =", sum(p.numel() for p in mdl.parameters())/1e9, "B")
PY