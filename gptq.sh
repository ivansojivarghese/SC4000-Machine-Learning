#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=10
#SBATCH --cpus-per-task=1
#SBATCH --job-name=GPTQ
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv
cd ~/exported-assets_sc4000

python - <<'PY'
#!/usr/bin/env python3
"""
merge_and_quantize_lora.py

This script:
1. Loads a base model (BF16/FP16)
2. Loads a LoRA adapter using PEFT
3. Merges the LoRA weights into the base model
4. Quantizes the merged model to 4-bit (BitsAndBytes NF4)
5. Saves the final quantized model
"""

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import torch
import os


# ===============================
# 🔧 User Configuration
# ===============================
# Use repo-standard paths
BASE_MODEL_PATH = "google/gemma-2-9b-it"
LORA_ADAPTER_PATH = "model_save/avg_lora"
MERGED_OUTPUT_DIR = "model_save/final_merged_model"
QUANTIZED_OUTPUT_DIR = "model_save/final_quantized_model"

# Optional: enable BF16 if available
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ===============================
# 🚀 Step 1: Load Base Model
# ===============================
print("\n[1/5] Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=3,   # 👈 match adapter’s label count
    torch_dtype=torch_dtype,
    device_map="auto",
)

print("✅ Base model loaded.")


# ===============================
# 🚀 Step 2: Load LoRA Adapter
# ===============================
print("\n[2/5] Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
print("✅ LoRA adapter attached.")


# ===============================
# 🚀 Step 3: Merge and Unload
# ===============================
print("\n[3/5] Merging LoRA adapter into base model...")
merged_model = model.merge_and_unload()
print("✅ Merge complete. Adapter weights are now integrated into the model.")

# Save merged BF16 model
os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
merged_model.save_pretrained(MERGED_OUTPUT_DIR)
tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
print(f"💾 Saved merged model to {MERGED_OUTPUT_DIR}")


# ===============================
# 🚀 Step 4: Quantize to 4-bit (BNB NF4)
# ===============================
print("\n[4/5] Quantizing merged model to 4-bit (BitsAndBytes NF4)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

quantized_model = AutoModelForCausalLM.from_pretrained(
    MERGED_OUTPUT_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)

print("✅ Quantization complete (4-bit NF4).")


# ===============================
# 🚀 Step 5: Save Quantized Model
# ===============================
os.makedirs(QUANTIZED_OUTPUT_DIR, exist_ok=True)
quantized_model.save_pretrained(QUANTIZED_OUTPUT_DIR)
tokenizer.save_pretrained(QUANTIZED_OUTPUT_DIR)

print(f"💾 Saved quantized model to {QUANTIZED_OUTPUT_DIR}")
print("\n🎉 All done! Your model is merged and quantized successfully.")

PY