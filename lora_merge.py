import os
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM


def merge_lora(base_model: str, lora_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    peft = PeftModel.from_pretrained(base, lora_dir)
    merged = peft.merge_and_unload()
    merged.save_pretrained(out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    merge_lora(args.base_model, args.lora_dir, args.out_dir)
