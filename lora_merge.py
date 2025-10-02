import os
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM


def merge_lora(base_model: str, lora_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Ensure standard HTTP path; disable xet/transfer accelerations in-process
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        local_files_only=False,
    )
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
