import os
import json
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def train_lora(
    base_model: str,
    output_dir: str,
    data_path: str,
    tokenizer_path: Optional[str] = None,
    num_epochs: int = 1,
    lr: float = 1e-5,
    max_length: int = 1024,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    qlora: bool = False,
    bf16: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    # Ensure standard HTTP path; disable xet/transfer accelerations in-process just in case
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    tok_src = tokenizer_path or base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True, local_files_only=False, trust_remote_code=True)
    except Exception as e_fast:
        # Fallback: try slow tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False, local_files_only=False, trust_remote_code=True)
        except Exception as e_slow:
            raise RuntimeError(
                f"Failed to load tokenizer from '{tok_src}'. If your base model dir lacks tokenizer files, "
                f"pass --tokenizer-path pointing to a compatible model/repo with tokenizer files (e.g., meta-llama/Meta-Llama-3-70B)."
            ) from e_slow
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )

    # Allow passing a HF token via env if needed; step1 passes token at login level already
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        token=hf_token,
        trust_remote_code=True,
        local_files_only=False,
    )

    if qlora:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("csv", data_files={"train": data_path})

    def build_text(ex):
        # For post-pretrain, concatenate prompt+response like next-token prediction
        # (Assumes columns 'prompt' and 'chosen' or 'response' exist)
        prompt = ex.get("prompt") or ex.get("question") or ""
        resp = ex.get("chosen") or ex.get("response") or ex.get("answer") or ""
        return tokenizer(
            str(prompt) + "\n" + str(resp),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = ds["train"].map(build_text, batched=False)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--bf16", action="store_true")
    a = p.parse_args()

    train_lora(
        base_model=a.base_model,
        output_dir=a.output_dir,
        data_path=a.data_path,
        tokenizer_path=a.tokenizer_path,
        num_epochs=a.epochs,
        lr=a.lr,
        max_length=a.max_length,
        r=a.r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        qlora=a.qlora,
        bf16=a.bf16,
    )
