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
    num_epochs: float = 1.0,
    lr: float = 1e-5,
    max_length: int = 1024,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    qlora: bool = False,
    bf16: bool = False,
    attn_impl: str = None,
    grad_checkpoint: bool = False,
    load_8bit: bool = False,
    grad_accum: int = 64,
    subset_size: int = None,
    per_device_batch: int = 1,
    max_steps: int = -1,
):
    os.makedirs(output_dir, exist_ok=True)
    # Ensure standard HTTP path; disable xet/transfer accelerations in-process just in case
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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
    from functools import partial
    model_kwargs = dict(
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        token=hf_token,
        trust_remote_code=True,
        local_files_only=False,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    if load_8bit and not qlora:
        # Only use 8-bit when not already doing 4bit QLoRA
        model_kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Prepare model for k-bit training in both 4-bit (QLoRA) and 8-bit LoRA scenarios
    if qlora or load_8bit:
        model = prepare_model_for_kbit_training(model)
    if grad_checkpoint:
        model.gradient_checkpointing_enable()
        if hasattr(model, 'config'):
            model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Debug: report trainable vs total parameters to ensure LoRA adapters are active
    try:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = (trainable / total * 100) if total else 0.0
        print(f"[DEBUG] Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
        if trainable == 0:
            raise RuntimeError(
                "No trainable parameters detected after applying LoRA. "
                "If using --load-8bit without --qlora, ensure prepare_model_for_kbit_training is applied."
            )
    except Exception as debug_e:
        print(f"[WARN] Failed to compute trainable parameter stats: {debug_e}")

    # Robust CSV -> Dataset loading. Arrow inference sometimes fails if a column mixes numeric and string tokens
    # (e.g., a mostly numeric column with a stray string like 'evol_instruct'). We attempt standard load first;
    # on failure we fall back to pandas and coerce mixed columns to string to avoid DatasetGenerationError.
    from datasets import Dataset, DatasetDict
    ds = None
    try:
        ds = load_dataset("csv", data_files={"train": data_path})
    except Exception as e_csv:
        print(f"[WARN] load_dataset csv auto-infer failed: {type(e_csv).__name__}: {e_csv}\n[WARN] Falling back to pandas coercion mode.")
        import pandas as pd
        # Read with low_memory=False so pandas infers types in one pass; then coerce object-like / mixed to string.
        pdf = pd.read_csv(data_path, low_memory=False)
        # Force every column to pure string to completely sidestep Arrow casting issues (safe for text-only training).
        for col in pdf.columns:
            pdf[col] = pdf[col].astype(str)
        # Replace any 'nan' textual artifacts after cast.
        pdf = pdf.replace({'nan': ''}).fillna("")
        ds = DatasetDict({"train": Dataset.from_pandas(pdf, preserve_index=False)})
        print(f"[INFO] Fallback dataset constructed: {len(ds['train'])} rows | columns: {list(pdf.columns)[:15]}{'...' if len(pdf.columns)>15 else ''}")
        # Optionally persist a debug sample
        try:
            pdf.head(50).to_csv(os.path.join(output_dir, 'debug_dataset_sample.csv'), index=False)
        except Exception as _e_dbg:
            print(f"[WARN] Could not write debug sample: {_e_dbg}")
    if subset_size is not None:
        if subset_size > 0:
            # Take a deterministic head subset for speed
            orig_len = len(ds["train"])
            capped = min(subset_size, orig_len)
            ds["train"] = ds["train"].select(range(capped))
            print(f"[INFO] Subsetting training data: {capped} / {orig_len} examples (requested {subset_size})")
        else:
            print(f"[INFO] subset_size={subset_size} -> ignoring and using full dataset of {len(ds['train'])} examples")

    pad_token_id = tokenizer.pad_token_id
    def build_text(ex):
        prompt = ex.get("prompt") or ex.get("question") or ""
        resp = ex.get("chosen") or ex.get("response") or ex.get("answer") or ""
        text = (str(prompt).strip() + "\n" if prompt else "") + str(resp).strip()
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        input_ids = encoded["input_ids"]
        labels = [tid if tid != pad_token_id else -100 for tid in input_ids]
        encoded["labels"] = labels
        return encoded

    tokenized = ds["train"].map(build_text, batched=False)

    # If max_steps provided (>0), we let Trainer cap steps and ignore epochs after reaching it.
    # Trainer expects an int for max_steps; use -1 (default behavior) instead of None
    eff_max_steps = max_steps if (isinstance(max_steps, int) and max_steps > 0) else -1
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        max_steps=eff_max_steps,
        learning_rate=lr,
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=grad_checkpoint,
        logging_steps=20,
        save_strategy="epoch" if eff_max_steps == -1 else "steps",
        save_steps=max(100, (eff_max_steps // 5) if eff_max_steps > 0 else 100),
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
    p.add_argument("--epochs", type=float, default=1.0, help="Number of epochs (can be fractional, ignored early if --max-steps hit)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--attn-impl", type=str, default=None, help="Attention implementation override (e.g. eager, sdpa, flash_attention_2)")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit (ignored if --qlora)")
    p.add_argument("--grad-accum", type=int, default=64, help="Gradient accumulation steps (global batch = per_device * grad_accum)")
    p.add_argument("--subset-size", type=int, default=None, help="Optional cap on number of training examples")
    p.add_argument("--per-device-batch", type=int, default=1, help="Per-device (GPU) batch size")
    p.add_argument("--max-steps", type=int, default=-1, help="Stop after this many optimizer steps (overrides epochs if >0)")
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
        attn_impl=a.attn_impl,
        grad_checkpoint=a.grad_checkpoint,
        load_8bit=a.load_8bit,
        grad_accum=a.grad_accum,
        subset_size=a.subset_size,
        per_device_batch=a.per_device_batch,
        max_steps=a.max_steps,
    )
