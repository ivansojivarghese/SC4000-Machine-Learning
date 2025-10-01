import argparse
import os
import random
from typing import List, Dict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
except Exception as e:
    AutoGPTQForCausalLM = None
    BaseQuantizeConfig = None


def build_input_text(row: pd.Series) -> str:
    # Reuse the same formatting used by the student pipeline for distribution matching
    if {'prompt', 'response_a', 'response_b'}.issubset(row.index):
        return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"
    # Fallback to single text column
    if 'text' in row.index:
        return str(row['text']).strip()
    # Best-effort join of string-like columns
    parts = []
    for c in row.index:
        v = row[c]
        if isinstance(v, str) and v:
            parts.append(v)
    return " ".join(parts) if parts else ""


def make_calibration_examples(
    tokenizer: AutoTokenizer,
    csv_path: str,
    text_columns_hint: str = '',
    max_samples: int = 1024,
    max_length: int = 512,
) -> List[Dict[str, np.ndarray]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Optional column filtering if user provides a hint like: "prompt,response_a,response_b"
    if text_columns_hint:
        cols = [c.strip() for c in text_columns_hint.split(',') if c.strip()]
        cols = [c for c in cols if c in df.columns]
        if cols:
            # Keep only hinted columns
            df = df[cols]

    # Sample up to max_samples uniformly at random
    n = len(df)
    if n == 0:
        raise ValueError("Calibration CSV is empty.")
    idx = list(range(n))
    if n > max_samples:
        random.shuffle(idx)
        idx = idx[:max_samples]
    df = df.iloc[idx].reset_index(drop=True)

    texts = (df.apply(build_input_text, axis=1)).tolist()

    examples: List[Dict[str, np.ndarray]] = []
    # Batch-tokenize to speed things up a bit
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )
        input_ids = enc['input_ids']  # [B, T] torch.LongTensor
        attention_mask = enc['attention_mask']  # [B, T] torch.LongTensor
        for j in range(input_ids.shape[0]):
            examples.append({
                'input_ids': input_ids[j].cpu(),
                'attention_mask': attention_mask[j].cpu(),
            })
    return examples


def quantize_with_calibration(
    base_model_dir: str,
    out_dir: str,
    calib_csv: str,
    text_columns_hint: str = '',
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
    max_calib_samples: int = 1024,
    max_length: int = 512,
    seed: int = 42,
    use_safetensors: bool = True,
):
    if AutoGPTQForCausalLM is None:
        raise RuntimeError("auto-gptq is not installed or failed to import. Please ensure it is available in your environment.")

    os.makedirs(out_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
    )

    # Load base model on CPU by default (AutoGPTQ does this implicitly)
    model = AutoGPTQForCausalLM.from_pretrained(
        base_model_dir,
        quantize_config,
        device_map='auto',
    )

    # Build calibration set and run quantization
    examples = make_calibration_examples(
        tokenizer,
        calib_csv,
        text_columns_hint=text_columns_hint,
        max_samples=max_calib_samples,
        max_length=max_length,
    )
    if len(examples) == 0:
        raise ValueError("No calibration examples were constructed.")

    model.quantize(examples)

    model.save_quantized(out_dir, use_safetensors=use_safetensors)
    tokenizer.save_pretrained(out_dir)


def main():
    ap = argparse.ArgumentParser(description='Calibration-aware GPTQ quantization for a CausalLM using a held-out CSV')
    ap.add_argument('--model-dir', required=True, help='Path or HF id of the base (float) CausalLM to quantize')
    ap.add_argument('--out-dir', required=True, help='Directory to save the quantized model')
    ap.add_argument('--calib-csv', required=True, help='CSV with columns prompt,response_a,response_b (or a single text column)')
    ap.add_argument('--text-columns-hint', default='', help='Optional comma-separated list of text columns to use from the CSV')
    ap.add_argument('--bits', type=int, default=4)
    ap.add_argument('--group-size', type=int, default=128)
    ap.add_argument('--desc-act', action='store_true')
    ap.add_argument('--max-calib-samples', type=int, default=1024)
    ap.add_argument('--max-length', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-safetensors', action='store_true', help='Disable safetensors when saving')
    args = ap.parse_args()

    quantize_with_calibration(
        base_model_dir=args.model_dir,
        out_dir=args.out_dir,
        calib_csv=args.calib_csv,
        text_columns_hint=args.text_columns_hint,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        max_calib_samples=args.max_calib_samples,
        max_length=args.max_length,
        seed=args.seed,
        use_safetensors=not args.no_safetensors,
    )


if __name__ == '__main__':
    main()
