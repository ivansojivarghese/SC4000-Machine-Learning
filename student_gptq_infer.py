"""
Inference for GPTQ-quantized CausalLM exports with a classification head adapter.

This script loads a quantized CausalLM (e.g., LLaMA/Qwen) using AutoGPTQ and applies a
lightweight classification head on top of the model's hidden states to produce 3-way
probabilities (winner_model_a, winner_model_b, winner_tie) for the Kaggle submission.

Two ways to provide the classification head:
1) Adapter weights path (preferred): a small torch .pt containing a linear layer weights/bias.
2) Derive from an existing sequence-classifier directory via --classifier-from-dir (extract the head).

Notes:
- We use the CLS/pooled representation if available, else mean-pool last_hidden_state.
- Supports TTA by averaging probabilities across multiple max_length values.
- Applies temperature calibration if calibration.json exists in the classifier dir or provided path.
"""

import argparse
import json
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer

try:
    from auto_gptq import AutoGPTQForCausalLM
except Exception:
    AutoGPTQForCausalLM = None


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def load_temperature(calibration_json_path: Optional[str]) -> float:
    if not calibration_json_path:
        return 1.0
    if not os.path.exists(calibration_json_path):
        return 1.0
    try:
        with open(calibration_json_path, 'r') as f:
            data = json.load(f)
        return float(data.get('temperature', 1.0))
    except Exception:
        return 1.0


def extract_classifier_from_dir(classifier_dir: str, num_labels: int = 3) -> nn.Module:
    """
    Attempt to extract a classifier head from a fine-tuned sequence classification directory by
    reading a saved head (if exported) or initializing a new linear layer with the same hidden size.
    We search for hidden_size in config.json and expect classifier weights in classifier_head.pt.
    """
    cfg_path = os.path.join(classifier_dir, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in {classifier_dir}")
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    hidden_size = cfg.get('hidden_size') or cfg.get('hidden_sizes', [None])[0]
    if hidden_size is None:
        # fallback to common keys
        hidden_size = cfg.get('d_model') or cfg.get('hidden_size', 1024)
    head = nn.Linear(int(hidden_size), num_labels)

    # If a dumped head exists, load it
    head_path = os.path.join(classifier_dir, 'classifier_head.pt')
    if os.path.exists(head_path):
        state = torch.load(head_path, map_location='cpu')
        try:
            head.load_state_dict(state)
        except Exception:
            pass
    return head


def load_classifier_head(head_path: Optional[str], classifier_from_dir: Optional[str], hidden_size: int, num_labels: int = 3) -> nn.Module:
    if head_path and os.path.exists(head_path):
        state = torch.load(head_path, map_location='cpu')
        head = nn.Linear(hidden_size, num_labels)
        head.load_state_dict(state)
        return head
    if classifier_from_dir:
        return extract_classifier_from_dir(classifier_from_dir, num_labels=num_labels)
    # default randomly initialized head
    return nn.Linear(hidden_size, num_labels)


def infer(
    model_dir: str,
    test_csv: str,
    submission_path: str,
    head_path: Optional[str] = None,
    classifier_from_dir: Optional[str] = None,
    tta_lengths: Optional[List[int]] = None,
    batch_size: int = 8,
    max_length: int = 512,
    device: Optional[str] = None,
    temperature_json: Optional[str] = None,
):
    if AutoGPTQForCausalLM is None:
        raise RuntimeError("auto-gptq is not installed. Please install auto-gptq to use this script.")

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    df = pd.read_csv(test_csv)
    for c in ['id', 'prompt', 'response_a', 'response_b']:
        if c not in df.columns:
            raise ValueError(f"Missing column in test.csv: {c}")
    texts = (df.apply(build_input_text, axis=1)).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = AutoGPTQForCausalLM.from_quantized(
        model_dir,
        device_map='auto' if dev.type == 'cuda' else None,
    )
    model.eval()

    # Hidden size discovery via config if available
    hidden_size = getattr(model.config, 'hidden_size', None) or getattr(model.config, 'hidden_sizes', [None])[0]
    if hidden_size is None:
        hidden_size = getattr(model.config, 'd_model', 1024)

    head = load_classifier_head(head_path, classifier_from_dir, hidden_size, num_labels=3).to(dev)
    head.eval()

    # Temperature from provided path or fallback to classifier dir
    temp_path = temperature_json
    if not temp_path and classifier_from_dir:
        temp_path = os.path.join(classifier_from_dir, 'calibration.json')
    temperature = load_temperature(temp_path)

    lengths = tta_lengths if tta_lengths else [max_length]
    probs_accum = None

    with torch.no_grad():
        for L in lengths:
            probs_list = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = tokenizer(batch_texts, truncation=True, padding='max_length', max_length=L, return_tensors='pt')
                enc = {k: v.to(dev) for k, v in enc.items()}

                # forward to get hidden states
                try:
                    outputs = model(**enc, output_hidden_states=True)
                except TypeError:
                    # Some wrappers may require passing output_hidden_states via config; fallback to common submodules
                    if hasattr(model, 'transformer'):
                        outputs = model.transformer(**enc, output_hidden_states=True)
                    else:
                        outputs = model.model(**enc, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]  # [B, T, H]
                # Try CLS token (first token) if present otherwise mean pool
                if tokenizer.cls_token_id is not None:
                    pooled = last_hidden[:, 0]
                else:
                    pooled = (last_hidden * enc['attention_mask'].unsqueeze(-1)).sum(dim=1) / enc['attention_mask'].sum(dim=1, keepdim=True)

                logits = head(pooled)
                if temperature != 1.0:
                    logits = logits / temperature
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                probs_list.append(probs)

            probs = np.vstack(probs_list)
            probs_accum = probs if probs_accum is None else (probs_accum + probs)

    probs = probs_accum / len(lengths)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({
        'id': df['id'],
        'winner_model_a': probs[:, 0],
        'winner_model_b': probs[:, 1],
        'winner_tie': probs[:, 2],
    })
    sub.to_csv(submission_path, index=False)
    return submission_path


def main():
    ap = argparse.ArgumentParser(description='Inference with GPTQ-quantized CausalLM using a classification head adapter')
    ap.add_argument('--model-dir', required=True, help='Path to GPTQ quantized model directory')
    ap.add_argument('--test-csv', default='./data/test.csv')
    ap.add_argument('--out', default='./sub/student_submission.csv')
    ap.add_argument('--head-path', default=None, help='Path to classifier_head.pt (state_dict for nn.Linear)')
    ap.add_argument('--classifier-from-dir', default=None, help='Directory of a seq-classification model to extract head and calibration.json')
    ap.add_argument('--tta-lengths', default='', help='Comma-separated list, e.g. 512,1024')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=512)
    ap.add_argument('--device', default=None)
    ap.add_argument('--temperature-json', default=None, help='Optional explicit path to calibration.json')
    args = ap.parse_args()

    tta = [int(x) for x in args.tta_lengths.split(',') if x.strip()] if args.tta_lengths else None

    infer(
        model_dir=args.model_dir,
        test_csv=args.test_csv,
        submission_path=args.out,
        head_path=args.head_path,
        classifier_from_dir=args.classifier_from_dir,
        tta_lengths=tta,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        temperature_json=args.temperature_json,
    )


if __name__ == '__main__':
    main()
