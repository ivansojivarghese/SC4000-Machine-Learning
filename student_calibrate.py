# student_calibrate.py
"""
Temperature scaling for the trained student classifier.
- Splits train.csv into train/val (same as student_train_hf), loads the trained model,
  computes logits on validation, fits a temperature to minimize NLL (log loss),
  saves calibration to ./model_save/student_distilbert/calibration.json.
"""

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_MAP = {"model_a": 0, "model_b": 1, "tie": 2, "tie (both bad)": 2}


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def load_dataset_for_cal(train_csv: str, max_samples: int = 4000) -> Dataset:
    df = pd.read_csv(train_csv)
    text_cols = ['prompt', 'response_a', 'response_b']
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in train.csv: {c}")

    if 'winner' in df.columns:
        labels = df['winner'].map(LABEL_MAP)
    else:
        candidates = [
            ['winner_model_a', 'winner_model_b', 'winner_tie'],
            ['winner_model_a_prob', 'winner_model_b_prob', 'winner_tie_prob'],
        ]
        probs = None
        for trio in candidates:
            if all(c in df.columns for c in trio):
                probs = df[trio].astype(float)
                break
        if probs is None:
            raise ValueError("Could not find labels: expected 'winner' or probabilities columns.")
        labels = probs.values.argmax(axis=1)

    df = df.dropna(subset=text_cols).copy()
    df['text'] = df.apply(build_input_text, axis=1)
    df['label'] = labels

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    return Dataset.from_pandas(df[['text', 'label']])


def compute_nll(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    labels = labels.astype(int)
    scaled = logits / float(T)
    m = scaled.max(axis=1, keepdims=True)
    log_probs = scaled - (m + np.log(np.sum(np.exp(scaled - m), axis=1, keepdims=True)))
    nll = -np.mean([log_probs[i, labels[i]] for i in range(len(labels))])
    return float(nll)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    # Fit temperature by minimizing NLL with simple line search on log T
    # Works well enough and is stable.
    labels = labels.astype(int)
    best_T = 1.0
    best_loss = 1e9
    grid = np.linspace(-2.0, 2.0, 81)  # logT in [-2, 2] => T in [0.135, 7.39]
    for logT in grid:
        T = float(np.exp(logT))
        # stable NLL via helper
        nll = compute_nll(logits, labels, T)
        if nll < best_loss:
            best_loss = nll
            best_T = T
    return best_T


def calibrate_student(
    model_dir: str = './model_save/student_distilbert',
    train_csv: str = './data/train.csv',
    output_json: str = './model_save/student_distilbert/calibration.json',
    batch_size: int = 16,
    max_length: int = 512,
) -> Dict:
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    dataset = load_dataset_for_cal(train_csv)
    # First split: create a validation pool (20% of data)
    split1 = dataset.train_test_split(test_size=0.2, seed=42)
    val_pool = split1['test']
    # Second split: split validation pool into calibration and holdout (50/50)
    split2 = val_pool.train_test_split(test_size=0.5, seed=123)
    cal_ds = split2['train']
    holdout_ds = split2['test']

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def compute_logits(ds):
        texts = ds['text']
        labels = np.array(ds['label'])
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(texts[i:i+batch_size], truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits.cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits, labels

    cal_logits, cal_labels = compute_logits(cal_ds)
    hold_logits, hold_labels = compute_logits(holdout_ds)

    # Fit temperature on calibration split only
    T = fit_temperature(cal_logits, cal_labels)

    # Metrics on calibration split
    cal_before = compute_nll(cal_logits, cal_labels, T=1.0)
    cal_after = compute_nll(cal_logits, cal_labels, T=T)

    # Metrics on holdout split (reporting set)
    hold_before = compute_nll(hold_logits, hold_labels, T=1.0)
    hold_after = compute_nll(hold_logits, hold_labels, T=T)

    payload = {
        'temperature': float(T),
        'cal_size': int(len(cal_labels)),
        'holdout_size': int(len(hold_labels)),
        'nll_cal_before': float(cal_before),
        'nll_cal_after': float(cal_after),
        'nll_cal_improvement': float(cal_before - cal_after),
        'nll_holdout_before': float(hold_before),
        'nll_holdout_after': float(hold_after),
        'nll_holdout_improvement': float(hold_before - hold_after),
    }

    with open(output_json, 'w') as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == '__main__':
    info = calibrate_student()
    print(info)
