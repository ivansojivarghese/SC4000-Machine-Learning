# student_infer_hf.py
"""
Run inference with the trained small student classifier (DistilBERT) on test.csv
- Loads model from ./model_save/student_distilbert
- Reads ./data/test.csv (id, prompt, response_a, response_b)
- Writes probabilities submission to ./sub/student_submission.csv
"""

import os
import numpy as np
import pandas as pd
from typing import Dict

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def infer_student(
    model_dir: str = './model_save/student_distilbert',
    test_csv: str = './data/test.csv',
    submission_path: str = './sub/student_submission.csv',
    batch_size: int = 16,
    max_length: int = 512,
):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    df = pd.read_csv(test_csv)
    needed = ['id', 'prompt', 'response_a', 'response_b']
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column in test.csv: {c}")

    texts = (df.apply(build_input_text, axis=1)).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load optional calibration
    calib_path = os.path.join(model_dir, 'calibration.json')
    temperature = 1.0
    if os.path.exists(calib_path):
        try:
            import json
            with open(calib_path, 'r') as f:
                temperature = float(json.load(f).get('temperature', 1.0))
        except Exception:
            temperature = 1.0

    probs_list = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits  # [B,3]
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)

    probs = np.vstack(probs_list)
    # safety normalization
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


if __name__ == '__main__':
    p = infer_student()
    print({"saved": p})
