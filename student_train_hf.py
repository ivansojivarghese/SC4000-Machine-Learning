# student_train_hf.py
"""
Minimal real training loop for a small student classifier using Hugging Face Trainer.
- Model: distilbert-base-uncased (sequence classification, 3 labels)
- Data: ./data/train.csv with columns prompt,response_a,response_b,winner
- Output: ./model_save/student_distilbert (adapter-free; full finetune on CPU/GPU)
"""

import os
import pandas as pd
from typing import Dict

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers import EarlyStoppingCallback
import numpy as np

LABEL_MAP = {"model_a": 0, "model_b": 1, "tie": 2, "tie (both bad)": 2}


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def load_dataset(train_csv: str, max_samples: int | None = 2000) -> Dataset:
    df = pd.read_csv(train_csv)
    text_cols = ['prompt', 'response_a', 'response_b']
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in train.csv: {c}")

    # Determine labels
    label_series = None
    if 'winner' in df.columns:
        label_series = df['winner'].map(LABEL_MAP)
    else:
        # Try probability/one-hot columns
        prob_cols = None
        candidates = [
            ['winner_model_a', 'winner_model_b', 'winner_tie'],
            ['winner_model_a_prob', 'winner_model_b_prob', 'winner_tie_prob'],
        ]
        for trio in candidates:
            if all(c in df.columns for c in trio):
                prob_cols = trio
                break
        if prob_cols is not None:
            probs = df[prob_cols].astype(float)
            label_series = probs.values.argmax(axis=1)
        else:
            raise ValueError("Could not find labels: expected 'winner' or probability columns like winner_model_a/b/tie")

    needed = text_cols
    df = df.dropna(subset=needed).copy()
    df['text'] = df.apply(build_input_text, axis=1)
    df['label'] = label_series
    df = df.dropna(subset=['label'])

    # Optional sub-sampling for quick local run
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    return Dataset.from_pandas(df[['text', 'label']])


def tokenize_function(examples: Dict, tokenizer, max_length: int = 512) -> Dict:
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )


def train_student(
    train_csv: str = './data/train.csv',
    output_dir: str = './model_save/student_distilbert',
    max_samples: int | None = 2000,
    num_epochs: int = 1,
    model_name: str = 'distilbert-base-uncased',
    label_smoothing: float = 0.05,
    early_stopping_patience: int = 2,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(train_csv, max_samples=max_samples)

    # Simple train/val split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    tokenized_train = dataset['train'].map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_val = dataset['test'].map(lambda x: tokenize_function(x, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        logging_steps=20,
        label_smoothing_factor=label_smoothing,
        fp16=False,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        return {'accuracy': acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return metrics


if __name__ == '__main__':
    m = train_student()
    print({"eval": m})
