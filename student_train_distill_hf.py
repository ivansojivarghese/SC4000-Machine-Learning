"""
Student training with knowledge distillation (KL + CE) using Hugging Face Trainer.
- Supports multiple teacher logits .npy files (averaged as probabilities at T_soft)
- Loss: alpha * KL(student||teacher) at T_soft + (1-alpha) * CE(labels)
- Uses Kaggle-style log loss for evaluation metrics
"""

from typing import Dict, List, Optional
import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainerCallback,
)
from scipy.special import softmax
from sklearn.metrics import log_loss
import time

LABEL_MAP = {"model_a": 0, "model_b": 1, "tie": 2, "tie (both bad)": 2}


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def _label_from_df(df: pd.DataFrame) -> np.ndarray:
    if 'winner' in df.columns:
        return df['winner'].map(LABEL_MAP).values
    trio = None
    candidates = [
        ['winner_model_a', 'winner_model_b', 'winner_tie'],
        ['winner_model_a_prob', 'winner_model_b_prob', 'winner_tie_prob'],
    ]
    for cand in candidates:
        if all(c in df.columns for c in cand):
            trio = cand
            break
    if trio is None:
        raise ValueError("Could not find labels: expected 'winner' or probability columns")
    return df[trio].astype(float).values.argmax(axis=1)


def _load_base_csv_with_idx(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    text_cols = ['prompt', 'response_a', 'response_b']
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in train.csv: {c}")
    labels = _label_from_df(df)
    df = df.dropna(subset=text_cols).copy()
    df['label'] = labels
    df = df.dropna(subset=['label'])
    # Base index aligned with post-processed order (for teacher logits alignment)
    df['base_idx'] = np.arange(len(df), dtype=np.int64)
    return df[['prompt', 'response_a', 'response_b', 'label', 'base_idx']]


def _load_extra_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize schema similar to baseline
    def _norm(df_in: pd.DataFrame) -> pd.DataFrame:
        df_in = df_in.copy()
        alt_a = ['response_a', 'answer_a', 'assistant_a', 'chosen', 'output_1', 'completion_a']
        alt_b = ['response_b', 'answer_b', 'assistant_b', 'rejected', 'output_2', 'completion_b']
        a_col = next((c for c in alt_a if c in df_in.columns), None)
        b_col = next((c for c in alt_b if c in df_in.columns), None)
        prompt_col = 'prompt' if 'prompt' in df_in.columns else next((c for c in ['question','instruction','query','messages','prompt_text','prompt_a'] if c in df_in.columns), None)
        cols = {}
        if prompt_col: cols['prompt'] = df_in[prompt_col]
        if a_col: cols['response_a'] = df_in[a_col]
        if b_col: cols['response_b'] = df_in[b_col]
        return pd.DataFrame(cols) if cols else df_in
    df = _norm(df)
    text_cols = ['prompt', 'response_a', 'response_b']
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in extra CSV: {path} -> missing {c}; available: {list(df.columns)}")
    try:
        labels = _label_from_df(df)
    except Exception:
        # No labels in extra; assume response_a is preferred if chosen/rejected source
        labels = np.full(len(df), LABEL_MAP['model_a'])
    df = df.dropna(subset=text_cols).copy()
    df['label'] = labels
    df = df.dropna(subset=['label'])
    df['base_idx'] = -1  # no teacher logits available
    return df[['prompt', 'response_a', 'response_b', 'label', 'base_idx']]


def load_dataset(train_csv: str, max_samples: Optional[int] = None, extra_csvs: Optional[List[str]] = None, shuffle_ab: bool = False, dedup_by_prompt: bool = False) -> Dataset:
    base_df = _load_base_csv_with_idx(train_csv)
    dfs = [base_df]
    if extra_csvs:
        for p in extra_csvs:
            if p and os.path.exists(p):
                dfs.append(_load_extra_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    # Deduplicate
    if dedup_by_prompt:
        df = df.drop_duplicates(subset=['prompt']).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=['prompt', 'response_a', 'response_b']).reset_index(drop=True)

    # Optional A/B shuffle only for extra rows (to avoid needing to swap teacher probs)
    if shuffle_ab:
        mask_extra = df['base_idx'] < 0
        extra_idx = df[mask_extra].sample(frac=0.5, random_state=42).index
        a = df.loc[extra_idx, 'response_a'].copy()
        b = df.loc[extra_idx, 'response_b'].copy()
        df.loc[extra_idx, 'response_a'] = b
        df.loc[extra_idx, 'response_b'] = a
        lab = df.loc[extra_idx, 'label'].astype(int).copy()
        flipped = lab.map({0: 1, 1: 0}).fillna(2).astype(int)
        df.loc[extra_idx, 'label'] = flipped

    df['text'] = df.apply(build_input_text, axis=1)
    # idx used for teacher alignment (-1 means no teacher)
    df['idx'] = df['base_idx'].astype(int)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).sort_index()

    return Dataset.from_pandas(df[['text', 'label', 'idx']])


def tokenize_function(examples: Dict, tokenizer, max_length: int = 512) -> Dict:
    # Dynamic padding via DataCollatorWithPadding (no padding here)
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
    )


class DistillTrainer(Trainer):
    def __init__(self, *args, teacher_probs: np.ndarray, alpha: float = 0.7, T_soft: float = 3.0, label_smoothing: float = 0.0, mse_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_probs = teacher_probs  # shape [N,3] aligned to dataset idx
        self.alpha = float(alpha)
        self.T_soft = float(T_soft)
        self.label_smoothing = float(label_smoothing)
        self.mse_weight = float(mse_weight)
        self.kl = KLDivLoss(reduction='batchmean')

    def compute_loss(self, model, inputs, return_outputs=False):
        # Pull out indices for teacher alignment
        idx = inputs.pop('idx')
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.logits  # [B,3]

        # CE loss with optional label smoothing
        ce = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        # KL distillation at T_soft (mask rows without teacher)
        student_logp_T = F.log_softmax(logits / self.T_soft, dim=-1)
        with torch.no_grad():
            idx_np = idx.detach().cpu().numpy()
        valid_mask = (idx >= 0) & (idx < len(self.teacher_probs))
        if valid_mask.any():
            # Build teacher probs batch with only valid rows
            valid_idx = idx[valid_mask].detach().cpu().numpy()
            with torch.no_grad():
                tp_valid = torch.from_numpy(self.teacher_probs[valid_idx]).to(logits.device)
            slp_valid = student_logp_T[valid_mask]
            # per-row KL then mean over full batch size to keep scale stable
            kl_elem = F.kl_div(slp_valid, tp_valid, reduction='none').sum(dim=-1)  # [n_valid]
            kl = kl_elem.sum() / logits.size(0)
        else:
            kl = torch.tensor(0.0, device=logits.device)
        kl = kl * (self.T_soft ** 2)

        # Optional third loss: MSE between student probs@T and teacher probs
        if self.mse_weight > 0 and valid_mask.any():
            with torch.no_grad():
                tp_full = torch.from_numpy(self.teacher_probs[idx_np]).to(logits.device)
            sp_T = F.softmax(logits / self.T_soft, dim=-1)
            mse = F.mse_loss(sp_T[valid_mask], tp_full[valid_mask], reduction='mean')
        else:
            mse = torch.tensor(0.0, device=logits.device)

        loss = self.alpha * kl + (1.0 - self.alpha) * ce + self.mse_weight * mse
        return (loss, outputs) if return_outputs else loss


def average_teacher_probs(teacher_logits_files: List[str], num_classes: int = 3, T_soft: float = 3.0) -> np.ndarray:
    probs_list = []
    for path in teacher_logits_files:
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != num_classes:
            raise ValueError(f"Teacher logits file has wrong shape: {path} -> {arr.shape}")
        p = softmax(arr / T_soft, axis=1)
        probs_list.append(p)
    probs = np.mean(probs_list, axis=0) if len(probs_list) > 1 else probs_list[0]
    # safety normalize
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits, axis=1)
    ll = float(log_loss(labels, probs, labels=[0, 1, 2]))
    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean())
    return {'log_loss': ll, 'accuracy': acc}


def _parse_temp_schedule(spec: str, num_epochs: int, T_init: float) -> List[float]:
    """Parse a simple schedule string like 'linear:5,2' -> linear from T_init to 2 over 5 epochs.
    Returns list length num_epochs of per-epoch temperatures. Unknown spec returns constant list.
    """
    temps = [T_init for _ in range(int(num_epochs))]
    if not spec:
        return temps
    try:
        mode, rest = spec.split(':', 1)
        parts = rest.split(',')
        if mode == 'linear' and len(parts) == 2:
            span = int(parts[0])
            T_end = float(parts[1])
            span = max(1, min(span, int(num_epochs)))
            for e in range(span):
                frac = e / max(1, span - 1)
                temps[e] = (1 - frac) * T_init + frac * T_end
            for e in range(span, int(num_epochs)):
                temps[e] = T_end
    except Exception:
        return temps
    return temps


def train_student_distill(
    train_csv: str = './data/train.csv',
    output_dir: str = './model_save/student_distilbert',
    teacher_logits: Optional[List[str]] = None,
    model_name: str = 'distilbert-base-uncased',
    max_samples: Optional[int] = None,
    num_epochs: int = 1,
    alpha: float = 0.7,
    T_soft: float = 3.0,
    temp_schedule: str = '',
    label_smoothing: float = 0.0,
    mse_weight: float = 0.0,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.06,
    fp16: bool = False,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int = 2,
    max_length: int = 512,
    extra_csvs: Optional[List[str]] = None,
    shuffle_ab: bool = False,
    dedup_by_prompt: bool = False,
    use_fast_tokenizer: bool = True,
    dataloader_num_workers: int = 0,
    num_folds: int = 1,
    fold_idx: int = 0,
):
    if not teacher_logits:
        raise ValueError('teacher_logits list is required for distillation')
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=bool(use_fast_tokenizer))
    # Capability-based precision handling: V100 doesn't support bf16
    if bf16:
        bf16_supported = False
        try:
            bf16_supported = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
        except Exception:
            bf16_supported = False
        if not bf16_supported:
            bf16 = False
            fp16 = True
    logger.info(f"Precision config -> bf16={bf16}, fp16={fp16}")
    ds = load_dataset(train_csv, max_samples=max_samples, extra_csvs=extra_csvs, shuffle_ab=shuffle_ab, dedup_by_prompt=dedup_by_prompt)
    # Fold-aware split: deterministic K-fold over dataset order
    if num_folds and num_folds > 1:
        n = len(ds)
        fold_idx = int(fold_idx) % int(num_folds)
        fold_size = n // int(num_folds)
        start = fold_idx * fold_size
        end = n if fold_idx == num_folds - 1 else (start + fold_size)
        indices = np.arange(n)
        val_mask = (indices >= start) & (indices < end)
        train_mask = ~val_mask
        train_ds = ds.select(np.where(train_mask)[0].tolist())
        val_ds = ds.select(np.where(val_mask)[0].tolist())
        logger.info(f"[Distill] Using fold {fold_idx}/{num_folds}: train={len(train_ds)} eval={len(val_ds)}")
    else:
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds, val_ds = split['train'], split['test']

    # Multiprocess tokenization + dynamic padding
    _num_proc = max(1, min(8, int(dataloader_num_workers) or 1))
    train_tok = train_ds.map(lambda x: tokenize_function(x, tokenizer, max_length=max_length), batched=True, num_proc=_num_proc)
    val_tok = val_ds.map(lambda x: tokenize_function(x, tokenizer, max_length=max_length), batched=True, num_proc=_num_proc)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            try:
                model.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                model.gradient_checkpointing_enable()

    # Load and average teacher probs
    teacher_probs = average_teacher_probs(teacher_logits, num_classes=3, T_soft=T_soft)
    if len(teacher_probs) < len(ds):
        raise ValueError('Teacher logits length is smaller than dataset; ensure alignment/order match')

    # Training args
    pin_mem = bool(torch.cuda.is_available() or (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()))
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='log_loss',
        greater_is_better=False,
        logging_steps=20,
        label_smoothing_factor=0.0,  # handled in CE above
        fp16=fp16,
        bf16=bf16,
        warmup_ratio=warmup_ratio,
        dataloader_pin_memory=pin_mem,
        dataloader_num_workers=int(dataloader_num_workers),
        group_by_length=True,
        report_to='none',
    )
    _pad_to_multiple = 8 if (fp16 or bf16) else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=_pad_to_multiple)

    # Throughput logging
    class ThroughputLoggerCallback(TrainerCallback):
        def __init__(self, ds):
            self._ds = ds
            self._epoch_start = None
            # estimate total tokens from attention_mask if present
            self._examples = len(ds)
            toks = 0
            try:
                for ex in ds:
                    am = ex.get('attention_mask')
                    if am is not None:
                        toks += int(sum(am))
            except Exception:
                toks = self._examples * max_length
            self._tokens = toks
            if self._examples > 0:
                avg_len = toks / self._examples
                logger.info(f"[Distill] Train set -> examples={self._examples}, total_tokens={self._tokens}, avg_len={avg_len:.1f}")

        def on_epoch_begin(self, args, state, control, **kwargs):
            self._epoch_start = time.time()

        def on_epoch_end(self, args, state, control, **kwargs):
            if self._epoch_start is None:
                return
            dt = max(1e-6, time.time() - self._epoch_start)
            ex_per_sec = self._examples / dt
            tok_per_sec = self._tokens / dt
            ep = state.epoch if state.epoch is not None else 0
            logger.info(f"[Distill] Epoch {int(ep)} finished in {dt:.2f}s | examples/sec: {ex_per_sec:.2f} | tokens/sec: {tok_per_sec:.2f}")

    # Optional per-epoch temperature schedule (adjusts T_soft used in KL and MSE)
    temps = _parse_temp_schedule(temp_schedule, num_epochs, T_soft)

    trainer = DistillTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            ThroughputLoggerCallback(train_tok),
        ],
        teacher_probs=teacher_probs,
        alpha=alpha,
        T_soft=T_soft,
        label_smoothing=label_smoothing,
        mse_weight=mse_weight,
    )

    # Hook to update trainer temperature at each epoch begin
    class _TempUpdater(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            ep = int(state.epoch or 0)
            if 0 <= ep < len(temps):
                new_T = float(temps[ep])
                if getattr(trainer, 'T_soft', None) != new_T:
                    trainer.T_soft = new_T
                    logger.info(f"[Distill] Updated temperature T_soft -> {new_T}")

    trainer.add_callback(_TempUpdater())

    trainer.train()
    metrics = trainer.evaluate()
    # Persist CV metrics for ensembling weights
    try:
        os.makedirs(output_dir, exist_ok=True)
        fold_tag = f"fold_{fold_idx}_of_{num_folds}" if num_folds and num_folds > 1 else "fold_single"
        with open(os.path.join(output_dir, f"cv_metrics_{fold_tag}.json"), "w") as f:
            import json
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write CV metrics: {e}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return metrics


if __name__ == '__main__':
    m = train_student_distill()
    print({'eval': m})
