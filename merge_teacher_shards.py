#!/usr/bin/env python
"""Merge per-shard teacher inference outputs produced by step4_infer_teacher_logits.sh.

Assumptions:
  - Sharded files are suffixed: _sh{ID}-of-{NUM}.pt or .parquet/.csv
  - You used INFER_TRAIN_SHARDS=N and INFER_TRAIN_SHARD_ID in 0..N-1 when running step4.
  - Each shard wrote train_probs/logprobs (and optionally val_* if INFER_INCLUDE_VAL=1).

What this script does per fold & model:
  1. Collect all train_probs shard tensors and concatenate (row order preserved from range or mod strategy).
  2. Same for train_logprobs if present.
  3. Optionally repeat for val_*.
  4. Collect OOF tables, concatenate, and write a consolidated table.
  5. Recompute per-row ensemble across models if requested.

Usage examples:
  python merge_teacher_shards.py --fold 0 --models llama --shards 5
  python merge_teacher_shards.py --fold 1 --models llama,qwen --shards 8 --dir model_save/teacher_logits
  python merge_teacher_shards.py --fold 2 --models llama --shards 5 --recompute-ensemble

Outputs:
  model_save/teacher_logits/{model}_fold_{fold}_train_probs_merged.pt
  model_save/teacher_logits/{model}_fold_{fold}_train_logprobs_merged.pt (if available)
  model_save/teacher_logits/{model}_fold_{fold}_val_probs_merged.pt (if available)
  model_save/teacher_logits/{model}_fold_{fold}_val_logprobs_merged.pt (if available)
  model_save/teacher_logits/oof_probs_fold{fold}_merged.parquet (or .csv)
  model_save/teacher_logits/ensemble_oof_probs_fold{fold}_merged.pt (if recompute ensemble)

Notes:
  - Concatenation order depends on file name sorting; for range strategy shards naturally follow row order.
  - For mod strategy, row ordering is interleaved; you may want to sort by original index column ('orig_idx') in OOF table.
  - Ensemble recomputation averages model probabilities at identical (fold,row_id) pairs.
"""
from __future__ import annotations
import argparse, os, re, glob, sys
import torch
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, help='Fold ID to merge.')
    ap.add_argument('--models', type=str, default='llama', help='Comma list of model names used in inference.')
    ap.add_argument('--dir', type=str, default='model_save/teacher_logits', help='Directory containing shard outputs.')
    ap.add_argument('--shards', type=int, required=True, help='Total number of shards expected.')
    ap.add_argument('--recompute-ensemble', action='store_true', help='Recompute ensemble probabilities across models for this fold.')
    ap.add_argument('--prefer-parquet', action='store_true', help='Write merged OOF table as parquet if possible.')
    return ap.parse_args()

def find_shard_files(base_dir: str, model: str, fold: int, kind: str):
    # kind examples: train_probs, train_logprobs, val_probs, val_logprobs
    pattern = os.path.join(base_dir, f"{model}_fold_{fold}_{kind}_sh*-of-*.pt")
    files = sorted(glob.glob(pattern))
    return files

def load_and_concat_tensors(files):
    if not files:
        return None
    tensors = []
    for f in files:
        try:
            t = torch.load(f)
            if not torch.is_tensor(t):
                print(f"[Warn] {f} is not a tensor; skipping.")
                continue
            tensors.append(t)
        except Exception as e:
            print(f"[Warn] Failed loading {f}: {e}")
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)

def merge_tables(base_dir: str, fold: int):
    # Accept both parquet and csv
    pq_pattern = os.path.join(base_dir, f"oof_probs_sh*-of-*.parquet")
    csv_pattern = os.path.join(base_dir, f"oof_probs_sh*-of-*.csv")
    files = sorted(glob.glob(pq_pattern) + glob.glob(csv_pattern))
    tables = []
    for f in files:
        try:
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
            # Filter fold rows
            df_fold = df[df['fold'] == fold]
            tables.append(df_fold)
        except Exception as e:
            print(f"[Warn] Failed reading {f}: {e}")
    if not tables:
        return None
    merged = pd.concat(tables, ignore_index=True)
    return merged

def recompute_ensemble(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    # Use train split if val missing; else val preference
    basis = df[df['split']=='val'] if (df['split']=='val').any() else df[df['split']=='train']
    grp = basis.groupby(['fold','row_id'])[['pA','pB','pTie']].mean().reset_index()
    tens = torch.tensor(grp[['pA','pB','pTie']].values, dtype=torch.float32)
    return tens

def main():
    args = parse_args()
    os.makedirs(args.dir, exist_ok=True)
    models = [m.strip() for m in args.models.replace(',', ' ').split() if m.strip()]
    print(f"[Merge] fold={args.fold} models={models} shards={args.shards} dir={args.dir}")

    # Merge tensors per model
    for m in models:
        for kind in ['train_probs','train_logprobs','val_probs','val_logprobs']:
            files = find_shard_files(args.dir, m, args.fold, kind)
            if not files:
                continue
            merged = load_and_concat_tensors(files)
            if merged is None:
                continue
            out_path = os.path.join(args.dir, f"{m}_fold_{args.fold}_{kind}_merged.pt")
            torch.save(merged, out_path)
            print(f"[Merge] Saved {kind} merged tensor -> {out_path} shape={tuple(merged.shape)} (from {len(files)} shards)")

    # Merge OOF tables
    merged_table = merge_tables(args.dir, args.fold)
    if merged_table is not None:
        # Optionally sort by orig_idx if present for deterministic order
        if 'orig_idx' in merged_table.columns:
            merged_table = merged_table.sort_values('orig_idx').reset_index(drop=True)
        table_out = os.path.join(args.dir, f"oof_probs_fold{args.fold}_merged.{'parquet' if args.prefer_parquet else 'csv'}")
        if args.prefer_parquet:
            merged_table.to_parquet(table_out, index=False)
        else:
            merged_table.to_csv(table_out, index=False)
        print(f"[Merge] OOF table merged -> {table_out} rows={len(merged_table)}")
    else:
        print("[Merge] No OOF shard tables found.")

    if args.recompute_ensemble and merged_table is not None:
        ens = recompute_ensemble(merged_table)
        if ens is not None:
            ens_out = os.path.join(args.dir, f"ensemble_oof_probs_fold{args.fold}_merged.pt")
            torch.save(ens, ens_out)
            print(f"[Merge] Recomputed ensemble -> {ens_out} shape={tuple(ens.shape)}")
        else:
            print("[Merge] Ensemble recompute skipped (no basis rows).")

if __name__ == '__main__':
    main()
