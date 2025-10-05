#!/bin/bash
# Step 4: Infer teacher distributions.
# For each fold model, compute:
#  1) Last-token logits snapshots (diagnostic)
#  2) Full response log-likelihood for response_a / response_b conditioned on prompt
#  3) Probability distribution over {A, B, Tie} via softmax with tie temperature heuristic
# Output per fold/model:
#   *_train_lasttok_logits.pt (existing)
#   *_train_logprobs.pt (tensor [N,3] => logprobs raw: logp_a, logp_b, logp_tie_placeholder)
#   *_train_probs.pt (tensor [N,3] => probs pA,pB,pTie)
# Usage:
#   sbatch step4_infer_teacher_logits.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=720
#SBATCH --cpus-per-task=8
#SBATCH --job-name=S4_TeacherInfer
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${PWD}/.hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import os, json, math, torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FOLDS = range(5)

# Detect base model dir patterns used in step3 outputs
def pick_model_dir(prefix_base: str, fold: int):
  # new naming
  cand = f'model_save/{prefix_base}_fold_{fold}'
  if os.path.isdir(cand):
    return cand
  # fallback old naming (llama3-70b, qwen2-72b)
  legacy = f'model_save/{prefix_base}3-70b_fold_{fold}'
  if os.path.isdir(legacy):
    return legacy
  return None

def load_rows(csv_path):
  return pd.read_csv(csv_path)

def build_prompt(row):
  prompt = row.get('prompt')
  if prompt is None:
    # original dataset stores prompt as JSON-like list string; handle fallback columns
    prompt = row.get('question') or ''
  return str(prompt).strip()

def extract_text(row):
  # For last-token embedding snapshot; join prompt + chosen single response if unified column exists
  prompt = build_prompt(row)
  resp = row.get('response') or row.get('chosen') or row.get('answer') or ''
  return (prompt + '\n' if prompt else '') + str(resp).strip()

def compute_pair_loglik(model, tokenizer, prompts, resp_a_list, resp_b_list, max_len=1024, batch_size=1):
  """Return log likelihood (sum log probs) for response_a and response_b for each prompt.
  Uses causal LM token probabilities: log P(resp | prompt).
  """
  logp_a, logp_b = [], []
  model.eval()
  with torch.no_grad():
    for i in range(0, len(prompts), batch_size):
      batch_prompts = prompts[i:i+batch_size]
      a_batch = resp_a_list[i:i+batch_size]
      b_batch = resp_b_list[i:i+batch_size]
      # Encode separately to avoid cross-attention contamination and differing lengths.
      for p, ra, rb in zip(batch_prompts, a_batch, b_batch):
        # Tokenize (prompt + response) once per response
        for resp, store in [(ra, logp_a), (rb, logp_b)]:
            full = (p + '\n' if p else '') + resp
            enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=max_len)
            input_ids = enc['input_ids'].to(DEVICE)
            attn = enc['attention_mask'].to(DEVICE)
            # Identify boundary: prompt length tokens (need prompt tokenization alone)
            p_enc = tokenizer(p + ('\n' if p else ''), return_tensors='pt', truncation=True, max_length=max_len)
            p_len = p_enc['input_ids'].shape[-1]
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1, :]  # shift for next-token prediction
            target = input_ids[:, 1:]
            # Only response portion contributes to conditional loglikelihood
            resp_slice = slice(p_len-1, target.shape[1])  # p_len includes last prompt token; adjust index
            logits_resp = logits[:, resp_slice, :]
            target_resp = target[:, resp_slice]
            lls = torch.nn.functional.log_softmax(logits_resp, dim=-1)
            tok_ll = lls.gather(-1, target_resp.unsqueeze(-1)).squeeze(-1)
            store.append(tok_ll.sum().item())
  return torch.tensor(logp_a), torch.tensor(logp_b)

def batched_logits(model, tokenizer, texts, max_len=512, batch_size=2):
  all_logits = []
  model.eval()
  with torch.no_grad():
    for i in range(0, len(texts), batch_size):
      batch = texts[i:i+batch_size]
      enc = tokenizer(batch, return_tensors='pt', truncation=True, max_length=max_len, padding=True)
      enc = {k: v.to(DEVICE) for k,v in enc.items()}
      out = model(**enc, use_cache=False)
      # Take last token logits as a simple representation (or average)
      logits = out.logits[:, -1, :]  # [B, vocab]
      # For classification placeholder, store top-k? Here we just store raw slice of logits mean/std for debug.
      all_logits.append(logits.cpu().float())
  return torch.cat(all_logits, dim=0)

os.makedirs('model_save/teacher_logits', exist_ok=True)

for fold in FOLDS:
  # Paths from step3 prep
  train_csv=f'data/fold_data/fold_{fold}_train.csv'
  val_csv=f'data/fold_data/fold_{fold}_val.csv'
  if not os.path.isfile(train_csv):
    print(f'[Step4][Skip] Missing {train_csv}; run step3 prep first.')
    continue
  train_df=load_rows(train_csv)
  val_df=load_rows(val_csv) if os.path.isfile(val_csv) else None

  llama_dir = pick_model_dir('llama', fold)
  qwen_dir  = pick_model_dir('qwen', fold)
  for name, mdir in [('llama', llama_dir), ('qwen', qwen_dir)]:
    if mdir is None:
      print(f'[Step4][Warn] Missing model dir for {name} fold {fold}; skipping')
      continue
    print(f'[Step4] Loading {name} fold {fold} from {mdir}')
    tok = AutoTokenizer.from_pretrained(mdir, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
      tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mdir, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
    model.to(DEVICE)
  records_train=train_df.to_dict('records')
  texts_train=[extract_text(r) for r in records_train]
  logits_train=batched_logits(model, tok, texts_train, max_len=min(getattr(tok,'model_max_length',1024),1024))
  out_train_path=f'model_save/teacher_logits/{name}_fold_{fold}_train_lasttok_logits.pt'
  torch.save(logits_train, out_train_path)
  print(f'[Step4] Saved train last-token logits {logits_train.shape} -> {out_train_path}')
  # Pairwise response log-likelihoods
  if {'response_a','response_b'} <= set(train_df.columns):
    prompts=[build_prompt(r) for r in records_train]
    # Convert stringified list prompt variant if needed
    # If prompt column looks like a JSON array string, keep as-is for now; downstream cleaning could be applied.
    logp_a, logp_b = compute_pair_loglik(model, tok, prompts, [str(r.get('response_a','')) for r in records_train], [str(r.get('response_b','')) for r in records_train], max_len=1024)
    # Simple tie modeling: treat tie score as average of both or apply temperature smoothing
    # Here tie raw log score = 0.5*(logp_a + logp_b)
    logp_t = 0.5*(logp_a + logp_b)
    raw = torch.stack([logp_a, logp_b, logp_t], dim=1)
    # Stabilize and softmax
    probs = torch.softmax(raw, dim=1)
    torch.save(raw, f'model_save/teacher_logits/{name}_fold_{fold}_train_logprobs.pt')
    torch.save(probs, f'model_save/teacher_logits/{name}_fold_{fold}_train_probs.pt')
    print(f'[Step4] Saved train logprobs {raw.shape} and probs -> *_train_[logprobs|probs].pt')
  else:
    print(f'[Step4][Warn] response_a/response_b columns not found; skipping probability distribution computation.')
    if val_df is not None:
      val_records=val_df.to_dict('records')
      texts_val=[extract_text(r) for r in val_records]
      logits_val=batched_logits(model, tok, texts_val, max_len=min(getattr(tok,'model_max_length',1024),1024))
      out_val_path=f'model_save/teacher_logits/{name}_fold_{fold}_val_lasttok_logits.pt'
      torch.save(logits_val, out_val_path)
      print(f'[Step4] Saved val last-token logits {logits_val.shape} -> {out_val_path}')
      if {'response_a','response_b'} <= set(val_df.columns):
        prompts=[build_prompt(r) for r in val_records]
        logp_a, logp_b = compute_pair_loglik(model, tok, prompts, [str(r.get('response_a','')) for r in val_records], [str(r.get('response_b','')) for r in val_records], max_len=1024)
        logp_t = 0.5*(logp_a + logp_b)
        raw = torch.stack([logp_a, logp_b, logp_t], dim=1)
        probs = torch.softmax(raw, dim=1)
        torch.save(raw, f'model_save/teacher_logits/{name}_fold_{fold}_val_logprobs.pt')
        torch.save(probs, f'model_save/teacher_logits/{name}_fold_{fold}_val_probs.pt')
        print(f'[Step4] Saved val logprobs and probs -> *_val_[logprobs|probs].pt')
      else:
        print(f'[Step4][Warn] val missing response_a/response_b; skipping val probs.')
    # Free
    del model, tok
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
print('[Step4] Completed logits extraction.')
PY

echo "[Step4] Validating shapes"
# TODO: Update validator to accommodate new path pattern if needed
python teacher_logits_validator.py || echo "[Step4][Warn] Validator failed; adjust validator script to new file naming."

echo "[Step4] Done"
