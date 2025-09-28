# LMSYS Chatbot Arena First Place Solution (Mock + Student Baseline)

This workspace contains:
- A lightweight, runnable mock reproduction of the first-place pipeline phases (post-pretrain, CV, distillation, LoRA merge, GPTQ, TTA).
- A real small-model student training path using Hugging Face (DistilBERT/BERT) that generates a valid probabilities submission and supports calibration.

What you can do locally:
- Run mock train/inference to validate structure and output a submission (`./sub/final_submission.csv`).
- Train a real student model (DistilBERT or BERT), calibrate probabilities with temperature scaling, evaluate on a dedicated holdout, and produce a calibrated submission (`./sub/student_submission.csv`).

## Data layout (not included)
- Train CSV (one of):
  - `./data/train.csv`
  - `./data/lmsys-chatbot-arena/train.csv`
  - Columns required: `prompt,response_a,response_b` and either `winner` or probability columns: `winner_model_a(_prob), winner_model_b(_prob), winner_tie(_prob)`.
- Test CSV: `./data/test.csv`.

## Quick start (Windows PowerShell)
1) Install dependencies
```
pip install -r requirements.txt
pip install torch transformers datasets accelerate
```

2) Mock pipeline (optional)
```
python main.py --mode train
python main.py --mode inference
```
Artifacts: `training_summary.json`, `./sub/final_submission.csv`.

## Real student training (HF)
Train DistilBERT (default):
```
python main.py --mode student-train --student-epochs 1 --student-max-samples 2000
```

Try BERT base with label smoothing and early stopping:
```
python main.py --mode student-train \
  --student-model bert-base-uncased \
  --student-epochs 3 \
  --student-max-samples 6000 \
  --student-label-smoothing 0.1 \
  --student-early-stopping 2
```

Trainer details:
- Label smoothing: configurable via `--student-label-smoothing` (default 0.05)
- Early stopping: `--student-early-stopping` patience (default 2), monitors eval_loss
- Best-checkpoint restore: `load_best_model_at_end=True` using `eval_loss`
- Checkpoints: `save_strategy='epoch'`, `save_total_limit=1`

Artifacts:
- Model + tokenizer: `./model_save/student_distilbert` (path reused for simplicity)
- Calibration: `./model_save/student_distilbert/calibration.json`
- Submission: `./sub/student_submission.csv`

## Probability calibration + holdout eval
Fit temperature on a calibration split and evaluate on a distinct holdout (no leakage):
```
python main.py --mode student-calibrate
python main.py --mode student-eval-holdout
```
The console logs and `calibration.json` include:
- temperature
- nll_cal_before / nll_cal_after / nll_cal_improvement
- nll_holdout_before / nll_holdout_after / nll_holdout_improvement

Notes:
- In this small baseline, calibration may slightly help or hurt on holdout; expect better gains as the student improves (more epochs, stronger backbone, soft-label distillation).

## Calibrated inference
`student_infer_hf.py` automatically loads `calibration.json` (if present) and divides logits by T before softmax.
```
python main.py --mode student-infer
```
Outputs `./sub/student_submission.csv` with probability columns:
- `winner_model_a`
- `winner_model_b`
- `winner_tie`

## Tips for better Log Loss
- Train longer (5–8 epochs) with early stopping.
- Increase `--student-max-samples` or use the full dataset.
- Use a stronger model (e.g., `bert-base-uncased`, `deberta-v3-base`).
- Add soft-label distillation with teacher logits (KL + CE).
- Keep calibration on a separate split; report holdout metrics.
