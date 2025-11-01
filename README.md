# Pipeline flow:

Step 1: Post-pretrain large models on UT dataset [just LLaMA only]

Step 2: Split Kaggle + 33k data into 5 folds [3 folds]

Step 2.5: Remove argmax samples from folds [3 folds]

Step 3: Train LLaMA3 70B & Qwen2 72B on folds [just LLaMA only]

Step 4: Infer logits for all training data [just LLaMa only]

Step 4.5: Calibrate logits with vector scaling [just LLaMa only]

Step 5: Distill logits into Gemma2-9B model [from LLaMa only]

Step 6: Ensemble LoRA layers from 5 folds

Step 7: Quantize final model to 8-bit (GPTQ)

Step 8: Apply TTA during inference

Step 8.5: TTA Symmetrization Post-processing

Step 9: Evaluate CV and LB

---

## References

- [Training Gemma 2 9B 4-bit QLoRA Fine-Tuning (Kaggle Notebook)](https://www.kaggle.com/code/emiz6413/training-gemma-2-9b-4-bit-qlora-fine-tuning/notebook#Note)

- [BlackPearl No-Leak 1st Place Solution – LMSYS Chatbot Arena (Kaggle Competition Write-up)](https://www.kaggle.com/competitions/lmsys-chatbot-arena/writeups/blackpearl-no-leak-1st-place-solution-distill-is-a)

- [RLP: Reinforcement as a Pretraining Objective](https://research.nvidia.com/labs/adlr/RLP/)

| Section                     | Quote                                                         | Relevance                                            | Action       |
| ---------------------------- | ------------------------------------------------------------- | ---------------------------------------------------- | ------------ |
| **Reward Shaping Mechanism** | “Rewards computed by contrasting predictor with EMA baseline” | Shows how to weight folds relative to baseline       | Implement  |
| **Group-Relative Advantages** | `A(c) = r(c) - mean(r_group)` ensures unbiased gradients      | Explains why relative weighting works mathematically | Study      |
| **Information Gain**         | “Reward = increase in log-likelihood”                         | How to weight hard samples higher                    | Implement  |
| **Monotonic Improvement Proof** | “Even negative rewards improve if relatively better”        | Justifies not discarding the last Fold                      | Understand |

---

HF_token:

## SLURM Start-up Commands:

module load anaconda

conda activate myenv

cd ~/exported-assets_sc4000/

## SLURM Exit Commands:

conda deactivate myenv

exit

## Quick Model Validation (against test cases):

sbatch check_qwen_params.sh

---

## Step 1:

RUN_STAGE=gemma SCRATCH_BASE=/scratch-shared/tc1proj005 sbatch step1_post_pretrain.sh

RUN_STAGE=llama SCRATCH_BASE=/scratch-shared/tc1proj005 sbatch step1_post_pretrain.sh

---

## Step 2:

sbatch step2_make_folds.sh

---

## Step 2.5: 

sbatch remove_argmax_from_folds.sh 

---

## Step 3:

TEACHER_SUBSET_SIZE=20000 TEACHER_MAX_STEPS=300 TEACHER_FOLDS="0" SKIP_QWEN=1 sbatch step3_train_teachers.sh

TEACHER_SUBSET_SIZE=20000 TEACHER_MAX_STEPS=300 TEACHER_FOLDS="1" SKIP_QWEN=1 sbatch step3_train_teachers.sh

TEACHER_SUBSET_SIZE=20000 TEACHER_MAX_STEPS=300 TEACHER_FOLDS="2" SKIP_QWEN=1 sbatch step3_train_teachers.sh

---

## Step 4:

INFER_FOLDS=0 INFER_MODELS=llama INFER_PREFER_LORA=1 INFER_LLAMA_SUBSET=15000-20000 INFER_LOGPROB_BATCH=8 INFER_PROGRESS_EVERY=5 sbatch step4_infer_teacher_logits.sh 

INFER_FOLDS=1 INFER_MODELS=llama INFER_PREFER_LORA=1 INFER_LLAMA_SUBSET=15000-20000 INFER_LOGPROB_BATCH=8 INFER_PROGRESS_EVERY=5 sbatch step4_infer_teacher_logits.sh

INFER_FOLDS=2 INFER_MODELS=llama INFER_PREFER_LORA=1 INFER_LLAMA_SUBSET=15000-20000 INFER_LOGPROB_BATCH=8 INFER_PROGRESS_EVERY=5 sbatch step4_infer_teacher_logits.sh

---

## Step 4.5:

sbatch calibrate_vector_scaling.sh (OPTIONAL TO RUN)

---

## Step 5:

FOLDS=0 sbatch step5_distill_student.sh

FOLDS=1 sbatch step5_distill_student.sh

FOLDS=2 sbatch step5_distill_student.sh

---

## Step 6:

sbatch step6_lora_ensemble.sh

---

## Step 7: 

sbatch step7_quantize_gptq.sh (OPTIONAL TO RUN)

---

## Step 8:

sbatch step8_infer_tta.sh

---

## Step 8.5: TTA Symmetrization Post-processing

---

## Step 9:

sbatch step9_eval.sh (OPTIONAL TO RUN)

---

## Notes:

### Recommended Steps (based on top solutions):
#### Teacher Training
Train strong teacher models (e.g., LLaMA, Qwen, Gemma) on all available labeled data (Kaggle train + external datasets).
Use cross-validation folds for robust evaluation.
#### Distillation
Use the teacher models to generate soft labels (logits or probabilities) for the student model.
Train a smaller/faster student model (e.g., DistilBERT, LoRA-adapted LLM) to mimic the teacher’s outputs (distillation).
#### Calibration
Calibrate the student’s output probabilities (temperature scaling) for better log-loss.
#### Inference & Ensembling
Run inference on the test set using the student model.
Optionally ensemble predictions from multiple models/folds for better performance.
#### Submission
Output a CSV with probabilities for each class (A, B, tie) for each test sample.

### What to do next in your pipeline:
Finish training your teacher models (your script is set up for this).
Run distillation to train your student model using teacher outputs.
Calibrate and validate your student model’s predictions.
Generate your submission file with class probabilities.
#### Tips from the winning solution:
Use all available data (Kaggle + external) for training.
Distillation is key: student models trained on teacher logits generalize better.
Calibration improves log-loss.
Ensembling multiple models/folds can further boost performance.
