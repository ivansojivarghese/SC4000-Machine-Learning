# Solution:

=== Referencing Winning Solution: [BlackPearl No-Leak 1st Place Solution – LMSYS Chatbot Arena (Kaggle Competition Write-up)](https://www.kaggle.com/competitions/lmsys-chatbot-arena/writeups/blackpearl-no-leak-1st-place-solution-distill-is-a) ===

Models: 
- [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) (Student)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (Teacher)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Teacher)

## Pipeline Flow: 

Step `k`: Winning Solution step [our solution step / difference(s)]

- Step 1: Post-pretrain Teacher models on UT dataset [just LLaMA only]
- Step 2: Split Kaggle + 33k data into 5 folds [3 folds]
- Step 2.5: Remove argmax samples from folds [3 folds]
- Step 3: Train Teacher models on folds [just LLaMA only]
- Step 4: Infer logits for all training data [just LLaMa only]
- Step 4.5: Calibrate logits with vector scaling [just LLaMa only]
- Step 5: Distill logits into Gemma2-9B model [from LLaMa only]

=== End of Winning Solution ===

=== Referencing Inference Solution [Inference Gemma-2 9b 4-bit QLoRA](https://www.kaggle.com/code/emiz6413/inference-gemma-2-9b-4-bit-qlora/notebook) ===

- Step 6: Direct inference (& ensembling) of LoRA adapters (from Folds) to Gemma2ForSequenceClassification
- Step 7: TTA Symmetrization Post-processing

===

=== Below Steps of Winning Solution not used for final model ===

- Step 6: Ensemble LoRA layers from 5 folds
- Step 7: Quantize final model to 8-bit (GPTQ)
- Step 8: Apply TTA during inference
- Step 9: Evaluate CV and LB

===

---

## Notes (of good relevance)

| Section                     | Quote                                                         | Relevance                                            | Action       |
| ---------------------------- | ------------------------------------------------------------- | ---------------------------------------------------- | ------------ |
| **Reward Shaping Mechanism** | “Rewards computed by contrasting predictor with EMA baseline” | Shows how to weight folds relative to baseline       | Implement  |
| **Group-Relative Advantages** | `A(c) = r(c) - mean(r_group)` ensures unbiased gradients      | Explains why relative weighting works mathematically | Study      |
| **Information Gain**         | “Reward = increase in log-likelihood”                         | How to weight hard samples higher                    | Implement  |
| **Monotonic Improvement Proof** | “Even negative rewards improve if relatively better”        | Justifies not discarding the last Fold                      | Understand |

---

HF_token: REDACTED

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

## Winning Solution

### Step 1:

RUN_STAGE=gemma SCRATCH_BASE=/scratch-shared/tc1proj005 sbatch step1_post_pretrain.sh

RUN_STAGE=llama SCRATCH_BASE=/scratch-shared/tc1proj005 sbatch step1_post_pretrain.sh

---

### Step 2:

sbatch step2_make_folds.sh

---

### Step 2.5: 

sbatch remove_argmax_from_folds.sh 

---

### Step 3:

TEACHER_SUBSET_SIZE=20000 TEACHER_MAX_STEPS=300 TEACHER_FOLDS="0" SKIP_QWEN=1 sbatch step3_train_teachers.sh

TEACHER_SUBSET_SIZE=20000 TEACHER_MAX_STEPS=300 TEACHER_FOLDS="1" SKIP_QWEN=1 sbatch step3_train_teachers.sh

TEACHER_SUBSET_SIZE=20000 TEACHER_MAX_STEPS=300 TEACHER_FOLDS="2" SKIP_QWEN=1 sbatch step3_train_teachers.sh

---

### Step 4:

INFER_FOLDS=0 INFER_MODELS=llama INFER_PREFER_LORA=1 INFER_LLAMA_SUBSET=15000-20000 INFER_LOGPROB_BATCH=8 INFER_PROGRESS_EVERY=5 sbatch step4_infer_teacher_logits.sh 

INFER_FOLDS=1 INFER_MODELS=llama INFER_PREFER_LORA=1 INFER_LLAMA_SUBSET=15000-20000 INFER_LOGPROB_BATCH=8 INFER_PROGRESS_EVERY=5 sbatch step4_infer_teacher_logits.sh

INFER_FOLDS=2 INFER_MODELS=llama INFER_PREFER_LORA=1 INFER_LLAMA_SUBSET=15000-20000 INFER_LOGPROB_BATCH=8 INFER_PROGRESS_EVERY=5 sbatch step4_infer_teacher_logits.sh

---

### Step 4.5:

sbatch calibrate_vector_scaling.sh (OPTIONAL TO RUN)

---

### Step 5: Each Fold trained for 3 epochs (5000 steps)

FOLDS=0 sbatch step5_distill_student.sh

FOLDS=1 sbatch step5_distill_student.sh

FOLDS=2 sbatch step5_distill_student.sh

---

## Inference Solution

---

### Step 6: Direct inference (& ensembling) of LoRA adapters (from Folds) to Gemma2ForSequenceClassification

---

### Step 7: TTA Symmetrization Post-processing

---

## Winning Solution (did not use the below steps for final model)

### Step 6:

sbatch step6_lora_ensemble.sh

---

### Step 7: 

sbatch step7_quantize_gptq.sh (OPTIONAL TO RUN)

---

### Step 8:

sbatch step8_infer_tta.sh

---

### Step 9:

sbatch step9_eval.sh (OPTIONAL TO RUN)

---

## References

- [Training Gemma 2 9B 4-bit QLoRA Fine-Tuning (Kaggle Notebook)](https://www.kaggle.com/code/emiz6413/training-gemma-2-9b-4-bit-qlora-fine-tuning/notebook#Note)

- [Inference Gemma-2 9b 4-bit QLoRA](https://www.kaggle.com/code/emiz6413/inference-gemma-2-9b-4-bit-qlora/notebook)

- [BlackPearl No-Leak 1st Place Solution – LMSYS Chatbot Arena (Kaggle Competition Write-up)](https://www.kaggle.com/competitions/lmsys-chatbot-arena/writeups/blackpearl-no-leak-1st-place-solution-distill-is-a)

- [RLP: Reinforcement as a Pretraining Objective](https://research.nvidia.com/labs/adlr/RLP/)
