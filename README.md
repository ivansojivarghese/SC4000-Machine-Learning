# Solution:

=== Referencing Winning Solution: [BlackPearl No-Leak 1st Place Solution – LMSYS Chatbot Arena (Kaggle Competition Write-up)](https://www.kaggle.com/competitions/lmsys-chatbot-arena/writeups/blackpearl-no-leak-1st-place-solution-distill-is-a) ===

Models: 
- [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) (Student)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (Teacher)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Teacher)

## Pipeline Flow: 

Step `k`: Winning Solution step [our solution step / difference(s)]

- Step 1: Post-pretrain all models on the [UT dataset](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/499756) for 1 epoch each [just LLaMA and Gemma only, 0.01 epochs each]
- Step 2: Split Kaggle + [33k data](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/500973) into 5 folds [3 folds]
- Step 2.5: [Remove argmax samples from all 3 folds, data cleaning]
- Step 3: Train Teacher models on folds for 1 epoch each [just LLaMA only, 0.2 epochs, training subset reduced to 20000 samples]
- Step 4: Infer logits for all training data [just LLaMa only, training subset reduced to 15000-20000 samples]
- Step 4.5: [Calibrate logits with vector scaling]
- Step 5: Distill logits into Gemma2-9B model [from LLaMa only, 3.6 epochs per fold]
- Step 6: Ensemble LoRA layers from Folds [3 folds, to 16-bit initial model]
- Step 7: Quantize final model to 8-bit in GPTQ [& 4-bit GPTQ]

=== End of Winning Solution ===

=== Referencing Inference Solution [Inference Gemma-2 9b 4-bit QLoRA](https://www.kaggle.com/code/emiz6413/inference-gemma-2-9b-4-bit-qlora/notebook) ===

Mostly fusioned/self-derived Steps from this point. No direct referencing from Kaggle sources or solutions.

- Step 8: Direct TTA inferencing (& possible ensembling) of LoRA adapters (from Folds), or with single LoRa adapter from the best Fold, using the quantized 8-bit final model. Pairwise TTA (symmetrization) also implemented. See [RETHINKING REWARD MODELING IN PREFERENCE
BASED LARGE LANGUAGE MODEL ALIGNMENT](https://openreview.net/pdf/c86736447d5c66dec8140360ab743a130d9ff219.pdf#:~:text=reward%20estimation%20in%20the%20context,LLMs%2C%202%20datasets%2C%203%20response).
- Step 9: Involving the concept of 'Self-ensembling'. See [Self-Ensemble: Mitigating Confidence Mis-calibration for Large Language Models](https://arxiv.org/html/2506.01951v2#:~:text=set,frequencies%20on%20a%20validation%20set). Overconfident predictions are slightly pulled down. Underconfident ones stay roughly the same. See [DORM: Preference Data Weights Optimization for Reward Modeling in LLM Alignment](https://aclanthology.org/2025.findings-emnlp.1237/) as well - in [PDF](https://aclanthology.org/2025.findings-emnlp.1237.pdf).
- Step 10: There is no such thing as a 'tie' case, especially in human preferences. So we need to possibly get a majority probability for either responses A or B in any test case scenario. See [Aligning Large Language Models with Implicit Preferences from
 User-Generated Content](https://aclanthology.org/2025.acl-long.384.pdf). Conceptual ideas for this:
  * [A Survey on Human Preference Learning for Large Language Models](https://arxiv.org/html/2406.11191v1) - Preference signals have different sources and formats (direct human feedback, model-simulated feedback, heuristic/inductive biases) and these impact quality vs scale. The usage of preference signals spans SFT, contrastive preference learning, RLHF, preference-conditioned generation.
  * [Larger or Smaller Reward Margins to Select Preferences for LLM Alignment?](https://openreview.net/forum?id=ncTwQagrj8) - They propose an alignment potential metric that integrates explicit reward margin (e.g., label/regression gap) and implicit reward margin (model’s current margin) to quantify how “useful” a preference pair is for alignment.
- Step 11: Added some NLP-based research components. Referencing from cognitive science, machine learning, and NLP evaluation research, [MLHP](https://mlhp.stanford.edu/src/chap1.html) introduces human preference modeling as structured, decomposable cognition — logical structure (truth-conditional) and hedonic structure (human-aligned desirability). Further [Apple research](https://machinelearning.apple.com/research/predicting-preferences) frames predicting preferences as learning a latent preference manifold from implicit feedback — rather than fixed metrics. To define latent proxies (readability, conciseness, mediation) that indirectly map onto user-like judgments. Referencing [Deep Reinforcement Learning
 from Human Preferences](https://proceedings.neurips.cc/paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf), the ```combined_preference_score``` and ```compare_with_human_bias``` functions in ```nlp_research.py``` are conceptually equivalent to the reward model in RLHF:
  * The logical score is akin to truthfulness reward.
  * The human score is akin to helpfulness or readability reward.
  * The final scalar mixture ```(1 - human_weight) * logical + human_weight * human_pref``` is exactly how RLHF reward functions blend multiple alignment objectives.
  * This uses analytical proxies for human-likeness.

| Function Cluster                                    | Cognitive Construct                                  | MLHP / Apple Analogy                |
| --------------------------------------------------- | ---------------------------------------------------- | ----------------------------------- |
| `semantic_consistency_score`, `contradiction_ratio` | Rational coherence (System 2)                        | Truth-conditional preference        |
| `readability_score`, `conciseness_sweet_spot_score` | Communicative efficiency (Grice’s Maxim of Quantity) | Human preference clarity            |
| `mediator_quality_score`                            | Social cognition / empathy                           | Cooperative intent                  |
| `verbosity_penalty`                                 | Cognitive load minimization                          | Personalized readability preference |


===

=== Below Steps of Winning Solution not used for final model ===

- Step 8: Apply TTA during inference
- Step 9: Evaluate CV and LB

===

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

### Step 5: Each Fold trained for 3.6 epochs (6000 steps)

FOLDS=0 sbatch step5_distill_student.sh

FOLDS=1 sbatch step5_distill_student.sh

FOLDS=2 sbatch step5_distill_student.sh

### Step 6:

sbatch step6_lora_ensemble.sh (16 BIT)

---

### Step 7: Quantization

sbatch gptq.sh (4 BIT QUANTIZATION - NOT USED FOR FINAL MODEL)

sbatch gptq_8bit.sh (8 BIT QUANTIZATION - USED FOR FINAL MODEL)

---

## Inference Solution

### Step 8: Direct TTA inferencing (& possible ensembling) of LoRA adapters (from Folds), or with single LoRa adapter from the best Fold, using the quantized 8-bit final model. Pairwise TTA (symmetrization) also implemented.

### Step 9: Involving the concept of 'Self-ensembling'. Overconfident predictions are slightly pulled down. Underconfident ones stay roughly the same.

---

## Winning Solution (did not use the below steps for final model)

### Step 8:

sbatch step8_infer_tta.sh (OPTIONAL TO RUN)

---

### Step 9:

sbatch step9_eval.sh (OPTIONAL TO RUN)

---

## References

- [Training Gemma 2 9B 4-bit QLoRA Fine-Tuning (Kaggle Notebook)](https://www.kaggle.com/code/emiz6413/training-gemma-2-9b-4-bit-qlora-fine-tuning/notebook#Note)

- [Inference Gemma-2 9b 4-bit QLoRA](https://www.kaggle.com/code/emiz6413/inference-gemma-2-9b-4-bit-qlora/notebook)

- [BlackPearl No-Leak 1st Place Solution – LMSYS Chatbot Arena (Kaggle Competition Write-up)](https://www.kaggle.com/competitions/lmsys-chatbot-arena/writeups/blackpearl-no-leak-1st-place-solution-distill-is-a)

- [RLP: Reinforcement as a Pretraining Objective](https://research.nvidia.com/labs/adlr/RLP/)

---

## Notes (of good relevance)

| Section                     | Quote                                                         | Relevance                                            | Action       |
| ---------------------------- | ------------------------------------------------------------- | ---------------------------------------------------- | ------------ |
| **Reward Shaping Mechanism** | “Rewards computed by contrasting predictor with EMA baseline” | Shows how to weight folds relative to baseline       | Implement  |
| **Group-Relative Advantages** | `A(c) = r(c) - mean(r_group)` ensures unbiased gradients      | Explains why relative weighting works mathematically | Study      |
| **Information Gain**         | “Reward = increase in log-likelihood”                         | How to weight hard samples higher                    | Implement  |
| **Monotonic Improvement Proof** | “Even negative rewards improve if relatively better”        | Justifies not discarding the last Fold                      | Understand |

---
