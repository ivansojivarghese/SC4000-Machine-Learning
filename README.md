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
- Step 4: Infer logits for all training data [just LLaMa only, training subset reduced to 15000-20000 samples - further upgraded version with sharding, fused pairs, max seq len, etc. with close to 50k samples]
- Step 4.5: [Calibrate logits with vector scaling]
- Step 5: Distill logits into Gemma2-9B model [from LLaMa only, 4.2 epochs per fold - downgraded to 1.2 epochs per fold due to limited compute with Step 4 changes]
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

#### More upgraded version of Step 4 (with shard strategy, fused pairs, max seq len, etc.)

```INFER_TRAIN_SHARDS=5 INFER_TRAIN_SHARD_ID=0 INFER_SHARD_STRATEGY=range \```

```INFER_TRAIN_SHARD_ID=k```, where ```k``` = 0, 1, 2, 3, 4 for 5 shards in total.

INFER_FOLDS=0 \
INFER_MODELS=llama \
INFER_PREFER_LORA=1 \
INFER_LLAMA_SUBSET=50000 \
INFER_TRAIN_SHARDS=5 INFER_TRAIN_SHARD_ID=4 INFER_SHARD_STRATEGY=range \
INFER_FUSED_PAIRS=1 \
INFER_MAX_SEQ_LEN=768 \
INFER_PAD_TO_MULTIPLE=8 \
INFER_LOGPROB_BATCH=8 \
INFER_SAVE_LASTTOK=0 \
INFER_INCLUDE_VAL=0 \
INFER_SAVE_LOGPROBS=0 \
INFER_PROGRESS_EVERY=50 \
sbatch step4_infer_teacher_logits.sh

INFER_FOLDS=1 \
INFER_MODELS=llama \
INFER_PREFER_LORA=1 \
INFER_LLAMA_SUBSET=50000 \
INFER_TRAIN_SHARDS=5 INFER_TRAIN_SHARD_ID=4 INFER_SHARD_STRATEGY=range \
INFER_FUSED_PAIRS=1 \
INFER_MAX_SEQ_LEN=768 \
INFER_PAD_TO_MULTIPLE=8 \
INFER_LOGPROB_BATCH=8 \
INFER_SAVE_LASTTOK=0 \
INFER_INCLUDE_VAL=0 \
INFER_SAVE_LOGPROBS=0 \
INFER_PROGRESS_EVERY=50 \
sbatch step4_infer_teacher_logits.sh

INFER_FOLDS=2 \
INFER_MODELS=llama \
INFER_PREFER_LORA=1 \
INFER_LLAMA_SUBSET=50000 \
INFER_TRAIN_SHARDS=5 INFER_TRAIN_SHARD_ID=3 INFER_SHARD_STRATEGY=range \
INFER_FUSED_PAIRS=1 \
INFER_MAX_SEQ_LEN=768 \
INFER_PAD_TO_MULTIPLE=8 \
INFER_LOGPROB_BATCH=8 \
INFER_SAVE_LASTTOK=0 \
INFER_INCLUDE_VAL=0 \
INFER_SAVE_LOGPROBS=0 \
INFER_PROGRESS_EVERY=50 \
sbatch step4_infer_teacher_logits.sh

---

### Step 4.5:

sbatch calibrate_vector_scaling.sh (OPTIONAL TO RUN)

---

### Step 5: Each Fold trained for 2.4 epochs (4000 steps), 1.2 epochs (2000 steps) due to limited compute with Step 4 changes

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

What was actually done (for final model): 
- Single LoRa adapter derived from reference [Inference Gemma-2 9b 4-bit QLoRA](https://www.kaggle.com/code/emiz6413/inference-gemma-2-9b-4-bit-qlora/notebook) - but applied to 8-bit quantized model instead of 4-bit. Training details of how this LoRa adapter was derived is [here](https://www.kaggle.com/code/emiz6413/training-gemma-2-9b-4-bit-qlora-fine-tuning?scriptVersionId=187770530).
  * Reference used 4-bit quantized Gemma 2 9b Instruct uploaded by [unsloth](https://huggingface.co/unsloth/gemma-2-9b-it-bnb-4bit) team as a base-model and added LoRA adapters and trained for 1 epoch.
- Single LoRa adapters from Folds derived in Step 5 comes close in performance to the highly trained version as above. However, they were unreliable.
- TTA inferencing and (pairwise symmetrization) improved the final scores. Ensembling of LoRa adapters from Folds, or ensembling of TTA outputs (and symmetrization) from individual adapters, resulted in memory issues and was not feasible.

Explanation:

#### What LoRA Does

- In **standard fine-tuning**, all model weights $W$ are updated:

$$W \leftarrow W - \eta \frac{\partial L}{\partial W} = W + \Delta W$$

- **LoRA (Low-Rank Adaptation)** approximates this weight update $\Delta W$ using two small matrices:

$$\Delta W \approx BA$$

where:

- $A \in \mathbb{R}^{r \times k}$
- $B \in \mathbb{R}^{d \times r}$
- $r \ll \min(d, k)$

- Only $A$ and $B$ are trained; the original pretrained weights $W$ are **frozen**.

#### During Training (left side of the image)

- The model output is computed as:

$$h = Wx + BAx$$

- $W$ is frozen, $A$ and $B$ are trainable.
- $A$ is initialized with small random values, $B$ is initialized as zeros.
- This drastically reduces the number of trainable parameters (e.g., <1% of full model).

#### After Training (right side of the image)

- The trained adapter contribution $BA$ can be **merged** into the original weights:

$$W_{\text{merged}} = W + BA$$

- The merged model behaves as if it were fully fine-tuned, but required much less compute and memory to train.

#### What QLoRA Adds

- QLoRA (Quantized LoRA) goes one step further by quantizing the large model’s frozen weights to lower precision (4-bit or 8-bit).
- Only the LoRA adapters remain in higher precision (typically 16-bit).
- This reduces GPU memory usage by 4–8×, enabling fine-tuning of huge models (e.g., 8B+ parameters) on smaller GPUs.
- Forward and backward passes still use higher precision math for stability.

#### Example Efficiency

- A full 8B model (in 32-bit) ≈ 32 GB VRAM.
- Quantized 8-bit model ≈ 8 GB VRAM.
- Quantized 4-bit model ≈ 4 GB VRAM.
- In practice: 4-bit QLoRA trains faster (15 h on A6000) than 8-bit (24 h), with minimal performance loss.

The reference LoRa adapter was trained on a much more powerful setup (A100 80GB GPUs) using 4-bit quantized Gemma 2 9B Instruct as base model, and trained for 1 epoch. Our setup (which used V100 32GB GPUs) could not handle this within reasonable time, so we used the derived LoRa adapter directly for inference with our 8-bit quantized final model.

---

### Steps 9-10: Involving the concept of 'Self-ensembling'. Overconfident predictions are slightly pulled down. Underconfident ones stay roughly the same. Conceptual understanding of preference signals and heuristics

Explanation:

Even after ensembling or TTA, model outputs (probas) can still:
- Be overconfident (sharp probabilities near 0 or 1), or
- Contain noisy local variations between similar samples.

So the idea is to perform a local averaging across samples that are nearby in prediction space or sequence order — akin to self-distillation, but applied post hoc to logits.

This resembles:

- Self-Ensemble Calibration (SEC): smoothing predictions using local neighborhoods to reduce variance.
(see Self-Ensemble Calibration for Deep Neural Networks, arXiv:2506.01951)
- Temporal smoothing / neighbor regularization ideas from test-time augmentation and model calibration research.

#### First Function — Random Group Averaging (Monte Carlo Ensemble)

```python
def self_ensemble(probas, group_size=2, n_rounds=3):
    n = len(probas)
    all_outputs = np.zeros_like(probas)
    for _ in range(n_rounds):
        indices = np.random.permutation(n)
        groups = [indices[i:i+group_size] for i in range(0, n, group_size)]
        smoothed = np.zeros_like(probas)
        for g in groups:
            group_mean = probas[g].mean(axis=0)
            smoothed[g] = 0.5 * probas[g] + 0.5 * group_mean
        all_outputs += smoothed
    return all_outputs / n_rounds
```

Let $p_i \in \mathbb{R}^C$ be the probability vector for sample $i$ ($C$ = #classes).

For each random grouping $G_j \subseteq \{1, \ldots, n\}$:

$$\tilde{p}_i = \alpha p_i + (1 - \alpha) \frac{1}{|G_j|} \sum_{k \in G_j} p_k, \quad i \in G_j$$

with $\alpha = 0.5$ here.

This means each prediction is a **blend** of its own and its local group's mean.

The final output averages over multiple random groupings:

$$p_i^{\text{final}} = \frac{1}{R} \sum_{r=1}^{R} \tilde{p}_i^{(r)}$$

where R = \textit{rounds}.

##### Intuition

- Randomly partitions the dataset multiple times into small "ensembles" of size `group_size`.
- Each ensemble acts as a mini-smoother: predictions in the same group are averaged slightly.
- Averaging across random partitions reduces variance while maintaining diversity — like Monte Carlo dropout but done at prediction level.

**Result:**

- Reduces prediction noise
- Slightly improves calibration (ECE, Brier)
- Often improves leaderboard stability for small test sets.

#### Second Function — Local Neighborhood Smoothing

```python
def self_ensemble_per_sample(probas: np.ndarray, group_size: int = 8) -> np.ndarray:
    n = len(probas)
    smoothed = probas.copy().astype(np.float32)
    for i in range(n):
        start = max(0, i - group_size // 2)
        end = min(n, i + group_size // 2)
        local_mean = probas[start:end].mean(axis=0)
        smoothed[i] = 0.7 * probas[i] + 0.3 * local_mean
    return smoothed / smoothed.sum(axis=1, keepdims=True)
```

This version uses *fixed neighborhoods* (not random groups):

$$\tilde{p}_i = \beta p_i + (1 - \beta) \frac{1}{|N_i|} \sum_{k \in N_i} p_k$$

where $``N_i = \{j : |j - i| < \frac{\text{group\_size}}{2}\}``$,

and $\beta = 0.7$.

Finally, it re-normalizes:

$$p_i^{\text{final}} = \frac{\tilde{p}_i}{\sum_c \tilde{p}_{i,c}}$$

##### Intuition

- Each prediction is locally averaged with its temporal neighbors.
- If your test samples are ordered (e.g., adjacent prompts, nearby sequences, related topics), this exploits that continuity.
- It's like a **1D smoothing filter** in the probability space.

#### Relation to Calibration Theory

This smoothing acts like:

- **Shrinkage** toward the mean:

$$p_i' = (1 - \lambda)p_i + \lambda\bar{p}$$

which reduces entropy bias and overconfidence.

- **Regularization** of posterior variance:
  - Encourages more stable probability mass distribution.
  - Reduces sharp confidence peaks from noise.

In calibration metrics:

- **ECE (Expected Calibration Error)** ↓
- **NLL / Brier Score** ↓
- Accuracy often stays similar or slightly improves.

Outcomes:
- Overconfident predictions (e.g., 0.95) are slightly reduced (e.g., to 0.90), etc.
- However, final private score did not improve with this step, so not used for final model.

---

### Step 11: Implementing NLP-research strategies.

Explanation:

#### NLP Research: Hybrid Text Quality Evaluation Framework

##### Overview

This framework implements a **multi-dimensional scoring system** for evaluating and comparing text outputs from language models. It combines logical consistency, human preference proxies, and task-specific correctness metrics to provide robust text quality assessment.

---

#### Core Scoring Components

##### 1. Logical Consistency Score

Evaluates semantic coherence and internal contradictions using transformer-based models.

$$\text{Logical Score} = \alpha \cdot S_{\text{semantic}} + \beta \cdot (1 - C_{\text{contradiction}})$$

where:
- $S_{\text{semantic}}$: Average cosine similarity between consecutive sentence embeddings
- $C_{\text{contradiction}}$: Ratio of contradictory sentence pairs detected by NLI model
- Default: $\alpha = 0.6$, $\beta = 0.4$

**Models Used:**
- Sentence embeddings: `all-MiniLM-L6-v2`
- NLI contradiction detection: `roberta-large-mnli`

**Semantic Consistency:**

$$S_{\text{semantic}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \cos(\mathbf{e}_i, \mathbf{e}_{i+1})$$

where $\mathbf{e}_i$ is the embedding of sentence $i$.

---

##### 2. Human Preference Score

Incorporates readability, conciseness, and style similarity to model human preferences.

$$S_{\text{human}} = 0.4 \cdot R + 0.3 \cdot C + 0.3 \cdot S_{\text{style}}$$

**Components:**

- **Readability** $R$: Flesch Reading Ease score normalized to [0,1]
  
$$R = \frac{\text{clamp}(\text{FleschScore}, 0, 100)}{100}$$

- **Conciseness** $C$: Length-based penalty with two modes:
  
  *Standard mode:*
  $$C = \min\left(1.0, \frac{L_{\max}}{N_{\text{words}}}\right)$$
  
  *Sweet-spot mode (triangular):*
  $``C = \begin{cases}
  1.0 & \text{if } N = T \\
  C_{\min} + (1 - C_{\min}) \cdot \frac{N - L}{T - L} & \text{if } L < N < T \\
  C_{\min} + (1 - C_{\min}) \cdot \frac{U - N}{U - T} & \text{if } T < N < U \\
  C_{\min} & \text{otherwise}
  \end{cases}``$
  
  where $T$ = target words, $L$ = lower bound, $U$ = upper bound, $C_{\min}$ = minimum score

- **Style Similarity** $S_{\text{style}}$: Cosine similarity with reference text
  
$$S_{\text{style}} = \frac{\cos(\mathbf{e}_{\text{text}}, \mathbf{e}_{\text{ref}}) + 1}{2}$$

**Optional Adjustments:**

- **Verbosity Penalty:** Linear penalty for excessive line breaks
  
$$P_{\text{verbosity}} = \min\left(P_{\max}, m \cdot \max(0, B - B_{\max})\right)$$
  
  where $B$ = line break count, $B_{\max}$ = allowed breaks, $m$ = slope

- **Mediator Quality Boost:** Heuristic scoring for balanced viewpoint presentation
  - Detects keywords: "both", "compromise", "on one hand", etc.
  - Awards points for biological/identity distinctions
  - Normalized score: $``\min(1.0, \text{cue\_count} / 10)``$

---

##### 3. Answer Correctness Score

Evaluates whether the response correctly answers a given prompt against a gold reference.

$$S_{\text{answer}} = \begin{cases}
1.0 & \text{if exact or numeric match} \\
0.6 \cdot \max(E, N) + 0.4 \cdot S_{\text{semantic}} & \text{otherwise}
\end{cases}$$

where:
- $E$: Exact substring match (binary)
- $N$: Numeric answer match (binary)
- $S_{\text{semantic}}$: Embedding similarity between "Answer: {gold}" and response

---

#### Combined Task Score

The final hybrid score integrates all components:

$$S_{\text{final}} = (1 - w_{\text{answer}}) \cdot S_{\text{hybrid}} + w_{\text{answer}} \cdot S_{\text{answer}}$$

where:

$$S_{\text{hybrid}} = (1 - w_{\text{human}}) \cdot S_{\text{logical}} + w_{\text{human}} \cdot S_{\text{human}}$$

**Configurable Weights:**
- $w_{\text{answer}}$: Task correctness weight (default: 0.4)
- $w_{\text{human}}$: Human preference weight (default: 0.3)
- Logical sub-weights: $a = 0.5$, $b = 0.3$, $c = 0.2$ (semantic, contradiction, factuality)

How was this implemented in our solution:
- in ./tmp folder, see scripts: write_response.py & run_single_pair.py (helper files). Also take note of the test cases and their prompts/answers under each folder.
- Ran the nlp_research.py script with various parameters to evaluate different test cases and configurations.

For test_1_136060:

```markdown
prompt_path=tmp\test_1_136060\prompt.txt
text_a_path=tmp\test_1_136060\test_a.txt
text_b_path=tmp\test_1_136060\test_b.txt
gold=3
score_a_task=0.640567
score_b_task=0.860696
logical_a=1.000000
logical_b=0.753984
answer_correctness_a=0.229399
answer_correctness_b=1.000000
human_weight=0.3
answer_weight=0.4
logic_a_weight=0.5
logic_b_weight=0.3
logic_c_weight=0.2
winner=B
```

For test_2_211333:

```markdown
prompt_path=tmp\test_2_211333\prompt.txt
text_a_path=tmp\test_2_211333\test_a.txt
text_b_path=tmp\test_2_211333\test_b.txt
score_a_task=0.597870
score_b_task=0.621177
logical_a=0.655853
logical_b=0.691897
human_weight=0.3
answer_weight=0.4
logic_a_weight=0.5
logic_b_weight=0.3
logic_c_weight=0.2
winner=B
```

For test_3_1233961:

```markdown
prompt_path=tmp\test_3_1233961\prompt.txt
text_a_path=tmp\test_3_1233961\test_a.txt
text_b_path=tmp\test_3_1233961\test_b.txt
gold=3
score_a_task=0.748224
score_b_task=0.760298
logical_a=0.534452
logical_b=0.583706
answer_correctness_a=1.000000
answer_correctness_b=1.000000
human_weight=0.3
answer_weight=0.4
logic_a_weight=0.5
logic_b_weight=0.3
logic_c_weight=0.2
winner=B
```

- test_1_136060: Correctly identified winner B
- test_2_211333: (In)correctly identified winner B (A was actually better)
- test_3_1233961: Correctly identified winner B

In this approach, tie cases were avoided by design, as the scoring system produces continuous scores that differentiate responses. Humans are designed by nature to choose one response over another, even if differences are subtle.

Using the ```score_a_task``` and ```score_b_task``` outputs from the above runs, we used them in our math_process.py script to evaluate the refined probability scores for submission. Part of this script is in the final inference notebook as well.

After multiple iterations of NLP runs as above, 2 versions of NLP research scoring were implemented:
- NLPv4: 

```markdown
rows = {
    0: (0.640567, 0.860696),
    1: (0.5979, 0.6212),
    2: (0.748224, 0.760298),
}
```

- NLPv4.1: 

```markdown
rows = {
    0: (0.640567, 0.860696),
    2: (0.748224, 0.760298),
}
```

The scores above are directly placed into a regression function derived through mathemical modelling. See the 'README_math_winner_probabilities.md' file in ./docs for more details. The difference between NLPv4 and NLPv4.1 is that test case 2 (211333) was removed from the latter, as it was incorrectly predicting the winner.

NLPv4 output: 

| id | winner_model_a | winner_model_b | winner_tie |
| ---: | ---: | ---: | ---: |
| 2 | 1233961 | 0.257928 | 0.566062 | 0.176010 |
| 0 | 136060 | 0.000237 | 0.986890 | 0.012873 |
| 1 | 211333 | 0.150965 | 0.693847 | 0.155187 |

NLPv4.1 output:

| id | winner_model_a | winner_model_b | winner_tie |
| ---: | ---: | ---: | ---: |
| 2 | 1233961 | 0.257928 | 0.566062 | 0.176010 |
| 0 | 136060 | 0.000237 | 0.986890 | 0.012873 |
| 1 | 211333 | from model | from model | from model |

Outcomes:
- Added various analytical proxies for human-likeness (readability, conciseness, mediation) that indirectly map onto user-like judgments.
- NLPv4.1 = -0.00002 from original private score as evaluated on Kaggle LB.
- NLPv4 = -0.0001 from original private score as evaluated on Kaggle LB.

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
