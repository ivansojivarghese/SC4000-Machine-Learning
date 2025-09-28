# LMSYS Chatbot Arena First Place Solution (Mock Reproduction)

This workspace contains a lightweight, runnable reproduction of the high-level pipeline from the BlackPearl first place solution. It simulates the phases (post-pretrain, CV, distillation, LoRA merge, GPTQ, TTA) without requiring large models/GPUs.

What you can do locally:
- Run a full mock training pipeline to produce configs, mock logits, and a merged model folder.
- Run an inference pipeline on `./data/test.csv` to generate `./sub/final_submission.csv`.

Data layout expected:
- Training CSV: `./data/lmsys-chatbot-arena/train.csv` with columns: `prompt,response_a,response_b,winner`.
- Test CSV: `./data/test.csv` (already present in this workspace).

Quick start:
1. Install minimal deps (Windows PowerShell):
   - `pip install -r requirements.txt`
2. (Optional) Split folds from train CSV:
   - `python data_preprocessing.py`
3. Run the mock training pipeline:
   - `python main.py --mode train`
4. Run inference (uses mock quantization + TTA) and write submission:
   - `python main.py --mode inference`

Notes:
- This is a mock pipeline for structural validation. To train real models, replace the mock code with Hugging Face/PEFT training and ensure sufficient GPU resources.
