# main.py
import os
import argparse
import logging
from typing import Dict, Any

from training import TrainingOrchestrator
from inference import InferencePipeline
from student_train_hf import train_student
from student_infer_hf import infer_student
from student_calibrate import calibrate_student

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_directories():
    for d in [
        './model_path', './model_save', './model_save_or', './data', './data/processed_data', './data/oof', './sub'
    ]:
        os.makedirs(d, exist_ok=True)


def load_config() -> Dict[str, Any]:
    return {
        'models': {
            'llama3-70b': {
                'model_path': './model_path/llama3_70b',
                'max_length': 1024, 'lora_r': 64, 'lora_alpha': 128
            },
            'qwen2-72b': {
                'model_path': './model_path/qwen2_72b',
                'max_length': 1024, 'lora_r': 64, 'lora_alpha': 128
            },
            'gemma2-9b': {
                'model_path': './model_path/Gemma2_9b',
                'max_length': 1024, 'lora_r': 64, 'lora_alpha': 128
            }
        },
        'batch_size': 8,
        'gradient_accumulation_steps': 8,
        'max_length': 1024,
        'lora_r': 64,
        'lora_alpha': 128,
        'temperature': 3.0,
        'alpha': 0.7,
        'use_tta': True,
        'tta_lengths': [1024, 2000],
        'quantization_bits': 8,
        'data_paths': {
            'kaggle_train': './data/lmsys-chatbot-arena/train.csv',
            'ut_data': './data/ut_data.csv',
            '33k_data': './data/33k_data.csv',
            'test_data': './data/test.csv'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='LMSYS Mock Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'full', 'student-train', 'student-infer', 'student-calibrate', 'student-eval-holdout'], default='full')
    parser.add_argument('--output-dir', type=str, default='./sub')
    # Student training knobs
    parser.add_argument('--student-epochs', type=int, default=1)
    parser.add_argument('--student-max-samples', type=int, default=2000)
    parser.add_argument('--student-model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--student-label-smoothing', type=float, default=0.05)
    parser.add_argument('--student-early-stopping', type=int, default=2)
    args = parser.parse_args()

    setup_directories()
    config = load_config()

    if args.mode in ['train', 'full']:
        logger.info('Starting training pipeline (mock)')
        orchestrator = TrainingOrchestrator(config)
        orchestrator.run_full_pipeline()
        orchestrator.save_training_summary()

    if args.mode in ['inference', 'full']:
        logger.info('Starting inference pipeline (mock)')
        pipeline = InferencePipeline({
            'max_length': 1024,
            'tta_lengths': [1024, 2000],
            'quantization_bits': 8,
            'model_name': 'gemma2-9b',
            'use_tta': True
        })
        results = pipeline.run_inference(
            test_data_path=config['data_paths']['test_data'],
            merged_model_path='./model_save/final_merged_model',
            output_dir=args.output_dir
        )
        logger.info('Inference completed; submission saved to ./sub/final_submission.csv')

    if args.mode == 'student-train':
        logger.info('Starting real small student training (HF/Transformers)')
        train_csv_candidates = [
            './data/train.csv',
            './data/lmsys-chatbot-arena/train.csv'
        ]
        train_csv = next((p for p in train_csv_candidates if os.path.exists(p)), None)
        if not train_csv:
            raise FileNotFoundError('train.csv not found in ./data or ./data/lmsys-chatbot-arena')
        metrics = train_student(
            train_csv=train_csv,
            output_dir='./model_save/student_distilbert',
            max_samples=args.student_max_samples,
            num_epochs=args.student_epochs,
            model_name=args.student_model,
            label_smoothing=args.student_label_smoothing,
            early_stopping_patience=args.student_early_stopping,
        )
        logger.info(f'Student training done. Eval metrics: {metrics}')

    if args.mode == 'student-infer':
        logger.info('Running student inference (DistilBERT)')
        test_csv = config['data_paths']['test_data']
        out = infer_student(model_dir='./model_save/student_distilbert', test_csv=test_csv, submission_path='./sub/student_submission.csv')
        logger.info(f'Student submission saved to {out}')

    if args.mode == 'student-calibrate':
        logger.info('Calibrating student probabilities via temperature scaling')
        train_csv_candidates = [
            './data/train.csv',
            './data/lmsys-chatbot-arena/train.csv'
        ]
        train_csv = next((p for p in train_csv_candidates if os.path.exists(p)), None)
        if not train_csv:
            raise FileNotFoundError('train.csv not found in ./data or ./data/lmsys-chatbot-arena')
        info = calibrate_student(model_dir='./model_save/student_distilbert', train_csv=train_csv)
        logger.info(f'Calibration saved. Info: {info}')

    if args.mode == 'student-eval-holdout':
        logger.info('Evaluating pre/post calibration log loss on a distinct holdout split')
        train_csv_candidates = [
            './data/train.csv',
            './data/lmsys-chatbot-arena/train.csv'
        ]
        train_csv = next((p for p in train_csv_candidates if os.path.exists(p)), None)
        if not train_csv:
            raise FileNotFoundError('train.csv not found in ./data or ./data/lmsys-chatbot-arena')
        info = calibrate_student(model_dir='./model_save/student_distilbert', train_csv=train_csv)
        logger.info(
            f"Holdout NLL (before/after): {info.get('nll_holdout_before'):.6f} -> {info.get('nll_holdout_after'):.6f} (Δ {info.get('nll_holdout_improvement'):.6f})"
        )
        if 'logloss_holdout_before' in info:
            logger.info(
                f"Holdout LogLoss (before/after): {info.get('logloss_holdout_before'):.6f} -> {info.get('logloss_holdout_after'):.6f} (Δ {info.get('logloss_holdout_improvement'):.6f})"
            )


if __name__ == '__main__':
    main()
