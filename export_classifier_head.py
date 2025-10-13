import argparse
import json
import os
import torch
from transformers import AutoModelForSequenceClassification


def export_head(model_dir: str, out_path: str = None):
    # Try to read num_labels from config; default to 3
    cfg_path = os.path.join(model_dir, 'config.json')
    num_labels = 3
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            num_labels = int(cfg.get('num_labels', 3))
        except Exception:
            pass

    # Ensure model is instantiated with correct head size to avoid PEFT modules_to_save mismatch
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
    except RuntimeError as e:
        # Try to infer num_labels from adapter_model.safetensors if present
        try:
            from safetensors.torch import load_file as sf_load
            adapter_path = os.path.join(model_dir, 'adapter_model.safetensors')
            if os.path.exists(adapter_path):
                state = sf_load(adapter_path, device='cpu')
                # search for common classifier keys
                keys = [k for k in state.keys() if k.endswith('score.modules_to_save.default.weight') or k.endswith('classifier.modules_to_save.default.weight')]
                if keys:
                    w = state[keys[0]]
                    num_labels = int(w.shape[0])
        except Exception:
            pass
        # Retry with updated num_labels if changed, else re-raise
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
    # Try common attribute names
    head_module = None
    for name in ['classifier', 'score', 'classification_head', 'lm_head']:
        if hasattr(model, name):
            head_module = getattr(model, name)
            break
    if head_module is None:
        # Try the last module as a heuristic
        head_module = list(model.children())[-1]
    # Normalize state_dict keys to plain weight/bias if possible
    state = head_module.state_dict()
    norm = {}
    if 'weight' in state:
        norm['weight'] = state['weight']
    elif 'modules_to_save.default.weight' in state:
        norm['weight'] = state['modules_to_save.default.weight']
    elif 'original_module.weight' in state:
        norm['weight'] = state['original_module.weight']
    if 'bias' in state:
        norm['bias'] = state['bias']
    elif 'modules_to_save.default.bias' in state:
        norm['bias'] = state['modules_to_save.default.bias']
    elif 'original_module.bias' in state:
        norm['bias'] = state['original_module.bias']
    # If bias missing entirely, create a zero bias for compatibility
    if 'weight' in norm and 'bias' not in norm:
        import torch as _torch
        norm['bias'] = _torch.zeros(norm['weight'].shape[0])
    state = norm or state
    out = out_path or os.path.join(model_dir, 'classifier_head.pt')
    torch.save(state, out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Export classifier head weights from a seq-classification model directory')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    path = export_head(args.model_dir, args.out)
    print({'saved': path})
