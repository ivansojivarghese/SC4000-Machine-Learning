import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification


def export_head(model_dir: str, out_path: str = None):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # Try common attribute names
    head_module = None
    for name in ['classifier', 'score', 'classification_head', 'lm_head']:
        if hasattr(model, name):
            head_module = getattr(model, name)
            break
    if head_module is None:
        # Try the last module as a heuristic
        head_module = list(model.children())[-1]
    state = head_module.state_dict()
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
