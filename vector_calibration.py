import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
import glob
import os

### CONFIGURATION ###
fold_paths = sorted(glob.glob("logits_fold*.npy"))
num_classes = 3  # A, B, tie

# Match each logits file with corresponding labels file
logits_files = [f for f in fold_paths if "logits" in f]
labels_files = [f.replace("logits", "labels") for f in logits_files]

# Load and concatenate all folds
all_logits = np.concatenate([np.load(f) for f in logits_files], axis=0)  # shape (N, 3)
all_labels = np.concatenate([np.load(f) for f in labels_files], axis=0)  # shape (N,)

assert all_logits.shape[0] == all_labels.shape[0], "Mismatch in number of logits and labels"

### LOSS FUNCTION FOR VECTOR SCALING ###
def vector_calibration_loss(params, logits, labels):
    a = params[:num_classes]
    b = params[num_classes:]
    logits_calib = logits * a + b  # broadcasted elementwise scaling + bias
    log_probs = logits_calib - logsumexp(logits_calib, axis=1, keepdims=True)
    nll = -log_probs[np.arange(len(labels)), labels].mean()
    return nll

### OPTIMIZATION ###
init_params = np.concatenate([np.ones(num_classes), np.zeros(num_classes)])  # [a_A, a_B, a_T, b_A, b_B, b_T]
result = minimize(
    vector_calibration_loss,
    init_params,
    args=(all_logits, all_labels),
    method="L-BFGS-B"
)

opt_a = result.x[:num_classes]
opt_b = result.x[num_classes:]

print("Optimal scale factors (a):", opt_a)
print("Optimal biases (b):", opt_b)
print("Final NLL:", result.fun)

### SAVE CALIBRATION PARAMS ###
os.makedirs("calibration", exist_ok=True)
np.savez("calibration/vector_scaling_params.npz", a=opt_a, b=opt_b)
print("Saved to calibration/vector_scaling_params.npz")

### OPTIONAL: FUNCTION TO APPLY CALIBRATION TO LOGITS ###
def apply_vector_calibration(logits, a, b):
    logits_calib = logits * a + b
    probs = np.exp(logits_calib - logsumexp(logits_calib, axis=1, keepdims=True))
    return probs

# Example usage:
# calibrated_probs = apply_vector_calibration(raw_logits, opt_a, opt_b)
