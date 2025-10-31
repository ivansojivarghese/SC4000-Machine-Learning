'''
import torch

print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
'''

# quick Python (run in your environment)
import numpy as np
import pyarrow.parquet as pq
oof = pq.read_table('model_save/teacher_logits/oof_probs.parquet').to_pandas()
mask = (oof['fold']==2) & (oof['model']=='llama')
print(mask.sum(), 'rows present for llama fold2')
# also inspect for NaNs
print('NaNs in probs:', oof.loc[mask, ['pA','pB','pTie']].isna().sum())