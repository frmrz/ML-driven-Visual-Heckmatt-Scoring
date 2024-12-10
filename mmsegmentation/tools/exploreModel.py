import os
import numpy as np
import torch

from mmseg.apis import init_model
from mmengine import Config

# Define the arguments directly in the notebook
config_file = '/media/francesco/DEV001/PROJECT-THYROID/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0.py'
checkpoint_file = '/media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/FSHD_KNET_SWIN_f0/iter_80000.pth'

# Load the configuration file
cfg = Config.fromfile(config_file)
cfg.model.pretrained = None
cfg.test_dataloader['batch_size'] = 1

# Construct the model and load checkpoint
model = init_model(cfg, checkpoint_file, device='cuda:0')
model.backbone()

# Stop here, the rest of the script is not needed
print("Model loaded successfully.")
