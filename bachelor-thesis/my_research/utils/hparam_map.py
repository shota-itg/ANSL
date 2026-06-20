# utils/hparammap.py

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ReduceLROnPlateau
)

# Activation
ACTIVATION_MAP = {
    "ReLU": nn.ReLU, 
    "LeakyReLU": nn.LeakyReLU, 
    "GELU": nn.GELU, 
    "Sigmoid": nn.Sigmoid, 
    "Tanh": nn.Tanh, 
    "None": None
}

# Optimizer
OPTIMIZER_MAT = {
    "Adam": optim.Adamax, 
    "Adamax": optim.Adamax, 
    "AdamW": optim.AdamW, 
    "SGD": optim.SGD, 
    "RMSprop": optim.RMSprop
}

# Normalization
NOMALIZATION_MAP = {
    "BatchNorm": nn.BatchNorm1d, 
    "LayerNorm": nn.LayerNorm, 
    "None": None
}

# Scheduler
SCHEDULER_MAP = {
    "StepLR": StepLR, 
    "CosineAnnealingLR": CosineAnnealingLR, 
    "ReduceLROnPlateau": ReduceLROnPlateau, 
    "None": None
}