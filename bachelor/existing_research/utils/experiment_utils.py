# utils/experiment_utils.py

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def resolve_exp_dir(arg_exp_dir):
    if arg_exp_dir is None:
        return arg_exp_dir

    exp_dir = arg_exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False