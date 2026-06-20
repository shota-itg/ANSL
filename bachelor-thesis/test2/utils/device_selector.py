# utils/device_selector.py

import os
import torch


def select_device():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        if num_gpus == 1:
                device = torch.device("cuda:0")
                print("Only one GPU detected. Using cuda:0")
                return device
        
        os.system("nvidia-smi")
        while True:
            gpu_id = input(f'Enter GPU ID (0-{num_gpus-1}): ')
            if gpu_id.strip() == "":
                gpu_id = "0"
            
            try:
                gpu_id_int = int(gpu_id)
            except ValueError:
                print("\n[再入力]")
                continue
            if 0 <= gpu_id_int < num_gpus:
                device = torch.device(f'cuda:{gpu_id_int}')
                break
            else:
                print(f'GPU {gpu_id_int} is not exist.')
    else:
        print(f'GPU is not available. ')
        device = torch.device("cpu")

    print(f'Using device: {device}')
    return device