# utils/scheduler_factory.py

import torch.optim.lr_scheduler as lr_scheduler

from utils.hparam_map import SCHEDULER_MAP

def create_scheduler(scheduler_name, optimizer, cfg):
    if scheduler_name == "None":
        return None

    scheduler_class = SCHEDULER_MAP[scheduler_name]

    if scheduler_name == "StepLR":
        return scheduler_class(
            optimizer, 
            step_size=cfg["StepLR"].get("step_size", 10), 
            gamma=cfg["StepLR"].get("gamma", 0.1)
        )

    if scheduler_name == "CosineAnnealingLR":
        return scheduler_class(
            optimizer, 
            T_max=cfg["CosineAnnealingLR"].get("T_max", 10)
        )
        
    if scheduler_name == "ReduceLROnPlateau":
        return scheduler_class(
            optimizer, 
            patience=cfg["ReduceLROnPlateau"]["patience"], 
            factor=cfg["ReduceLROnPlateau"]["factor"]
        )

    return scheduler_class(optimizer)