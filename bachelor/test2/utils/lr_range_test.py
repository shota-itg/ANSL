# utils/lr_range_test.py

import os
import torch
import matplotlib.pyplot as plt

from utils.config_loader import load_config, load_hyperparameter


## LR Range Test
def lr_range_test(model, loader, criterion, device, exp_dir=None):

    # config.yaml からパラメータを読み込む
    config = load_config(exp_dir)
    topo_name = config["topology"]["name"]
    nodes = config["topology"]["nodes"]
    lr_min = float(config["lrrt"]["lr_min"])
    lr_max = float(config["lrrt"]["lr_max"])
    # default_results_path = config["paths"]["results"]["dir"]
    num_train_data = config["train"]["num_train_data"]
    lr_range_test_png = config["paths"]["results"]["filename"]["lr_range_test_png"]
    hparam_cfg = load_hyperparameter(exp_dir)
    batch_size = hparam_cfg["optimization"]["train_batch_size"]
    num_traffic = hparam_cfg["architecture"]["num_traffic"]

    # パラメータ
    N = len(nodes)


    optimizer = torch.optim.Adamax(model.parameters(), lr=lr_min)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (10 ** (1/5)) ** step)

    losses = []
    lrs = []

    model.train()
    for step, (modality1, modality2, labels) in enumerate(loader):
        modality1, modality2, labels = modality1.to(device), modality2.to(device), labels.to(device)

        outputs = model(modality1, modality2)
        # outputs_reshaped = outputs.reshape(-1, (N+1))
        outputs_reshaped = outputs.reshape(batch_size, num_traffic, N, (N+1))
        outputs_permuted = outputs_reshaped.permute(0, 3, 1, 2)

        # loss = criterion(outputs_reshaped, labels.view(-1))
        loss = criterion(outputs_permuted, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])

        check_lr = optimizer.param_groups[0]['lr']
        if lr_max <= check_lr:
            break
        
        scheduler.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('LR Range Test')
    plt.suptitle(f'{num_train_data} train item + Batch Size {batch_size}')
    plt.grid(True)
    if exp_dir is not None:
        save_lrrt_path = os.path.join(
            exp_dir, 
            "lr_range_test.png"
        )
    else:
        save_lrrt_path = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            "lr_range_test/", 
            lr_range_test_png
        )
    plt.savefig(save_lrrt_path)
    plt.show()