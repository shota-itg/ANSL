# scripts/visualize.py

import os
import matplotlib.pyplot as plt
import torch

from utils.experiment_utils import parse_args, resolve_exp_dir
from utils.config_loader import load_config, load_hyperparameter


def plot_learning_curve(train_log, val_log, num_data, batch_size, save_path):
    plt.figure(figsize=(12, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_log['loss'], label='Train Loss')
    plt.plot(val_log['loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Path Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_log['path_accuracy'], label='Train Path Accuracy')
    plt.plot(val_log['path_accuracy'], label='Val Path Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Path Accuracy')
    plt.legend()

    plt.suptitle(f'{num_data} Train item + Batch Size {batch_size}')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()
    print(f'[Saved Figure] Learning curve saved to: {save_path}')


def plot_aux_learning_curve(train_log, val_log, num_data, batch_size, save_path):
    plt.figure(figsize=(12, 6))
    
    # Demand Accuracy
    plt.subplot(1, 4, 2)
    plt.plot(train_log['demand_accuracy'], label='Train Demand Accuracy')
    plt.plot(val_log['demand_accuracy'], label='Val Demand Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Demand Accuracy')
    plt.legend()

    # Element Accuracy
    plt.subplot(1, 4, 4)
    plt.plot(train_log['element_accuracy'], label='Train Element Accuracy')
    plt.plot(val_log['element_accuracy'], label='Val Element Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Element Accuracy')
    plt.legend()

    plt.suptitle(f'{num_data} Train item + Batch Size {batch_size}')

    plt.savefig(save_path)
    plt.show()
    print(f'[Saved Figure] Learning curve saved to: {save_path}')


def plot_lr_schedule(lr_log, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(lr_log)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)

    plt.savefig(save_path)
    print(f'[Saved Figure] Learning rate schedule saved to: {save_path}')


def main(exp_dir=None):
    config = load_config(exp_dir)
    topo_name = config["topology"]["name"]
    num_train_data = config["train"]["num_train_data"]
    results_path_cfg = config["paths"]["results"]
    lr_log_pt = results_path_cfg["filename"]["lr_log_pt"]
    train_log_pt = results_path_cfg["filename"]["train_log_pt"]
    val_log_pt = results_path_cfg["filename"]["val_log_pt"]
    
    hparam_cfg  = load_hyperparameter(exp_dir)
    train_batch_size = hparam_cfg["optimization"]["train_batch_size"]
    lr_scheduler = hparam_cfg["schedule"]["lr_scheduler"]

    if exp_dir is not None:
        train_log_path = os.path.join(exp_dir, train_log_pt)
        val_log_path = os.path.join(exp_dir, val_log_pt)
        lr_log_path = os.path.join(exp_dir, lr_log_pt)
        figure_root = exp_dir
    else:
        train_log_path = os.path.join(
            results_path_cfg, 
            topo_name, 
            "logs/", 
            train_log_pt
        )
        val_log_path = os.path.join(
            results_path_cfg, 
            topo_name, 
            "logs/", 
            val_log_pt
        )
        lr_log_path = os.path.join(
            results_path_cfg, 
            topo_name, 
            "logs/", 
            lr_log_pt
        )
        figure_root = os.path.join(config["paths"]["resutls"]["root_dir"], topo_name, "figures/")

    if not os.path.exists(train_log_path) or not os.path.exists(val_log_path):
        print("Error: train_log.pt or val_log.pt not found.")
        return

    train_log = torch.load(train_log_path)
    val_log = torch.load(val_log_path)

    fig_path = os.path.join(figure_root, "learning_curve.png")
    plot_learning_curve(train_log, val_log, num_train_data, train_batch_size, fig_path)

    aux_fig_path = os.path.join(figure_root, "aux_learning_curve.png")
    plot_aux_learning_curve(train_log, val_log, num_train_data, train_batch_size, aux_fig_path)

    if lr_scheduler is not None:
        if os.path.exists(lr_log_path):
            lr_log = torch.load(lr_log_path)
            schdlr_fig_path = os.path.join(figure_root, "learning_rate_schedule.png")
            plot_lr_schedule(lr_log, schdlr_fig_path)
            

if __name__ == "__main__":
    args = parse_args()
    exp_dir = resolve_exp_dir(args.exp_dir)
    
    main(exp_dir)