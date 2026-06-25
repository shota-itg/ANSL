# scripts/train.py

import os
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn

from model.fully_net import FusionModel

from utils.experiment_utils import parse_args, resolve_exp_dir, set_seed
from utils.device_selector import select_device
from utils.config_loader import load_config, load_hyperparameter, load_runtime, save_runtime
from utils.lr_range_test import lr_range_test
from utils.hparam_map import OPTIMIZER_MAT
from utils.scheduler_factory import create_scheduler
from utils.early_stopping import EarlyStopping
from utils.metrics import demand_accuracy, path_accuracy, element_accuracy
from utils.json_utils import define_json

from datetime import datetime


def main(device, exp_dir=None):
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config = load_config(exp_dir)
    if config.get("seed") is not None:
        set_seed(config.get("seed"))
    topo_name = config["topology"]["name"]
    nodes = config["topology"]["nodes"] # ノードのリスト
    
    train_lf_enabled = config["train"]["train_lf_enabled"]
    num_train_data = config["train"]["num_train_data"]
    max_random_failure = config["train"]["max_random_failure"]
    train_ratio = config["train"]["train_ratio"]
    val_ratio = config["train"]["val_ratio"]
    max_epoch = config["train"]["max_epoch"]

    results_path_cfg = config["paths"]["results"]
    checkpoint_model_path = results_path_cfg["filename"]["checkpoint_pth"]
    lr_log_pt = results_path_cfg["filename"]["lr_log_pt"]
    train_log_pt = results_path_cfg["filename"]["train_log_pt"]
    val_log_pt = results_path_cfg["filename"]["val_log_pt"]
    result_json = results_path_cfg["filename"]["result_json"]
    
    hparam_cfg = load_hyperparameter(exp_dir)
    if config["lrrt"]["enabled"]:
        runtime_cfg = load_runtime(exp_dir)
        learning_rate = float(runtime_cfg["hyperparameter"]["learning_rate"])
    else:
        learning_rate = float(hparam_cfg["optimization"]["learning_rate"])
    train_batch_size = hparam_cfg["optimization"]["train_batch_size"]
    optimizer_name = hparam_cfg["optimization"]["optimizer"]
    weight_decay = hparam_cfg["optimization"]["weight_decay"]

    num_traffic = hparam_cfg["architecture"]["num_traffic"]

    scheduler_name = hparam_cfg["schedule"]["lr_scheduler"]

    erly_stppg_patience = hparam_cfg["early_stopping"]["patience"]
    erly_stppg_verbose = hparam_cfg["early_stopping"]["verbose"]
    erly_stppg_delta = hparam_cfg["early_stopping"]["delta"]

    datasets_path_cfg = config["paths"]["datasets"]
    lf_datasets_path_cfg = config["paths"]["lf_datasets"]
    if train_lf_enabled:
        train_modality1_pt = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            lf_datasets_path_cfg["train_dir"], 
            lf_datasets_path_cfg["filename"]["lf_modality1_pt"].format(numt=num_traffic, k=max_random_failure, num=num_train_data)
        )
        train_modality2_pt = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            lf_datasets_path_cfg["train_dir"], 
            lf_datasets_path_cfg["filename"]["lf_modality2_pt"].format(numt=num_traffic, k=max_random_failure, num=num_train_data)
        )
        train_labels_pt = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            lf_datasets_path_cfg["train_dir"], 
            lf_datasets_path_cfg["filename"]["lf_labels_pt"].format(numt=num_traffic, k=max_random_failure, num=num_train_data)
        )
    else:
        train_modality1_pt = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            datasets_path_cfg["train_dir"], 
            datasets_path_cfg["filename"]["modality1_pt"].format(numt=num_traffic, num=num_train_data)
        )
        train_modality2_pt = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            datasets_path_cfg["train_dir"], 
            datasets_path_cfg["filename"]["modality2_pt"].format(numt=num_traffic, num=num_train_data)
        )
        train_labels_pt = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            datasets_path_cfg["train_dir"], 
            datasets_path_cfg["filename"]["labels_pt"].format(numt=num_traffic, num=num_train_data)
        )


    # パラメータ
    N = len(nodes)  # ノード数
    optimizer_class = OPTIMIZER_MAT[optimizer_name]

    # データ読み込み（.pt ファイル）
    train_modality1 = torch.load(train_modality1_pt)
    train_modality2 = torch.load(train_modality2_pt)
    train_labels = torch.load(train_labels_pt)


    ## データセットの分割・シャッフル
    # TensorDataset にまとめる
    dataset = TensorDataset(train_modality1, train_modality2, train_labels)

    # train/val に分割
    n_train, n_val = int(len(dataset) *train_ratio), int(len(dataset) *val_ratio)
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=train_batch_size, drop_last=True)


    if exp_dir is None and config["lrrt"]["enabled"]:
        model = FusionModel().to(device)
        criterion = nn.CrossEntropyLoss()
        lr_range_test(model, train_loader, criterion, device, exp_dir)
        learning_rate = float(input("Learning Rate: "))
        save_runtime(runtime_cfg)

    ## モデル・損失関数・最適化
    model = FusionModel(exp_dir).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
        )
    scheduler = create_scheduler(
        scheduler_name, 
        optimizer, 
        hparam_cfg["schedule"]
    )
    early_stopping = EarlyStopping(
        patience=erly_stppg_patience, 
        verbose=erly_stppg_verbose, 
        delta=erly_stppg_delta
        )

    # log
    epoch_times = []
    final_epoch = 0
    learning_rate_log = [] if scheduler else None
    train_log, val_log = {}, {}
    train_log['loss'], train_log['demand_accuracy'], train_log['path_accuracy'], train_log['element_accuracy'] = [], [], [], []
    val_log['loss'], val_log['demand_accuracy'], val_log['path_accuracy'], val_log['element_accuracy'] = [], [], [], []
    final_train_loss, final_val_loss = 0, 0


    ## 学習
    start_time = time.time()
    print(">>> Beginning Model Training")
    for epoch in range(max_epoch):
        # debug #print(f'>> epoch: {epoch}/{max_epoch}')

        # GPU 同期（前の処理の影響を消す）
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        epoch_start = time.time()
        
        # train
        model.train()
        now_train_loss, train_loss = 0, 0
        for modality1, modality2, labels in train_loader:
            modality1, modality2, labels = modality1.to(device), modality2.to(device), labels.to(device)

            outputs = model(modality1, modality2)
            # outputs_reshaped = outputs.reshape(-1, (N+1))
            outputs_reshaped = outputs.reshape(train_batch_size, num_traffic, N, (N+1))
            outputs_permuted = outputs_reshaped.permute(0, 3, 1, 2)

            # now_train_loss = criterion(outputs_reshaped, labels.view(-1))
            now_train_loss = criterion(outputs_permuted, labels)
            optimizer.zero_grad()
            now_train_loss.backward()
            optimizer.step()

            train_loss += now_train_loss

        final_train_loss = now_train_loss
        train_loss /= len(train_loader)

        # validate
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for modality1, modality2, labels in val_loader:
                modality1, modality2, labels = modality1.to(device), modality2.to(device), labels.to(device)

                outputs = model(modality1, modality2)
                # outputs_reshaped = outputs.reshape(-1, (N+1))
                outputs_reshaped = outputs.reshape(train_batch_size, num_traffic, N, (N+1))
                outputs_permuted = outputs_reshaped.permute(0, 3, 1, 2)

                # val_loss += criterion(outputs_reshaped, labels.view(-1))
                val_loss += criterion(outputs_permuted, labels)


            val_loss /= len(val_loader)
            final_val_loss = val_loss

        # ログ保存
        # train
        train_log['loss'].append(train_loss.item())
        train_demand_acc = demand_accuracy(train_loader, train_batch_size, model, device, exp_dir)
        train_log['demand_accuracy'].append(train_demand_acc)
        train_path_acc = path_accuracy(train_loader, train_batch_size, model, device, exp_dir)
        train_log['path_accuracy'].append(train_path_acc)
        train_elem_acc = element_accuracy(train_loader, train_batch_size, model, device, exp_dir)
        train_log['element_accuracy'].append(train_elem_acc)

        # val
        val_log['loss'].append(val_loss.item())
        val_demand_acc = demand_accuracy(val_loader, train_batch_size, model, device, exp_dir)
        val_log['demand_accuracy'].append(val_demand_acc)
        val_path_acc = path_accuracy(val_loader, train_batch_size, model, device, exp_dir)
        val_log['path_accuracy'].append(val_path_acc)
        val_elem_acc = element_accuracy(val_loader, train_batch_size, model, device, exp_dir)
        val_log['element_accuracy'].append(val_elem_acc)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        if scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            learning_rate_log.append(current_lr)
            if scheduler_name == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        final_epoch += 1

        # GPU 同期（foward の時間を正確に測る）
        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_end = time.time()
        epoch_times.append(epoch_end -epoch_start)

    print(">>> Model Training Finished")
    end_time = time.time()
    learning_time = end_time -start_time
    average_epoch_time = sum(epoch_times) /len(epoch_times)


    ## Imformation
    print("=== Imformation ===")
    print(f'Learning Time: {int(learning_time //60)}m{learning_time %60:.2f}s ({learning_time:.4f}s)')
    print(f'Average Epoch Time: {average_epoch_time}s')

    print("=== Parameters ===")
    print(f'Train Datasets Size: {num_train_data}')
    print(f'Max Epoch: {max_epoch}')

    print("=== Hyperparameters ===")
    print(f'Batch Size: {train_batch_size}')
    print(f'Initial Learning Rate: {learning_rate}')

    print("=== Final Results ===")
    print(f'Final Epoch: {final_epoch}')
    print(f'Final Train Loss: {final_train_loss}')
    print(f'Final Train Demand Accuracy: {train_demand_acc}')
    print(f'Final Train Path Accuracy: {train_path_acc}')
    print(f'Final Train Element Accuracy: {train_elem_acc}')
    print(f'Final Val Loss: {final_val_loss}')
    print(f'Final Val Demand Accuracy: {val_demand_acc}')
    print(f'Final Val Path Accuracy: {val_path_acc}')
    print(f'Final Val Element Accuracy: {val_elem_acc}')

    """
    ## Final Test Results
    test_demand_acc = demand_accuracy(test_loader, model, device)
    test_path_acc   = path_accuracy(test_loader, model, device)
    test_elem_acc   = element_accuracy(test_loader, model, device)

    print(">>> Final Test Results")
    print(f'Test Demand Accuracy: {test_demand_acc:.4f}')
    print(f'Test Path Accuracy:   {test_path_acc:.4f}')
    print(f'Test Element Accuracy:{test_elem_acc:.4f}')
    """
    

    ## 保存処理
    define_json(config, hparam_cfg, start_time, exp_dir)

    if exp_dir is not None:
        results_checkpoint_path = os.path.join(exp_dir, checkpoint_model_path)
        results_learning_rate_log_path = os.path.join(exp_dir, lr_log_pt) if scheduler else None
        results_train_log_path = os.path.join(exp_dir, train_log_pt)
        results_val_log_path = os.path.join(exp_dir, val_log_pt)
        results_json_path = os.path.join(exp_dir, result_json)
        torch.save(model.state_dict(), Path(exp_dir)/Path(checkpoint_model_path))
        torch.save(learning_rate_log, Path(exp_dir)/Path(lr_log_pt)) if scheduler else None
        torch.save(train_log, Path(exp_dir)/Path(train_log_pt))
        torch.save(val_log, Path(exp_dir)/Path(val_log_pt))
    else:
        results_checkpoint_path = os.path.join(
            results_path_cfg["root_dir"], 
            topo_name, 
            "checkpoints/", 
            checkpoint_model_path
        )
        results_learning_rate_log_path = os.path.join(
            results_path_cfg["root_dir"], 
            topo_name, 
            "lr_range_test/", 
            lr_log_pt
        ) if scheduler else None
        results_train_log_path = os.path.join(
            results_path_cfg["root_dir"], 
            topo_name, 
            "logs/", 
            train_log_pt
        )
        results_val_log_path = os.path.join(
            results_path_cfg["root_dir"], 
            topo_name, 
            "logs/", 
            val_log_pt
        )
        results_json_path = os.path.join(
            results_path_cfg["root_dir"], 
            topo_name, 
            "jsons/", 
            result_json
        )
        torch.save(model.state_dict(), results_checkpoint_path)
        torch.save(learning_rate_log, results_learning_rate_log_path) if scheduler else None
        torch.save(train_log, results_train_log_path)
        torch.save(val_log, results_val_log_path)


    with open(results_json_path, "r") as f:
        overwrite_json = json.load(f)

    if config["lrrt"]["enabled"]:
        overwrite_json["hyperparameters"]["optimization"]["learning_rate"] = learning_rate
    overwrite_json["results"]["train"]["learning_time"] = learning_time
    overwrite_json["results"]["train"]["average_epoch_time"] = average_epoch_time
    overwrite_json["results"]["train"]["final_epoch"] = final_epoch
    overwrite_json["results"]["train"]["loss"]["final_train_loss"] = final_train_loss.detach().item()
    overwrite_json["results"]["train"]["loss"]["final_val_loss"] = final_val_loss.detach().item()
    overwrite_json["results"]["train"]["accuracy"]["final_train_demand_accuracy"] = train_demand_acc
    overwrite_json["results"]["train"]["accuracy"]["final_train_path_accuracy"] = train_path_acc
    overwrite_json["results"]["train"]["accuracy"]["final_train_element_accuracy"] = train_elem_acc
    overwrite_json["results"]["train"]["accuracy"]["final_val_demand_accuracy"] = val_demand_acc
    overwrite_json["results"]["train"]["accuracy"]["final_val_path_accuracy"] = val_path_acc
    overwrite_json["results"]["train"]["accuracy"]["final_val_element_accuracy"] = val_elem_acc

    with open(results_json_path, "w") as f:
            json.dump(overwrite_json, f, indent=4)

    print(f'[Saved Checkpoint Model] Checkpoint model saved to: {results_checkpoint_path}')
    if scheduler is None:
        print(f'[Saved Logs] Logs saved to: {results_train_log_path}, {results_val_log_path}')
    else:
        print(f'[Saved Logs] Logs saved to: {results_learning_rate_log_path}, {results_train_log_path}, {results_val_log_path}')
    print(f'[Saved JSON Results] JSON results saved to: {results_json_path}')


if __name__ == "__main__":
    args = parse_args()
    exp_dir = resolve_exp_dir(args.exp_dir)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = select_device()
        
    main(device=device, exp_dir=exp_dir)