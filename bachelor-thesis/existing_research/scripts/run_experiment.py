# scripts/run_experiment.py

import yaml
import datetime
import os
import shutil
import json
from pathlib import Path
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.device_selector import select_device
from utils.config_loader import load_config, load_hyperparameter, load_runtime, save_runtime, load_compare
from model.fully_net import PartiallyConnectedLayerNet
from utils.lr_range_test import lr_range_test

config = load_config()
compare_cfg = load_compare()
CONFIG_COMPARE_KEYS = compare_cfg["compare_keys"]


def compare_hyperparams(current, past):
    return current == past


def extract_relevant_config(cfg):
    return {key: cfg.get(key) for key in CONFIG_COMPARE_KEYS}


def compare_config_partial(current_cfg, past_cfg):
    cur = extract_relevant_config(current_cfg)
    pst = extract_relevant_config(past_cfg)
    return cur == pst


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_past_experiments(root=None):
    exps = []
    if not os.path.exists(root):
        return exps

    for exp_name in os.listdir(root):
        exp_dir = os.path.join(root, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        configs_path = config["paths"]["configs"]
        config_path = os.path.join(
            exp_dir, 
            configs_path["filename"]["config_yaml"]
        )
        hparam_path = os.path.join(
            exp_dir, 
            configs_path["filename"]["hparam_yaml"]
        )

        if os.path.exists(config_path) and os.path.exists(hparam_path):
            exps.append({
                "exp_dir": exp_dir,
                "config": load_yaml(config_path), 
                "hparam": load_yaml(hparam_path)
            })

    return exps


def main():
    config = load_config()
    topo_name = config["topology"]["name"]
    
    device = select_device()

    MAX_EXPERIMENTS = 3
    topo_root = os.path.join("experiments", topo_name)

    # 現在の設定
    current_configs = load_config()
    current_hparam = load_hyperparameter()

    # 現在のseed
    current_seed = current_configs.get("seed", None)

    # 過去の実験を読み込む
    past_exps = load_past_experiments(topo_root)

    # 過去の実験フォルダをすべて確認
    same_condition_exps = []
    for exp in past_exps:
        # 条件を比較
        if compare_hyperparams(current_hparam, exp["hparam"]) and compare_config_partial(current_configs, exp["config"]):
                same_condition_exps.append(exp)

    if 0 < len(same_condition_exps):
        if current_seed is None:
            print(f'[WARNING] {len(same_condition_exps)} experiment(s) with identical conditions already exist.')
            print(f'Previous experiment directory: {same_condition_exps}')
            while True:
                choice = input("Continue with the same conditions? (y/n): ").strip().lower()
                if choice in ("y", "n"):
                    break
                else:
                    print("\nError: 'y' か 'n'を入力してください。")
            if choice == "n":
                print("Experiment cancelled.")
                return
                
            if MAX_EXPERIMENTS <= len(same_condition_exps):
                # os.path.getmtime（最終アクセス時間）でソート
                same_condition_exps_sorted = sorted(
                    same_condition_exps, 
                    key=lambda e: os.path.getmtime(e["exp_dir"])
                )
                # 最も古いディレクトリを特定して削除（1つ削除して空きを作る）
                oldest_exp = same_condition_exps_sorted[0]["exp_dir"]
                shutil.rmtree(oldest_exp)
                print(f'[INFO] The maximum retention limit has been reached. Removing the oldest experiment: {oldest_exp}')
        else:
            same_seed_exps = [
                exp for exp in same_condition_exps
                if exp["config"].get("seed", None) == current_seed
            ]

            if 0 < len(same_seed_exps):
                print(f'[WARNING] Same condition AND same seed experiment exists:')
                for e in same_seed_exps:
                    print(" ", e["exp_dir"])

                while True:
                    choice = input("Overwrite this experiment? (y/n): ").strip().lower()
                    if choice in ("y", "n"):
                        break
                    else:
                        print("\nError: 'y' か 'n'を入力してください。")
                if choice == "n":
                    print("Experiment cancelled.")
                    return
                
                for e in same_seed_exps:
                    shutil.rmtree(e["exp_dir"])
                    print(f'[INFO] The same condition AND same seed experiment removed: {e["exp_dir"]}')

    exp_dir = None # 安全のために初期化
    try:
        # 実験フォルダ名を作成
        timestamp = datetime.datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        exp_dir = os.path.join("experiments", topo_name, timestamp)
        os.makedirs(exp_dir, exist_ok=True)
        print(f'exp_dir: {exp_dir}')

        # config.yaml と hyperparameter.yaml をコピー
        current_configs_path = current_configs["paths"]["configs"]
        current_config_path = os.path.join(
            current_configs_path["dir"], 
            current_configs_path["filename"]["config_yaml"]
        )
        current_hparam_path = os.path.join(
            current_configs_path["dir"], 
            current_configs_path["filename"]["hparam_yaml"]
        )
        shutil.copyfile(
            current_config_path, 
            Path(exp_dir)/current_configs_path["filename"]["config_yaml"]
        )
        shutil.copyfile(
            current_hparam_path, 
            Path(exp_dir)/current_configs_path["filename"]["hparam_yaml"]
        )
    

        config = load_config(exp_dir)
        if config["lrrt"]["enabled"]:
            ## LR Range Test
            num_train_data = config["train"]["num_train_data"]

            datasets_path_cfg = config["paths"]["datasets"]
            train_modality1_pt = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                datasets_path_cfg["train_dir"], 
                datasets_path_cfg["filename"]["modality1_pt"].format(num=num_train_data)
            )
            train_modality2_pt = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                datasets_path_cfg["train_dir"], 
                datasets_path_cfg["filename"]["modality2_pt"].format(num=num_train_data)
            )
            train_labels_pt = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                datasets_path_cfg["train_dir"], 
                datasets_path_cfg["filename"]["labels_pt"].format(num=num_train_data)
            )
            hparam_cfg = load_hyperparameter(exp_dir)
            train_batch_size = hparam_cfg["optimization"]["train_batch_size"]
            runtime_cfg = load_runtime()
            learning_rate = runtime_cfg["hyperparameter"]["learning_rate"]

            train_modality1 = torch.load(train_modality1_pt)
            train_modality2 = torch.load(train_modality2_pt)
            train_labels = torch.load(train_labels_pt)

            dataset = TensorDataset(train_modality1, train_modality2, train_labels)
            data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

            model = PartiallyConnectedLayerNet(exp_dir).to(device)
            criterion = nn.CrossEntropyLoss()
            lr_range_test(model, data_loader, criterion, device, exp_dir)
            learning_rate = float(input("Learning Rate: "))
            save_runtime(runtime_cfg, exp_dir)

        # runtime.yaml をコピー
        current_runtime_path = os.path.join(
            current_configs_path["dir"], 
            current_configs_path["filename"]["runtime_yaml"]
        )
        shutil.copyfile(
            current_runtime_path, 
            Path(exp_dir)/current_configs_path["filename"]["runtime_yaml"]
        )


        ## train → inference → visualize を順番に実行
        print("Running training script: python3 -m scripts.train")
        subprocess.run([
            "python3", "-m", "scripts.train", 
            "--device", str(device), 
            "--exp_dir", exp_dir
        ])
        print()
        print("Running visualizing script: python3 -m scripts.visualize")
        subprocess.run([
            "python3", "-m", "scripts.visualize", 
            "--device", str(device), 
            "--exp_dir", exp_dir
        ])
        print()
        print("Running inferencing scripts: python3 -m scripts.inference")
        subprocess.run([
            "python3", "-m", "scripts.inference", 
            "--device", str(device), 
            "--exp_dir", exp_dir
        ])
        print()
        print("Running evaluation scripts: python3 -m evaluation.routing_success_rate")
        subprocess.run([
            "python3", "-m", "evaluation.routing_success_rate", 
            "--device", str(device), 
            "--exp_dir", exp_dir
        ])
        print()

        results_path_cfg = config["paths"]["results"]
        result_json = results_path_cfg["filename"]["result_json"]
        results_json_path = os.path.join(exp_dir, result_json)
        if os.path.exists(results_json_path):
            with open(results_json_path, "r") as f:
                read_json = json.load(f)

            final_train_path_accuracy = read_json.get("results", {}).get("train", {}).get("accuracy", {}).get("final_train_path_accuracy", 0)
            final_val_path_accuracy = read_json.get("results", {}).get("train", {}).get("accuracy", {}).get("final_val_path_accuracy", 0)
            test_path_accuracy = read_json.get("results", {}).get("inference", {}).get("accuracy", {}).get("test_path_accuracy", 0)

            print(f'final_train_path_accuracy: {final_train_path_accuracy}')
            print(f'final_val_path_accuracy: {final_val_path_accuracy}')
            print(f'test_path_accuracy: {test_path_accuracy}')

        if 99 < final_train_path_accuracy and 99 < final_val_path_accuracy and 99 < test_path_accuracy:
            print("Running evaluation scripts: python3 -m evaluation.eval_link_failure")
            subprocess.run([
                "python3", "-m", "evaluation.eval_link_failure", 
                "--device", str(device), 
                "--exp_dir", exp_dir
            ])
            print()    

        print(f"Experiment completed: {exp_dir}")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Cleaning up experiment directory ... ")
        if exp_dir and os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
            print(f'Experiment directory removed: {exp_dir}.')
        print("Exiting safely.")

    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        print("Cleaning up experiment directory ... ")
        print(f'Experiment directory removed: {exp_dir}.')
        shutil.rmtree(exp_dir)
        raise

if __name__ == "__main__":
    main()