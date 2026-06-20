# data_gen/save_data.py

import os
import csv
import numpy as np

from utils.config_loader import load_config, load_runtime


## CSV保存
def save_data(data_name, data_log, target_log, data_one_hot_log, target_one_hot_log):

    config = load_config()
    topo_name = config["topology"]["name"]
    data_path_cfg = config["paths"]["data"]
    
    runtime_cfg = load_runtime()
    data_name = runtime_cfg["data"]["data_name"]
    num_train_data = runtime_cfg["data"]["num_train_data"]
    num_test_data = runtime_cfg["data"]["num_test_data"]

    if data_name == "train":
        save_traffic_log_path = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            data_path_cfg["train_dir"], 
            data_path_cfg["filename"]["traffic_log_csv"].format(num=num_train_data)
        )
        save_data_path = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            data_path_cfg["train_dir"], 
            data_path_cfg["filename"]["data_csv"].format(num=num_train_data)
        )
    else:
        save_traffic_log_path = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            data_path_cfg["test_dir"], 
            data_path_cfg["filename"]["traffic_log_csv"].format(num=num_test_data)
        )
        save_data_path = os.path.join(
            config["paths"]["results"]["root_dir"], 
            topo_name, 
            data_path_cfg["test_dir"], 
            data_path_cfg["filename"]["data_csv"].format(num=num_test_data)
        )

    with open(save_traffic_log_path, "w", newline="") as f:
        writer = csv.writer(f)

        # ヘッダーの自動生成（例: src0 ~ dst3, bw, path0 ~ pathN）
        data_len = len(data_log[0]) if data_log else 0
        target_len = len(target_log[0]) if target_log else 0
        header = [f"data_{i}" for i in range(data_len)] + [f"target_{i}" for i in range(target_len)]
        writer.writerow(header)

        # 各サンプルを 1 行にまとめて保存
        for d, t in zip(data_log, target_log):
            writer.writerow(d + t)

    with open(save_data_path, "w", newline="") as f:
        writer = csv.writer(f)

        # ヘッダーの自動生成（例: src0 ~ dst3, bw, path0 ~ pathN）
        data_len = len(data_one_hot_log[0]) if data_one_hot_log else 0
        target_len = len(target_one_hot_log[0]) if target_one_hot_log else 0
        header = [f"data_{i}" for i in range(data_len)] + [f"target_{i}" for i in range(target_len)]
        writer.writerow(header)

        # 各サンプルを 1 行にまとめて保存
        for d, t in zip(data_one_hot_log, target_one_hot_log):
            writer.writerow(d + t)