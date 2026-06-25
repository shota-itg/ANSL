# data_gen/preprocess.py

import os
import pandas as pd
import numpy as np
import torch

from utils.config_loader import load_config, load_runtime


## CSV 読み込み
def load_csv_to_tensor(csv_path, num_traffic, N):
    # pandas で読み込み
    df = pd.read_csv(csv_path)

    # 入力データ（modality1とmodality2）とラベルを分割
    modality1_cols = [c for c in df.columns if c.startswith("data_")]
    modality2_cols = [c for c in df.columns if c.startswith("links_")]
    target_cols = [c for c in df.columns if c.startswith("target_")]

    modality1_np = df[modality1_cols].to_numpy(dtype=np.float32)
    modality2_np = df[modality2_cols].to_numpy(dtype=np.float32)
    labels_np = df[target_cols].to_numpy(dtype=np.int64)

    # numpy -> Tensor
    modality1 = torch.tensor(modality1_np, dtype=torch.float32)
    modality2 = torch.tensor(modality2_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.int64)

    # reshape
    labels = labels.reshape(len(labels), num_traffic, N, (N+1))
    labels = torch.argmax(labels, 3)    # shape[len(labels), num_traffic, N]
    # labels = labels.reshape(len(labels), -1) # shape[len(labels), num_traffic *N]

    return modality1, modality2, labels


def preprocess_data(data_name, num_traffic, lf_enabled, k, num_data):
    config = load_config()
    topo_name = config["topology"]["name"]
    nodes = config["topology"]["nodes"]


    num_failure = k

    # CSV ファイルのパスを取得
    data_path_cfg = config["paths"]["data"]
    lf_data_path_cfg = config["paths"]["lf_data"]

    if data_name == "train":
        if lf_enabled:
            data_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                lf_data_path_cfg["train_dir"], 
                lf_data_path_cfg["filename"]["lf_data_csv"].format(numt=num_traffic, k=num_failure, num=num_data)
            )
        else:
            data_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                data_path_cfg["train_dir"], 
                data_path_cfg["filename"]["data_csv"].format(numt=num_traffic, num=num_data)
            )
    else:
        if lf_enabled:
            data_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                lf_data_path_cfg["test_dir"], 
                lf_data_path_cfg["filename"]["lf_data_csv"].format(numt=num_traffic, k=num_failure, num=num_data)
            )
        else:
            data_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                data_path_cfg["test_dir"],  
                data_path_cfg["filename"]["data_csv"].format(numt=num_traffic, num=num_data)
            )

    runtime_cfg = load_runtime()
    num_train_data = runtime_cfg["data"]["num_train_data"]
    num_test_data = runtime_cfg["data"]["num_test_data"]

    # パラメータ
    N = len(nodes)


    # データ変換
    modality1, modality2, labels = load_csv_to_tensor(data_path, num_traffic, N)

    print(f'modality1={modality1.shape}, modality2={modality2.shape}, labels={labels.shape}')


    ## 保存
    datasets_path_cfg = config["paths"]["datasets"]
    lf_datasets_path_cfg = config["paths"]["lf_datasets"]
    if data_name == "train":
        if lf_enabled:
            modality1_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                lf_datasets_path_cfg["train_dir"], 
                lf_datasets_path_cfg["filename"]["lf_modality1_pt"].format(numt=num_traffic, k=num_failure, num=num_train_data)
            )
            modality2_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                lf_datasets_path_cfg["train_dir"], 
                lf_datasets_path_cfg["filename"]["lf_modality2_pt"].format(numt=num_traffic, k=num_failure, num=num_train_data)
            )
            labels_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                lf_datasets_path_cfg["train_dir"], 
                lf_datasets_path_cfg["filename"]["lf_labels_pt"].format(numt=num_traffic, k=num_failure, num=num_train_data)
            )
        else:
            modality1_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                datasets_path_cfg["train_dir"], 
                datasets_path_cfg["filename"]["modality1_pt"].format(numt=num_traffic, num=num_train_data)
            )
            modality2_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                datasets_path_cfg["train_dir"], 
                datasets_path_cfg["filename"]["modality2_pt"].format(numt=num_traffic, num=num_train_data)
            )
            labels_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                datasets_path_cfg["train_dir"], 
                datasets_path_cfg["filename"]["labels_pt"].format(numt=num_traffic, num=num_train_data)
            )
    else:
        if lf_enabled:
            modality1_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                lf_datasets_path_cfg["test_dir"], 
                lf_datasets_path_cfg["filename"]["lf_modality1_pt"].format(numt=num_traffic, k=num_failure, num=num_test_data)
            )
            modality2_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                lf_datasets_path_cfg["test_dir"], 
                lf_datasets_path_cfg["filename"]["lf_modality2_pt"].format(numt=num_traffic, k=num_failure, num=num_test_data)
            )
            labels_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                lf_datasets_path_cfg["test_dir"], 
                lf_datasets_path_cfg["filename"]["lf_labels_pt"].format(numt=num_traffic, k=num_failure, num=num_test_data)
            )
        else:
            modality1_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                datasets_path_cfg["test_dir"], 
                datasets_path_cfg["filename"]["modality1_pt"].format(numt=num_traffic, num=num_test_data)
            )
            modality2_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                datasets_path_cfg["test_dir"], 
                datasets_path_cfg["filename"]["modality2_pt"].format(numt=num_traffic, num=num_test_data)
            )
            labels_path = os.path.join(
                config["paths"]["results"]["root_dir"], 
                topo_name, 
                datasets_path_cfg["test_dir"], 
                datasets_path_cfg["filename"]["labels_pt"].format(numt=num_traffic, num=num_test_data)
            )


    torch.save(modality1, modality1_path)
    torch.save(modality2, modality2_path)
    torch.save(labels, labels_path)

    print(f'Tensor データを保存: {modality1_path}, {modality2_path}, {labels_path}')


if __name__ == "__main__":
    num_failure = None
    while True:
        data_name = input("Chose 'train' or 'test': ").strip().lower()
        if data_name in ("train", "test"):
            break
        else:
            print("\nError: 'train' か 'test' を入力してください。")
    if data_name == "train":
        while True:
            try:
                num_traffic = int(input("再経路の対象トラフィック数: "))
                break
            except ValueError:
                print("\nError: 整数を入力してください。")
        while True:
            lf_enabled = input("リンク障害の有無 'true' or 'false': ")
            if lf_enabled in ("true", "false"):
                lf_enabled = (lf_enabled == "true")
                break
            else:
                print("\nError: 'true' か 'false' を入力してください。")
        if lf_enabled:
            while True:
                try:
                    num_failure = int(input("リンク障害数: "))
                    break
                except ValueError:
                    print("\nError: 整数を入力してください。")
        while True:
            try:
                num_train_data = int(input("How many data?: "))
                break
            except ValueError:
                print("\nError: 整数を入力してください。")
        num_data = num_train_data
    else:
        while True:
            try:
                num_traffic = int(input("再経路の対象トラフィック数: "))
                break
            except ValueError:
                print("\nError: 整数を入力してください。")
        while True:
            lf_enabled = input("リンク障害の有無 'true' or 'false': ")
            if lf_enabled in ("true", "false"):
                lf_enabled = (lf_enabled == "true")
                break
            else:
                print("\nError: 'true' か 'false' を入力してください。")
        if lf_enabled:
            while True:
                try:
                    num_failure = int(input("リンク障害数: "))
                    break
                except ValueError:
                    print("\nError: 整数を入力してください。")
        while True:
            try:
                num_test_data = int(input("How many data?: "))
                break
            except ValueError:
                print("\nError: 整数を入力してください。")
        num_data = num_test_data
    k = num_failure

    preprocess_data(data_name, num_traffic, lf_enabled, k, num_data)