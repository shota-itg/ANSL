# data_gen/preprocess.py

import os
import math
import pandas as pd
import numpy as np
import torch

from utils.config_loader import load_config, load_runtime

config = load_config()
topo_name = config["topology"]["name"]
nodes = config["topology"]["nodes"]

runtime_cfg = load_runtime()
num_train_data = runtime_cfg["data"]["num_train_data"]
num_test_data = runtime_cfg["data"]["num_test_data"]

# CSV ファイルのパスを取得
data_path_cfg = config["paths"]["data"]
train_data_path = os.path.join(
    config["paths"]["results"]["root_dir"], 
    topo_name, 
    data_path_cfg["train_dir"], 
    data_path_cfg["filename"]["data_csv"].format(num=num_train_data)
)
test_data_path = os.path.join(
    config["paths"]["results"]["root_dir"], 
    topo_name, 
    data_path_cfg["test_dir"], 
    data_path_cfg["filename"]["data_csv"].format(num=num_test_data)
)

# パラメータ
N = len(nodes)
NP_2 = math.perm(N, 2)


## CSV 読み込み
def load_csv_to_tensor(csv_path):
    # pandas で読み込み
    df = pd.read_csv(csv_path)

    # 入力データ（modality1とmodality2）とラベルを分割
    data_cols = [c for c in df.columns if c.startswith("data_")]
    target_cols = [c for c in df.columns if c.startswith("target_")]

    data_np = df[data_cols].to_numpy(dtype=np.float32)
    labels_np = df[target_cols].to_numpy(dtype=np.int64)

    # numpy -> Tensor
    data = torch.tensor(data_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.int64)

    # reshape
    labels = labels.reshape(len(labels), NP_2, N, (N+1))
    labels = torch.argmax(labels, 3)    # shape[len(labels), num_traffic, N]
    # labels = labels.reshape(len(labels), -1) # shape[len(labels), num_traffic *N]

    return data, labels

# データ変換
train_data, train_labels = load_csv_to_tensor(train_data_path)
test_data, test_labels = load_csv_to_tensor(test_data_path)

print(f'train_data={train_data.shape}, train_labels={train_labels.shape}')
print(f'test_data={test_data.shape}, test_labels={test_labels.shape}')


## 保存
datasets_path_cfg = config["paths"]["datasets"]
train_data_path = os.path.join(
    config["paths"]["results"]["root_dir"], 
    topo_name, 
    datasets_path_cfg["train_dir"], 
    datasets_path_cfg["filename"]["data_pt"].format(num=num_train_data)
)
train_labels_path = os.path.join(
    config["paths"]["results"]["root_dir"], 
    topo_name, 
    datasets_path_cfg["train_dir"], 
    datasets_path_cfg["filename"]["labels_pt"].format(num=num_train_data)
)
test_data_path = os.path.join(
    config["paths"]["results"]["root_dir"], 
    topo_name, 
    datasets_path_cfg["test_dir"], 
    datasets_path_cfg["filename"]["data_pt"].format(num=num_test_data)
)
test_labels_path = os.path.join(
    config["paths"]["results"]["root_dir"], 
    topo_name, 
    datasets_path_cfg["test_dir"], 
    datasets_path_cfg["filename"]["labels_pt"].format(num=num_test_data)
)


torch.save(train_data, train_data_path)
torch.save(train_labels, train_labels_path)
torch.save(test_data, test_data_path)
torch.save(test_labels, test_labels_path)

print(f'Tensor データを保存: {train_data_path}, {train_labels_path}, {test_data_path}, {test_labels_path}')