# scripts/inference.py

import os
import time
import math
import torch
import torch.nn.functional as F
import csv
import json

from model.fully_net import PartiallyConnectedLayerNet
from utils.device_selector import select_device
from utils.experiment_utils import parse_args, resolve_exp_dir, set_seed
from utils.config_loader import load_config, load_hyperparameter
from utils.metrics import demand_accuracy, path_accuracy, element_accuracy


def main(device, exp_dir=None):
    config = load_config(exp_dir)
    if config.get("seed") is not None:
        set_seed(config.get("seed"))
    nodes = config["topology"]["nodes"]
    topo_name = config["topology"]["name"]

    num_test_data = config["test"]["num_test_data"]
    datasets_path_cfg = config["paths"]["datasets"]
    test_data_pt = os.path.join(
        config["paths"]["results"]["root_dir"], 
        topo_name, 
        datasets_path_cfg["test_dir"], 
        datasets_path_cfg["filename"]["data_pt"].format(num=num_test_data)
    )
    test_labels_pt = os.path.join(
        config["paths"]["results"]["root_dir"], 
        topo_name, 
        datasets_path_cfg["test_dir"], 
        datasets_path_cfg["filename"]["labels_pt"].format(num=num_test_data)
    )
    results_path_cfg = config["paths"]["results"]
    checkpoint_model_pth = results_path_cfg["filename"]["checkpoint_pth"]
    result_json = results_path_cfg["filename"]["result_json"]
    output_csv = results_path_cfg["filename"]["output_csv"]

    hparam_cfg = load_hyperparameter(exp_dir)
    # inference_batch_size = hparam_cfg["optimization"]["inference_batch_size"]

    inputs_dim = hparam_cfg["architecture"]["inputs_dim"]
    hidden_dim = hparam_cfg["architecture"]["hidden_dim"]
    hidden_depth = hparam_cfg["architecture"]["hidden_depth"]
    outputs_dim = hparam_cfg["architecture"]["outputs_dim"]

    # パラメータ
    N = len(nodes)
    NP_2 = math.perm(N, 2)
    inference_batch_size = 1

    # データ読み込み (.ptファイル)
    test_data = torch.load(test_data_pt)
    test_labels = torch.load(test_labels_pt)


    ## データセットの分割・シャッフル
    # TensorDatasetにまとめる
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=inference_batch_size, drop_last=True)


    ## モデルのロード
    model = PartiallyConnectedLayerNet(
        NP_2, 
        inputs_dim, 
        hidden_dim, 
        hidden_depth, 
        outputs_dim
    ).to(device)
    if exp_dir is not None:
        checkpoint_model_path = os.path.join(exp_dir, checkpoint_model_pth)
    else:
        checkpoint_model_path = os.path.join(results_path_cfg["root_dir"], topo_name, "checkpoints", checkpoint_model_pth)
    model.load_state_dict(torch.load(checkpoint_model_path))
    model.eval()


    ## 評価
    test_demand_acc = demand_accuracy(test_loader, inference_batch_size, model, device, exp_dir)
    test_path_acc   = path_accuracy(test_loader, inference_batch_size, model, device, exp_dir)
    test_elem_acc   = element_accuracy(test_loader, inference_batch_size, model, device, exp_dir)

    print(f"Test Demand Accuracy: {test_demand_acc:.4f}")
    print(f"Test Path Accuracy:   {test_path_acc:.4f}")
    print(f"Test Element Accuracy:{test_elem_acc:.4f}")


    ## 推論結果を CSV に保存
    outputs_list = []
    inference_times = []

    start_time = time.time()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            # GPU 同期（前の処理の影響を消す）
            if device.type == "cuda":
                torch.cuda.synchronize()

            batch_start = time.time()

            outputs = model(data)
            outputs_reshaped = outputs.reshape(inference_batch_size, NP_2, N, (N+1))
            outputs_softmax  = F.softmax(outputs_reshaped, dim=3)
            outputs_argmax   = torch.argmax(outputs_softmax, dim=3)

            # GPU 同期（foward の時間を正確に測る）
            if device.type == "cuda":
                torch.cuda.synchronize()

            batch_end = time.time()
            inference_times.append(batch_end -batch_start)

            # outputs_list.append(outputs_argmax.cpu())
            outputs_flat = outputs_argmax.reshape(inference_batch_size, -1)
            outputs_list.append(outputs_flat.cpu())

    end_time = time.time()

    """
    outputs_tensor = torch.cat(outputs_list, dim=0)
    outputs_tensor = outputs_tensor.reshape(-1, outputs_tensor.shape[2])

    # CSV 出力
    with open("outputs.csv", "w", newline="") as f_out:
        writer = csv.writer(f_out)
        for row in outputs_tensor.numpy():
            writer.writerow(row)

    print("推論結果を outputs.csv に保存")
    """

    inference_time = end_time -start_time
    avrg_inference_time = sum(inference_times) /len(inference_times) /inference_batch_size
    print(f'Inference Time: {inference_time}')
    print(f'Average Inference Time: {avrg_inference_time}')


    ## 保存処理
    if exp_dir is not None:
        results_output_path = os.path.join(exp_dir, output_csv)
        results_json_path = os.path.join(exp_dir, result_json)
    else:
        results_output_path = os.path.join(
            results_path_cfg["root_dir"], 
            topo_name, 
            "outputs/", 
            output_csv
        )
        results_json_path = os.path.join(
            results_path_cfg["root_dir"], 
            topo_name, 
            "jsons/", 
            result_json
        )

    if os.path.exists(results_json_path):
        with open(results_json_path, "r") as f:
            overwrite_json = json.load(f)
    else:
        overwrite_json = {}

    overwrite_json["results"]["inference"]["inference_time"] = inference_time
    overwrite_json["results"]["inference"]["average_inference_time"] = avrg_inference_time
    overwrite_json["results"]["inference"]["num_test_datasets"] = num_test_data
    overwrite_json["results"]["inference"]["inference_batch_size"] = inference_batch_size
    overwrite_json["results"]["inference"]["accuracy"]["test_demand_accuracy"] = float(test_demand_acc)
    overwrite_json["results"]["inference"]["accuracy"]["test_path_accuracy"] = float(test_path_acc)
    overwrite_json["results"]["inference"]["accuracy"]["test_element_accuracy"] = float(test_elem_acc)

    with open(results_output_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        for row in torch.cat(outputs_list).numpy():
            writer.writerow(row)
            
    print(f'[Saved CSV]Inference results saved to: {results_output_path}')

    with open(results_json_path, "w") as f:
        json.dump(overwrite_json, f, indent=4)

    print(f'[Saved JSON] JSON results saved to: {results_json_path}')


if __name__ == "__main__":
    args = parse_args()
    exp_dir = resolve_exp_dir(args.exp_dir)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = select_device()

    main(device=device, exp_dir=exp_dir)