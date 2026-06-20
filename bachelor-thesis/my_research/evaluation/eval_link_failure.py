# evaluation/eval_link_failure.py

import os
import csv
import json
import random
import torch
import torch.nn.functional as F

from model.fully_net import FusionModel

from utils.experiment_utils import parse_args, resolve_exp_dir, set_seed
from utils.device_selector import select_device
from utils.config_loader import load_config, load_hyperparameter
from evaluation.utils_eval import (
    build_graph_from_topology, 
    build_topology_info, 
    parse_demand, 
    build_initial_load_dict_from_modality2, 
    decode_paths,
    is_reachable_path_dict, 
    is_reachable_path_nx, 
    check_congestion_dict, 
    check_congestion_nx
)


def eval_link_failure(
        testloader, inference_batch_size, model, device, 
        config, hparam_cfg, 
        num_trials=2
) -> float:
    """
    Args: 
        model: 学習済みモデル
        dataloader: 推論に使ったデータセット（デマンド集合単位）
        config: config.yaml
    Returns: 

    """
    links = config["topology"]["links"]
    N = len(config["topology"]["nodes"])
    max_random_failure = config["train"]["max_random_failure"]
    num_traffic = hparam_cfg["architecture"]["num_traffic"]


    adjacency_set, capacity_dict = build_topology_info(links)
    # debug #
    print(f'adjacency_set: {adjacency_set}')
    
    link_failure_results = [] # [(k, max, min, avg), ...]

    model.eval()
    for k in range(max_random_failure +1):
        trial_success_rates = []
        
        for _ in range(num_trials):
            success_count, total = 0, 0
            reachable_count, congestion_count = 0, 0

            for modality1, modality2, labels in testloader:
                modality1, modality2, labels = modality1.to(device), modality2.to(device), labels.to(device)

                with torch.no_grad():
                    outputs = model(modality1, modality2)

                outputs_reshaped = outputs.reshape(inference_batch_size, num_traffic, N, (N+1))
                outputs_softmax  = F.softmax(outputs_reshaped, dim=3)
                outputs_argmax   = torch.argmax(outputs_softmax, dim=3) # shape[inference_batch_size, num_traffic, N]

                # 1batchごとに評価（1デマンド集合）
                for i in range(inference_batch_size):
#################################################################################################################################
                    # initial_load_dict = build_initial_load_dict_from_modality2(modality2, i, N, capacity_dict) # 隣接行列用
                    initial_load_dict = build_initial_load_dict_from_modality2(modality2, i, capacity_dict) # リンク数用
#################################################################################################################################
                    current_adj = adjacency_set.copy()
                    current_capacity_dict = capacity_dict.copy()
                    current_load_dict = initial_load_dict

                    demands = []
                    for j in range(num_traffic):
                        src, dst, bw = parse_demand(modality1, i, j, N, num_traffic)
                        demands.append((src, dst, bw))

                    paths = decode_paths(outputs_argmax[i], N) # shape[num_traffic, ?] ?はデマンドごとに異なる経路長のため
            
                    # k個の障害リンクをランダムに選択
                    failed_links = random.sample(links, k)
                    for link in failed_links:
                        u = link["u"]
                        v = link["v"]
                        key = (min(u, v), max(u, v))
                        current_capacity_dict[key] = -1
                        current_load_dict[key] = -1
                        if key in current_adj:
                            current_adj.remove(key)

                    # 到達可能性と輻輳の評価
                    for path, (src, dst, bw) in zip(paths, demands):
                        total += 1
                        if is_reachable_path_dict(path, src, dst, current_adj):
                            reachable_count += 1
                            congestion_bool, current_load_dict = check_congestion_dict(path, bw, current_load_dict, current_capacity_dict)
                            if congestion_bool:
                                success_count += 1
                            else:
                                congestion_count += 1

            success_rate = success_count /total
            reachable_rate = reachable_count /total
            congestion_rate = congestion_count /total
            
            trial_success_rates.append(success_rate)

        max_success_rate = max(trial_success_rates)
        min_success_rate = min(trial_success_rates)
        average_success_rate = sum(trial_success_rates) /len(trial_success_rates)
        link_failure_results.append((k, max_success_rate *100, min_success_rate *100, average_success_rate *100))

        print(f'[Number of Link Failure: {k}] max={max_success_rate *100:.4f}, min={min_success_rate *100:.4f}, average={average_success_rate *100:.4f}, reachable_rate={reachable_rate *100:.4f}, congestion_rate={congestion_rate *100:.4f}')

    return link_failure_results


def main(device, exp_dir=None):
    config = load_config(exp_dir)
    if config.get("seed") is not None:
        set_seed(config.get("seed"))
    topo_name = config["topology"]["name"]
    num_test_data = config["test"]["num_test_data"]
    
    hparam_cfg = load_hyperparameter(exp_dir)
    inference_batch_size = hparam_cfg["optimization"]["inference_batch_size"]
    num_traffic = hparam_cfg["architecture"]["num_traffic"]

    datasets_path_cfg = config["paths"]["datasets"]
    test_modality1_pt = os.path.join(
        config["paths"]["results"]["root_dir"], 
        topo_name, 
        datasets_path_cfg["test_dir"], 
        datasets_path_cfg["filename"]["modality1_pt"].format(numt=num_traffic, num=num_test_data)
    )
    test_modality2_pt = os.path.join(
        config["paths"]["results"]["root_dir"], 
        topo_name, 
        datasets_path_cfg["test_dir"], 
        datasets_path_cfg["filename"]["modality2_pt"].format(numt=num_traffic, num=num_test_data)
    )
    test_labels_pt = os.path.join(
        config["paths"]["results"]["root_dir"], 
        topo_name, 
        datasets_path_cfg["test_dir"], 
        datasets_path_cfg["filename"]["labels_pt"].format(numt=num_traffic, num=num_test_data)
    )
    
    results_path_cfg = config["paths"]["results"]
    checkpoint_model_pth = results_path_cfg["filename"]["checkpoint_pth"]
    result_json = results_path_cfg["filename"]["result_json"]


    # データ読み込み (.ptファイル)
    test_modality1 = torch.load(test_modality1_pt)
    test_modality2 = torch.load(test_modality2_pt)
    test_labels = torch.load(test_labels_pt)


    ## データセットの分割・シャッフル
    # TensorDatasetにまとめる
    test_dataset = torch.utils.data.TensorDataset(test_modality1, test_modality2, test_labels)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=inference_batch_size, drop_last=True)


    ## モデルのロード
    model = FusionModel(exp_dir).to(device)
    if exp_dir is not None:
        checkpoint_model_path = os.path.join(exp_dir, checkpoint_model_pth)
    else:
        checkpoint_model_path = os.path.join(results_path_cfg["root_dir"], topo_name, "checkpoints", checkpoint_model_pth)
    model.load_state_dict(torch.load(checkpoint_model_path))

    link_failure_results = eval_link_failure(test_loader, inference_batch_size, model, device, config, hparam_cfg)


    ## 保存処理
    if exp_dir is not None:
        results_json_path = os.path.join(exp_dir, result_json)
        results_csv_path = os.path.join(exp_dir, "link_failure_results.csv")
    else:
        results_json_path = os.path.join(
            results_path_cfg["root_dir"], 
            topo_name, 
            "jsons/", 
            result_json
        )
        results_csv_path = os.path.join(
            results_path_cfg["root_dir"], 
            topo_name, 
            "evaluation/"
            "link_failure/", 
            "based.csv"
        )

    if os.path.exists(results_json_path):
        with open(results_json_path, "r") as f:
            overwrite_json = json.load(f)
    else:
        overwrite_json = {}

    overwrite_json["results"]["link_failure_resilience"]["num_test_datasets"] = num_test_data
    overwrite_json["results"]["link_failure_resilience"]["inference_batch_size"] = inference_batch_size

    for (fail_k, max_rate, min_rate, avg_rate) in link_failure_results:
        overwrite_json["results"]["link_failure_resilience"][f"random_{fail_k}_link_failure"] = {
            "max_success_rate": max_rate,
            "min_success_rate": min_rate,
            "average_success_rate": avg_rate
        }

    with open(results_json_path, "w") as f:
        json.dump(overwrite_json, f, indent=4)

    print(f'[Saved JSON] link failure resilience rate saved to: {results_json_path}')

    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["failure_k", "max_success_rate", "min_success_rate", "avg_success_rate"])
        for (k, max_success_rate, min_success_rate, average_success_rate) in link_failure_results:
            writer.writerow([k, max_success_rate, min_success_rate, average_success_rate])

    print(f"[Saved CSV] {results_csv_path}")


if __name__=="__main__":
    args = parse_args()
    exp_dir = resolve_exp_dir(args.exp_dir)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = select_device()

    main(device=device, exp_dir=exp_dir)