# evaluation/routing_success_rate.py

import os
import json
import time
import statistics
import numpy as np
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


def eval_routing_success(
        testloader, inference_batch_size, model, device, 
        config, hparam_cfg
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
    num_traffic = hparam_cfg["architecture"]["num_traffic"]
    
    adjacency_set, capacity_dict = build_topology_info(links)
    # debug #
    print(f'adjacency_set: {adjacency_set}')
    
    batch_times = []
    success_count, total = 0, 0
    reachable_count, congestion_count = 0, 0

    links_log, mlu_log, std_log, cv_log = [], [], [], []

    model.eval()
    start_time = time.time()
    for modality1, modality2, labels in testloader:
        # GPU 同期（前の処理の影響を消す）
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        batch_start = time.time()
        modality1, modality2, labels = modality1.to(device), modality2.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(modality1, modality2)

        outputs_reshaped = outputs.reshape(inference_batch_size, num_traffic, N, (N+1))
        outputs_softmax  = F.softmax(outputs_reshaped, dim=3)
        outputs_argmax   = torch.argmax(outputs_softmax, dim=3) # shape[inference_batch_size, num_traffic, N]

        # GPU 同期（foward の時間を正確に測る）
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        batch_end = time.time()
        batch_times.append(batch_end -batch_start)

        # 1batchごとに評価
        for i in range(inference_batch_size):
#########################################################################################################################
            # initial_load_dict = build_initial_load_dict_from_modality2(modality2, i, N, capacity_dict) # 隣接行列用
            initial_load_dict = build_initial_load_dict_from_modality2(modality2, i, capacity_dict) # リンク数用
#########################################################################################################################
            current_load_dict = initial_load_dict

            all_checker = True
            links_list = []

            demands = []
            for j in range(num_traffic):
                src, dst, bw = parse_demand(modality1, i, j, N, num_traffic)
                demands.append((src, dst, bw))

            paths = decode_paths(outputs_argmax[i], N) # shape[num_traffic, ?] ?はデマンドごとに異なる経路長のため

            for path, (src, dst, bw) in zip(paths, demands):
                total += 1
                if is_reachable_path_dict(path, src, dst, adjacency_set):
                    reachable_count += 1
                    congestion_bool, current_load_dict = check_congestion_dict(path, bw, current_load_dict, capacity_dict)
                    if congestion_bool:
                        success_count += 1
                    else:
                        congestion_count += 1
                        all_checker = False
                else:
                    all_checker = False

            if all_checker:
                for key in current_load_dict:
                    links_list += [current_load_dict[key].item()]
                
                links_log.append(links_list)
                mlu_log.append(max(links_list))
                std_log.append(statistics.stdev(links_list))
                cv_log.append(statistics.stdev(links_list) /statistics.mean(links_list))

    success_rate = success_count /total
    reachable_rate = reachable_count /total
    congestion_rate = congestion_count /total

    if len(links_log) != 0:
        links_log = np.array(links_log)
        mlu_log = np.array(mlu_log)
        std_log = np.array(std_log)
        cv_log = np.array(cv_log)

        mlu_avg = np.mean(mlu_log)
        mlu_std = np.std(mlu_log)
        mlu_min = np.min(mlu_log)
        mlu_max = np.max(mlu_log)
        std_avg = np.mean(std_log)
        std_std = np.std(std_log)
        cv_avg = np.mean(cv_log)
        cv_std = np.std(cv_log)
    else:
        mlu_avg = 0
        mlu_std = 0
        mlu_min = 0
        mlu_max = 0
        std_avg = 0
        std_std = 0
        cv_avg = 0
        cv_std = 0
    
    end_time = time.time()
    routing_success_evaluation_time = end_time -start_time
    average_routing_success_evaluation_time = sum(batch_times) /len(batch_times) /inference_batch_size

    print(f'Routing Success Evaluation Time: {routing_success_evaluation_time}s')
    print(f'Average Routing Success Evaluation Time: {average_routing_success_evaluation_time}s')
    print(f'Routing Success Rate = {success_rate *100:.4f}, reachable_rate={reachable_rate *100:.4f}, congestion_rate={congestion_rate *100:.4f}')
    print(f'MLU avg: {mlu_avg}, MLU Std: {mlu_std}, MLU min: {mlu_min}, MLU max: {mlu_max}, STD avg: {std_avg}, STD Std: {std_std}, CV avg: {cv_avg}, CV Std: {cv_std}')

    return routing_success_evaluation_time, average_routing_success_evaluation_time, success_rate *100, mlu_avg, mlu_std, mlu_min, mlu_max, std_avg, std_std, cv_avg, cv_std


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

    routing_success_evaluation_time, average_routing_success_evaluation_time, eval_routing_success_rate, mlu_avg, mlu_std, mlu_min, mlu_max, std_avg, std_std, cv_avg, cv_std = eval_routing_success(test_loader, inference_batch_size, model, device, config, hparam_cfg)


    ## 保存処理
    if exp_dir is not None:
        results_json_path = os.path.join(exp_dir, result_json)
    else:
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

    overwrite_json["results"]["routing_success_rate"]["routing_success_evaluation_time"] = routing_success_evaluation_time
    overwrite_json["results"]["routing_success_rate"]["average_routing_success_evaluation_time"] = average_routing_success_evaluation_time
    overwrite_json["results"]["routing_success_rate"]["num_test_datasets"] = num_test_data
    overwrite_json["results"]["routing_success_rate"]["inference_batch_size"] = inference_batch_size
    overwrite_json["results"]["routing_success_rate"]["accuracy"]["routing_success_rate"] = eval_routing_success_rate

    overwrite_json["results"]["routing_success_rate"]["load_balance_metrics"]["maximum_link_utilization"]["average"] = float(mlu_avg)
    overwrite_json["results"]["routing_success_rate"]["load_balance_metrics"]["maximum_link_utilization"]["standard_deviation"] = float(mlu_std)
    overwrite_json["results"]["routing_success_rate"]["load_balance_metrics"]["maximum_link_utilization"]["min"] = float(mlu_min)
    overwrite_json["results"]["routing_success_rate"]["load_balance_metrics"]["maximum_link_utilization"]["max"] = float(mlu_max)
    overwrite_json["results"]["routing_success_rate"]["load_balance_metrics"]["standard_deviation"]["average"] = float(std_avg)
    overwrite_json["results"]["routing_success_rate"]["load_balance_metrics"]["standard_deviation"]["standard_deviation"] = float(std_std)
    overwrite_json["results"]["routing_success_rate"]["load_balance_metrics"]["coefficient_of_variation"]["average"] = float(cv_avg)
    overwrite_json["results"]["routing_success_rate"]["load_balance_metrics"]["coefficient_of_variation"]["standard_deviation"] = float(cv_std)

    with open(results_json_path, "w") as f:
        json.dump(overwrite_json, f, indent=4)

    print(f'[Saved JSON] routing success rate saved to: {results_json_path}')


if __name__=="__main__":
    args = parse_args()
    exp_dir = resolve_exp_dir(args.exp_dir)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = select_device()

    main(device=device, exp_dir=exp_dir)