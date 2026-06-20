# evaluation/eval_link_failure_stateaware2.py

import os
import csv
import json
import random
import torch
import torch.nn.functional as F

from model.fully_net import FusionModel

from utils.experiment_utils import parse_args, resolve_exp_dir
from utils.device_selector import select_device
from utils.config_loader import load_config, load_hyperparameter
from data_gen.network_core import build_network, generate_graph
from data_gen.dijkstra import dijkstra, get_links_from_path
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
    num_traffic = hparam_cfg["architecture"]["num_traffic"]

    max_failure = len(links)
    
    adjacency_set, capacity_dict = build_topology_info(links)
    # debug #
    print(f'adjacency_set: {adjacency_set}')
    
    link_failure_stateaware_results = [] # [(k, max, min, avg), ...]

    model.eval()
    for k in range(max_failure):
        trial_success_rates = []
        
        for _ in range(num_trials):
            success_count, total = 0, 0
            reachable_count, congestion_count = 0, 0

            for djk_traffic, modality1, modality2, labels in testloader:
                djk_traffic, modality1, modality2, labels = djk_traffic.to(device), modality1.to(device), modality2.to(device), labels.to(device)
                modify_modality2 = modality2.clone()

                failed_links_list = []
                for i in range(inference_batch_size):
                    # k個の障害リンクをランダムに選択
                    failed_links = random.sample(links, k)
                    failed_links_list.append(failed_links)
                    
                    djk_links = build_network(config)
                    for link in failed_links:
                        u = link["u"]
                        v = link["v"]
                        key = (min(u, v), max(u, v))
                        djk_links[key].max_capacity = -1

                    # Dijkstra法で計算するトラフィック分をDijkstra法で再計算
                    for j in range(len(djk_traffic[i])): 
                        graph = generate_graph(djk_links)
                        path = dijkstra(graph, djk_traffic[i, j, 0], djk_traffic[i, j, 1])

                        if not path:
                            continue

                        link_path = get_links_from_path(path, djk_links)
                        allocated_links = []
                        for link in link_path:
                            if link.try_allocate(djk_traffic[i, j, 2]):
                                allocated_links.append(link)

                        if len(allocated_links) == len(link_path):
                            continue
                        else:
                            for l in allocated_links:
                                l.release(djk_traffic[i, j, 2])

##########################################################################################################################
                    # modality2 を更新
                    """ 隣接行列用
                    link_set = {(min(l["u"], l["v"]), max(l["u"], l["v"])) for l in links}
                    for u in range(N):
                        for v in range(N):
                            if u == v:
                                modify_modality2[i, (u *N) +v] = -1
                                continue
                            key = (min(u, v), max(u, v))
                            if key in link_set:
                                if djk_links[key].max_capacity == 0:
                                    modify_modality2[i, (u *N) +v] = -1
                                    modify_modality2[i, (v *N) +u] = -1
                                else:
                                    modify_modality2[i, (u *N) +v] = djk_links[key].used /djk_links[key].max_capacity
                                    modify_modality2[i, (v *N) +u] = djk_links[key].used /djk_links[key].max_capacity
                            else:
                                modify_modality2[i, (u *N) +v] = -1
                                modify_modality2[i, (v *N) +u] = -1                 
                    """



                    """ リンク数用
                    
                    """
                    for j, link in enumerate(links):
                        u = link["u"]
                        v = link["v"]
                        key = (min(u, v), max(u, v))
                        if djk_links[key].max_capacity == -1:
                            modify_modality2[i, j] = -1
                        else:
                            modify_modality2[i, j] = djk_links[key].used /djk_links[key].max_capacity                    
##########################################################################################################################

                with torch.no_grad():
                    outputs = model(modality1, modify_modality2)

                    # debug #ormal_outputs = model(modality1, modality2)

                outputs_reshaped = outputs.reshape(inference_batch_size, num_traffic, N, (N+1))
                outputs_softmax  = F.softmax(outputs_reshaped, dim=3)
                outputs_argmax   = torch.argmax(outputs_softmax, dim=3) # shape[inference_batch_size, num_traffic, N]

                """ debug
                normal_outputs_reshaped = normal_outputs.reshape(inference_batch_size, num_traffic, N, (N+1))
                normal_outputs_softmax  = F.softmax(normal_outputs_reshaped, dim=3)
                normal_outputs_argmax   = torch.argmax(normal_outputs_softmax, dim=3)                
                """


                # 1batchごとに評価（1デマンド集合）
                for i in range(inference_batch_size):
                    """ debug
                    print(f'stateaware_outputs_argmax: {outputs_argmax[i]}')
                    print(f'normal_outptus_argmax: {normal_outputs_argmax[i]}\n')                    
                    """

##############################################################################################################################
                    # initial_load_dict = build_initial_load_dict_from_modality2(modify_modality2, i, N) # 隣接行列用
                    initial_load_dict = build_initial_load_dict_from_modality2(modality2, i, capacity_dict) # リンク数用
##############################################################################################################################

                    current_adj = adjacency_set.copy()
                    current_capacity_dict = capacity_dict.copy()
                    current_load_dict = initial_load_dict

                    demands = []
                    for j in range(num_traffic):
                        src, dst, bw = parse_demand(modality1, i, j, N, num_traffic)
                        demands.append((src, dst, bw))

                    paths = decode_paths(outputs_argmax[i], N) # shape[num_traffic, ?] ?はデマンドごとに異なる経路長のため

                    # k個の障害リンクをランダムに選択
                    for link in failed_links_list[i]:
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
        link_failure_stateaware_results.append((k, max_success_rate *100, min_success_rate *100, average_success_rate *100))

        print(f'[Number of Link Failure: {k}] max={max_success_rate *100:.4f}, min={min_success_rate *100:.4f}, average={average_success_rate *100:.4f}, reachable_rate={reachable_rate *100:.4f}, congestion_rate={congestion_rate *100:.4f}')

    return link_failure_stateaware_results


def main(device, exp_dir=None):
    config = load_config(exp_dir)
    topo_name = config["topology"]["name"]
    num_test_data = config["test"]["num_test_data"]
    datasets_path_cfg = config["paths"]["datasets"]
    test_dijkstra_pt = os.path.join(
        config["paths"]["results"]["root_dir"], 
        topo_name, 
        datasets_path_cfg["test_dir"], 
        datasets_path_cfg["filename"]["djk_traffic_pt"].format(num=num_test_data)
    )
    test_modality1_pt = os.path.join(
        config["paths"]["results"]["root_dir"], 
        topo_name, 
        datasets_path_cfg["test_dir"], 
        datasets_path_cfg["filename"]["modality1_pt"].format(num=num_test_data)
    )
    test_modality2_pt = os.path.join(
        config["paths"]["results"]["root_dir"], 
        topo_name, 
        datasets_path_cfg["test_dir"], 
        datasets_path_cfg["filename"]["modality2_pt"].format(num=num_test_data)
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

    hparam_cfg = load_hyperparameter(exp_dir)
    inference_batch_size = hparam_cfg["optimization"]["inference_batch_size"]

    # データ読み込み (.ptファイル)
    test_dijkstra = torch.load(test_dijkstra_pt)
    test_modality1 = torch.load(test_modality1_pt)
    test_modality2 = torch.load(test_modality2_pt)
    test_labels = torch.load(test_labels_pt)


    ## データセットの分割・シャッフル
    # TensorDatasetにまとめる
    test_dataset = torch.utils.data.TensorDataset(test_dijkstra, test_modality1, test_modality2, test_labels)
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
        results_csv_path = os.path.join(exp_dir, "link_failure_stateaware_results.csv")
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
            "stateaware.csv"
        )

    if os.path.exists(results_json_path):
        with open(results_json_path, "r") as f:
            overwrite_json = json.load(f)
    else:
        overwrite_json = {}

    overwrite_json["results"]["link_failure_stateaware_resilience"]["num_test_datasets"] = num_test_data
    overwrite_json["results"]["link_failure_stateaware_resilience"]["inference_batch_size"] = inference_batch_size

    for (fail_k, max_rate, min_rate, avg_rate) in link_failure_results:
        overwrite_json["results"]["link_failure_stateaware_resilience"][f"random_{fail_k}_link_failure"] = {
            "max_success_rate": max_rate,
            "min_success_rate": min_rate,
            "average_success_rate": avg_rate
        }

    with open(results_json_path, "w") as f:
        json.dump(overwrite_json, f, indent=4)

    print(f'[Saved JSON] link failure stateaware resilience rate saved to: {results_json_path}')


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