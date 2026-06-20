# evaluation/utils_eval.py

import networkx as nx


def build_topology_info(links):
    """
    Args: トポロジー情報(adjacency +capacity)を構築
        links: list of dicts with fields {u, v, capacity}
    Returns: 
    """
    adjacency_set = set()
    capacity_dict = {}

    for link in links:
        u = link["u"]
        v = link["v"]
        capa = link["capacity"]
        key = (min(u, v), max(u, v))
        adjacency_set.add(key)
        capacity_dict[key] = capa

    return adjacency_set, capacity_dict


def build_graph_from_topology(links):
    """
    Args: 
        links: list of dicts with field {u, v, capacity}
    Returns: 
        network.Graph
    """
    G = nx.Graph()
    for link in links:
        u = link["u"]
        v = link["v"]
        capa = link["capacity"]
        G.add_edge(u, v, capacity=capa)
    return G


def parse_demand(modality1, number_of_batch, number_of_traffic, N, num_traffic):
    modality1_reshaped = modality1[number_of_batch].reshape(num_traffic, (N+1))
    # nodes = vec[:-1]
    # src = int(torch.where(nodes == -1)[0])
    # # dst = int(torch.where(nodes == 1)[0])
    src = int((modality1_reshaped[number_of_traffic][:-1] == -1).nonzero(as_tuple=True)[0])
    dst = int((modality1_reshaped[number_of_traffic][:-1] == 1).nonzero(as_tuple=True)[0])
    bw  = float(modality1_reshaped[number_of_traffic][-1])
    return src, dst, bw


########################################################################################################
def build_initial_load_dict_from_modality2_2(modality2, number_of_batch, N, capacity_dict) -> dict:
    """
    Args: (M -num_traffic)分の全リンク利用量を取得（隣接行列用）
        modality2: 各リンクの利用量を示している
    Returns: 
        dict: 
    """
    modality2_reshaped = modality2[number_of_batch].reshape(N, N)
    topology_link = modality2_reshaped
    initial_load_dict = {}
    for u in range(N):
        for v in range(N):
            if u < v:
                md2_load = topology_link[u, v].item()
                if 0 <= md2_load: # 初期負荷が存在（保険）
                    key = (min(u, v), max(u, v))
                    initial_load_dict[key] = md2_load *capacity_dict[key]
                elif md2_load == -2: # そもそもリンクがない場合
                    initial_load_dict[key] = -2
                else: # 障害リンクの場合
                    initial_load_dict[key] = -1

    return initial_load_dict


def build_initial_load_dict_from_modality2(modality2, number_of_batch, capacity_dict) -> dict:
    """
    Args: （リンク数用）
    """
    topology_link = modality2[number_of_batch]
    initial_load_dict = {}
    for i, key in enumerate(capacity_dict):
        if topology_link[i] != -1:
            initial_load_dict[key] = topology_link[i] *capacity_dict[key]
        else:
            initial_load_dict[key] = topology_link[i]

    return initial_load_dict
########################################################################################################



def decode_paths(pred_logits, nohop_index) -> list:
    """
    Args: モデルの出力をノード列にでコード
        pred_logits: Tensor [num_demands, max_hops, num_nodes+1]
        nohop_index: noHopを表す数値（6ノードトポロジの場合は6）
    Returns: 
        list of paths (each path is list of node indices)
    """
    paths = []
    for hops in pred_logits:
        path = []
        for idx in hops:
            idx = int(idx)
            if idx == nohop_index:
                break
            path.append(idx)
        paths.append(path)
    return paths


def is_reachable_path_dict(path, src, dst, current_adj) -> bool:
    """
    Args: 到達可能チェック
        path: 
        src: 
        dst: 
        adjacency_set: 
    Returns: 
        bool: 到達可能であれば True
    """
    # 経路が空
    if len(path) == 0:
        return False

    # 始点・終点チェック
    if path[0] != src or path[-1] != dst:
        return False

    if len(path) != len(set(path)):
        return False

    # 到達可能チェック
    for u, v in zip(path[:-1], path[1:]):
        key = (min(u, v), max(u, v))
        if key not in current_adj:
            return False
    
    return True


def is_reachable_path_nx(path, src, dst, graph) -> bool:
    """
    Args: 到達可能チェック
        path: 
        src: 
        dst: 
        graph: 
    Returns: 
        bool: 到達可能であれば True
    """
    # 経路が空
    if len(path) == 0:
        return False

    # 始点・終点チェック
    if path[0] != src or path[-1] != dst:
        return False

    if len(path) != len(set(path)):
        return False

    # 到達可能チェック
    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            return False
    
    return True


def check_congestion_dict(path, bw, current_load_dict, current_capacity_dict) -> bool:
    """
    Args: 輻輳チェック（リンク負荷を計算）
        paths: list of paths
        demands: list of bandwidths
        capacity_dict: {(u, v): capacity}
        
    Returns:
        bool: 輻輳していなければ True
    """
    """
    # 初期負荷で初期化
    link_load = {
        edge: initial_load_dict.get(edge, 0.0) for edge in capacity_dict.keys()
    }

    for u, v in zip(path[:-1], path[1:]):
        key = (min(u, v), max(u, v))
        current_load_dict[key] += bw

    # 経路ごとに負荷を加算
    for path, (_, _, bw) in zip(paths, demands):
        for u, v in zip(path[:-1], path[1:]):
            key = (min(u, v), max(u, v)) # 無向（双方向）リンク
            link_load[key] += bw # 要求帯域幅bwを加算    

    # 輻輳チェック
    for key, load in current_load_dict.items():
        if capacity_dict[key] < load:
            # print(f'輻輳 --> load: {load}')
            return False
    """

    update_load_dict = current_load_dict.copy()
        
    for u, v in zip(path[:-1], path[1:]):
        key = (min(u, v), max(u, v))
        if update_load_dict[key] +bw <= current_capacity_dict[key]:
            update_load_dict[key] += bw
            continue
        else:
            return False, current_load_dict
        
    return True, update_load_dict


def check_congestion_nx(paths, demands, G) -> bool:
    """
    Args: 輻輳チェック（リンク負荷を計算）（networks利用）
    Returns: 
    """
    # 初期化（load を0 に）
    for u, v in G.edges():
        G[u][v]["load"] = G[u][v].get("initial_load", 0.0)

    # 経路ごとに負荷を計算
    for path, (_, _, bw) in zip(paths, demands):
        for u, v in zip(path[:-1], path[1:]):
            G[u][v]["load"] += bw

    # 輻輳チェック
    for u, v in G.edges():
        if G[u][v]["capacity"] < G[u][v]["load"]:
            return False
        
    return True