

# Linkクラスは network_core.py からインポートされることを想定
# 例: from data_gen.network_core import Link 

def evaluate_route(links: dict, demand: tuple, route: list) -> tuple:
    """
    一つの経路がネットワークの基本制約（存在、到達、ループフリー）を満たすかチェックする。

    Args:
        links (dict): build_network()で生成されたトポロジー辞書 { (i, j): Linkオブジェクト }。
        demand (tuple): (source, destination, bandwidth)
        route (list): 学習モデルが出力したノードのリスト [n0, n1, ..., nk]。

    Returns:
        tuple: (is_valid, links_used_by_demand_with_bandwidth)
        links_used_by_demand_with_bandwidth は有効な場合 [((u, v), bandwidth), ...] のリスト
    """
    source, destination, bandwidth = demand
    
    # --- 1. 基本形式チェック ---
    if not route or len(route) < 2:
        return False, []

    # --- 5. 送信元/宛先チェック ---
    if route[0] != source or route[-1] != destination:
        return False, [] # 送信元/宛先が一致しない

    # --- 3. ループフリーチェック ---
    if len(route) != len(set(route)):
        return False, [] # 経路にループがある
        
    # --- 1. リンクの存在チェック & 2. 到達可能性チェック ---
    # 後の容量制約チェックのために、この要求が使用するリンクを記録
    links_used_by_demand = []
    
    for i in range(len(route) - 1):
        node_u = route[i]
        node_v = route[i+1]
        
        link_tuple = (node_u, node_v)
        
        # 1. リンクの存在を確認 (links辞書にキーが存在するか)
        if link_tuple not in links:
            return False, [] # リンクがトポロジーに存在しない
            
        # リンクが存在すれば、使用リストに追加
        # ((u, v), 帯域幅) の形式で記録
        links_used_by_demand.append((link_tuple, bandwidth))
        
    # 基本制約を満たしている
    return True, links_used_by_demand


# Linkクラスは network_core.py からインポートされることを想定

def check_all_capacity_constraints(links: dict, all_valid_links: list) -> bool:
    """
    全トラフィックの合計がリンク容量を超えていないか（輻輳がないか、式1b）をチェックする。

    Args:
        links (dict): build_network()で生成されたトポロジー辞書 { (i, j): Linkオブジェクト }。
        all_valid_links (list): すべての要求から集められた有効なリンク使用情報。
                                例: [((u1, v1), band1), ((u2, v2), band2), ...]

    Returns:
        bool: 輻輳が発生していなければ True
    """
    # リンクごとの合計使用帯域幅を計算
    link_utilization = {}
    
    for (u, v), bandwidth in all_valid_links:
        # リンク (u, v) の使用帯域幅を加算
        link_utilization[(u, v)] = link_utilization.get((u, v), 0.0) + bandwidth

    # --- 4. 容量制約チェック (式 1b) ---
    for (u, v), link_obj in links.items():
        # Linkオブジェクトから容量 C_{i, j} を取得
        capacity = link_obj.max_capacity
        
        # リンクの使用量をフェッチ (使用されていない場合は 0)
        used_band = link_utilization.get((u, v), 0.0)
        
        # 制約条件 (1b) のチェック: Sum(x_k) <= c_{i, j}
        if used_band > capacity:
            # 輻輳が発生している
            return False 

    # すべてのリンクで制約を満たした
    return True


def main():
    # data_genearate_data.py のループ内で使用するイメージ

    # --- 前提 ---
    # links = build_network() の返り値
    # all_demands = あるトラフィックセットに含まれる全要求 [(s1, d1, b1), (s2, d2, b2), ...]
    # all_proposed_routes = モデルが出力した全要求に対する経路 [route1, route2, ...]

    all_valid_links_and_bandwidth = []
    all_routes_valid_basic_constraints = True

    # 1. 各経路の基本制約（存在、到達、ループフリー）をチェック
    for demand, route in zip(all_demands, all_proposed_routes):
        is_valid, links_used = evaluate_route(links, demand, route)
    
        if not is_valid:
            # 基本制約のどれかが破られた時点で、このトラフィックセットの経路設計は失敗
            all_routes_valid_basic_constraints = False
            break
    
        # 有効な経路の使用リンク情報を集める
        all_valid_links_and_bandwidth.extend(links_used)

    # 2. 基本制約をすべて満たした場合のみ、容量制約（輻輳の有無）をチェック
    is_success = False
    if all_routes_valid_basic_constraints:
        # 評価式 (1b) のチェック
        is_capacity_safe = check_all_capacity_constraints(links, all_valid_links_and_bandwidth)
    
        if is_capacity_safe:
            # すべての要求が有効で、かつ輻輳がない場合のみ成功
            is_success = True

    # is_success を集計して、経路設計成功率を計算
    if is_success:
        # 成功回数をインクリメント
        success_counter += 1
    else:
        # 失敗回数をインクリメント (無効な経路 or 輻輳)
        failure_counter += 1


if __name__=="__main__":
    main()