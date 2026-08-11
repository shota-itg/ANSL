"""
STEP 2 (前半): 固定 ECMP 経路の前計算
提案手法（TM未知環境での2-SR中継点選択の直接最適化）の評価器 evaluator.py が使う
「経路の骨格」をここで作る。

中心となる出力:
  ecmp_unit_flows(topo) -> F  [n, n, E]
    F[s, t, e] = ODペア (s, t) の需要1単位を ECMP方式1 で流したときに
                 有向リンク e に載る量（= s->t トラフィックのうちリンク e を通る割合）。
    F[s, s, :] = 0。

なぜ NN 出力と切り離して前計算できるか:
  IGP ウェイトは固定なので最短経路構造は不変。ECMP方式1（各ルータで、宛先への
  最短次ホップに均等分割）も宛先だけで決まり、提案手法の分配比 alpha には依存しない。
  よって F は「定数テンソル」。alpha に対して微分可能なのは evaluator 側の線形結合。

ECMP方式1 の定義（確定事項）:
  各ノード u で、宛先 t への最短経路上にある出リンク (u, v)（= dist(u,t) == w(u,v)+dist(v,t)）
  に、1/(その本数) ずつ均等分割する。実機ルータの per-destination ECMP に相当。

計算法（宛先 t ごと）:
  分割行列 P[u, v] = 上記の均等分割割合。最短経路は距離が単調減少する DAG なので P は冪零。
  ノード u に「s 発の在荷量」a_s[u] は  a_s = e_s + P^T a_s  を満たすので
    A = (I - P^T)^{-1}          (A[u, s] = s 発トラフィックの u での通過量)
  リンク (u, v) の s 発フロー = A[u, s] * P[u, v]。これを全 e についてまとめて F にする。
"""

import torch


def all_pairs_dist(edges, weight, n_nodes):
    """有向グラフの全点対最短距離を Floyd-Warshall で計算（n が小さい前提）。

    Returns: FloatTensor [n, n]  dist[i, j] = i->j 最短距離（到達不能は inf）。
    """
    INF = float("inf")
    dist = torch.full((n_nodes, n_nodes), INF)
    for i in range(n_nodes):
        dist[i, i] = 0.0
    for e in range(edges.shape[0]):
        u, v = edges[e, 0].item(), edges[e, 1].item()
        w = weight[e].item()
        # 多重リンクがあれば最小を採用
        if w < dist[u, v]:
            dist[u, v] = w
    for k in range(n_nodes):
        # dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        dik = dist[:, k].unsqueeze(1)      # [n,1]
        dkj = dist[k, :].unsqueeze(0)      # [1,n]
        dist = torch.minimum(dist, dik + dkj)
    return dist


def ecmp_split_matrix(edges, weight, dist, t, n_nodes, eps=1e-9):
    """宛先 t に対する ECMP方式1 の分割行列 P [n, n] を返す。

    P[u, v] = ノード u が宛先 t 向けトラフィックを出リンク (u,v) に流す割合。
    u が t への最短次ホップを持たない（u==t または到達不能）行は全て 0。
    """
    P = torch.zeros(n_nodes, n_nodes)
    # 各ノードの「最短次ホップ」候補を集める
    for u in range(n_nodes):
        if u == t or dist[u, t] == float("inf"):
            continue
        nexthops = []
        for e in range(edges.shape[0]):
            if edges[e, 0].item() != u:
                continue
            v = edges[e, 1].item()
            w = weight[e].item()
            # (u,v) が t への最短経路上か: dist[u,t] == w + dist[v,t]
            if dist[v, t] != float("inf") and abs(dist[u, t].item() - (w + dist[v, t].item())) < eps:
                nexthops.append(v)
        if nexthops:
            share = 1.0 / len(nexthops)
            for v in nexthops:
                P[u, v] += share  # 平行リンクがあれば合算
    return P


def ecmp_unit_flows(topo):
    """全 ODペアの単位需要リンクフローテンソル F [n, n, E] を計算。

    F[s, t, e] = s->t 需要1単位のうち有向リンク e を通る量（ECMP方式1）。

    実装:
      宛先 t ごとに P を作り、A = (I - P^T)^{-1} を解く。
      A[u, s] = s発トラフィックの u での通過量。
      リンク e=(u,v) について F[s, t, e] = A[u, s] * P[u, v]。
    """
    edges = topo["edges"]
    weight = topo["weight"]
    n = topo["n_nodes"]
    E = edges.shape[0]

    dist = all_pairs_dist(edges, weight, n)
    F = torch.zeros(n, n, E)
    I = torch.eye(n)

    # リンク e ごとの (u, v) を先に取り出しておく
    us = edges[:, 0]  # [E]
    vs = edges[:, 1]  # [E]

    for t in range(n):
        P = ecmp_split_matrix(edges, weight, dist, t, n)   # [n, n]
        # a_s = e_s + P^T a_s  =>  A = (I - P^T)^{-1},  A[:, s] が s 発の在荷ベクトル
        A = torch.linalg.solve(I - P.transpose(0, 1), I)   # [n, n], A[u, s]
        # F[s, t, e] = A[u_e, s] * P[u_e, v_e]
        #   A[us]      : [E, n]  (行 e -> ノード u_e の在荷ベクトル over s)
        #   Pe         : [E]     (リンク e の分割割合 P[u_e, v_e])
        A_at_u = A[us]                       # [E, n]  = A[u_e, s]
        Pe = P[us, vs]                       # [E]
        contrib = A_at_u * Pe.unsqueeze(1)   # [E, n]  = F[s, t, e]（e,s 並び）
        F[:, t, :] = contrib.transpose(0, 1) # [n, E] -> F[s, t, e]
        F[t, t, :] = 0.0                     # 自分宛は 0

    return F, dist


# --- 検証用ユーティリティ（evaluator/verify から呼ぶ）-----------------------

def check_flow_conservation(F, edges, n_nodes, atol=1e-4):
    """F が流量保存を満たすか確認。

    各 (s, t), s != t について、有向リンクのフローがノード u で
      (出フロー) - (入フロー) = +1 (u=s) / -1 (u=t) / 0 (それ以外)
    を満たすはず。最大違反量を返す。
    """
    us = edges[:, 0]
    vs = edges[:, 1]
    E = edges.shape[0]
    max_viol = 0.0
    for s in range(n_nodes):
        for t in range(n_nodes):
            if s == t:
                continue
            f = F[s, t]                       # [E]
            net = torch.zeros(n_nodes)        # 出 - 入
            net.index_add_(0, us, f)          # 出
            net.index_add_(0, vs, -f)         # 入
            target = torch.zeros(n_nodes)
            target[s] += 1.0
            target[t] -= 1.0
            viol = (net - target).abs().max().item()
            max_viol = max(max_viol, viol)
    return max_viol


if __name__ == "__main__":
    import os
    from src.data_loader import build_dataset
    import glob

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "Repetita", "data", "2016TopologyZooUCL_inverseCapacity")
    graph = os.path.join(data_dir, "Abilene.graph")
    demands = sorted(glob.glob(os.path.join(data_dir, "Abilene.*.demands")))
    ds = build_dataset(graph, demands, k=4)

    F, dist = ecmp_unit_flows(ds)
    print(f"F shape = {tuple(F.shape)}  (n, n, E)")

    viol = check_flow_conservation(F, ds["edges"], ds["n_nodes"])
    print(f"max flow-conservation violation = {viol:.2e}  (0 に近ければ OK)")

    # ECMP がちゃんと分割している例を1つ見る: 対角に近いノード間
    s, t = 0, 5  # New_York -> Los_Angeles（複数最短経路がありそう）
    used = (F[s, t] > 1e-6).nonzero().squeeze(-1)
    print(f"\n{s}->{t} が使う有向リンク数 = {used.numel()}")
    for e in used.tolist():
        u, v = ds["edges"][e, 0].item(), ds["edges"][e, 1].item()
        print(f"  edge {e}: {u}->{v}  flow={F[s, t, e].item():.4f}")
    print(f"最短ホップ数 dist[{s},{t}] = {dist[s, t].item():.0f}  (weight=10/hop なので /10 がホップ数)")
