import time
import math
import random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

# ===== 1) トポロジ作成（K4: ノード4，リンク6）=====
def build_topology():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for u, v in edges:
        G.add_edge(u, v, capacity=1.0) # 全リンク容量=1.0
    return G, edges

# 候補経路をK本取得（短い順）
def k_candidate_paths(G, s, t, K=3) -> List[List[int]]:
    try:
        gen = nx.shortest_simple_paths(G, s, t, weight=None) # hop数最短→欠点...
        paths = []
        for p in gen:
            paths.append(p)
            if K <= len(paths):
                break
        return paths
    except nx.NetworkXNoPath:
        return []

# 経路のリンク列を edges のインデックス列に変換
def path_to_edge_indices(path: List[int], edges: List[Tuple[int, int]]) -> List[int]:
    idxs = []
    for u, v in zip(path[:-1], path[1:]):
        a, b = min(u, v), max(u, v) # 無向
        for i, (x, y) in enumerate(edges):
            if (x, y) == (a, b) or (x, y) == (b, a):
                idxs.append(i)
                break
    return idxs

# ある経路に需要demandを流した時の「名がs田後の最大リンク利用率」を計算
def max_util_after_routing(current_utils: List[float], edge_indices: List[int], demand: float, capacity=1.0) -> float:
    utils = current_utils[:]
    for ei in edge_indices:
        utils[ei] = min(1.0, utils[ei] + demand / capacity) # 単純加算（クリップ）
    return max(utils)


# ===== 2) データ生成 =====
def one_hot(n, i):
    v = [0.0]*n
    v[i] = 1.0
    return values

def gen_sample(G, edges, K=3):
    nodes = list(G.nodes())
    s = random.choice(nodes)
    t = random.choice([x for x in nodes if x != s])
    demand = random.uniform(0.1, 0.5)

    # ネットワーク状態（リンク利用率）: 0~0.5の乱数で初期化
    net_state = [random.uniform(0.0, 0.5) for _ in edges]

    # 候補経路
    paths = k_candidate_paths(G, s, t, K=K)
    if not paths:
        return None

    # 教師ラベル : 最小混雑（max利用率が最小）となる経路のインデックス
    costs = []
    edge_paths = []
    for p in paths:
        eidx = path_to_edge_indices(p, edges)
        edge_paths.append(eidx)
        costs.append(max_util_after_routing(net_state, eidx, demand, capacity=1.0))
    label = int(min(range(len(costs)), key=lambda i: costs[i]))

    # 入力ベクトル（モダリティ1 + モダリティ2）
    # M1: src(4) + dst(4) + demand(1) = 9
    m1 = one_hot(4, s) + one_hot(4, t) + [demand]
    # M2: ネットワーク状態（リンク利用率 長さ6）
    m2 = net_state
    
    return {
        "m1": torch.tensor(m1, dtype=torch.float32),
        "m2": torch.tensor(m2, dtype=torch.float32),
        "label": torch.tensor(label, dtype=torch.long),
        "paths": paths,             # 参考情報
        "edge_paths": edge_paths,   # 参考情報
    }

def build_dataset(G, edges, N=2000, K=3):
    data = []
    while len(data) < N:
        s = gen_sample(G, edges, K=K)
        if s is not None and s["label"].item() < K:  # 念のため
            data.append(s)
    return data

# ===== 3) モデル（中間層でモダリティ融合） =====
class TinyFusion(nn.Module):
    def __init__(self, m1_dim=9, m2_dim=6, hidden=32, out_classes=3):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.Linear(m1_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.m2 = nn.Sequential(
            nn.Linear(m2_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_classes)  # 候補経路Kクラス分類
        )

    def forward(self, m1, m2):
        h1 = self.m1(m1)
        h2 = self.m2(m2)
        h = torch.cat([h1, h2], dim=-1)  # Intermediate/Joint Fusion
        logits = self.head(h)
        return logits

# ===== 4) 学習ループ（時間計測つき） =====
def train_cpu():
    random.seed(42)
    torch.manual_seed(42)

    G, edges = build_topology()
    K = 3  # 候補経路数
    train_set = build_dataset(G, edges, N=2000, K=K)
    val_set   = build_dataset(G, edges, N=400,  K=K)

    model = TinyFusion(m1_dim=9, m2_dim=len(edges), hidden=32, out_classes=K)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def batchify(dataset, bs=64):
        for i in range(0, len(dataset), bs):
            chunk = dataset[i:i+bs]
            m1 = torch.stack([x["m1"] for x in chunk], dim=0)
            m2 = torch.stack([x["m2"] for x in chunk], dim=0)
            y  = torch.stack([x["label"] for x in chunk], dim=0)
            yield m1, m2, y

    EPOCHS = 5
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for m1, m2, y in batchify(train_set, bs=64):
            optimizer.zero_grad()
            logits = model(m1, m2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * m1.size(0)
        train_time = time.time() - t0

        # 検証
        model.eval()
        correct = 0
        with torch.no_grad():
            for m1, m2, y in batchify(val_set, bs=128):
                pred = model(m1, m2).argmax(dim=1)
                correct += (pred == y).sum().item()
        val_acc = correct / len(val_set)

        print(f"Epoch {epoch}/{EPOCHS} | Loss {total_loss/len(train_set):.4f} | ValAcc {val_acc*100:.1f}% | Time {train_time:.2f}s")

if __name__ == "__main__":
    train_cpu()