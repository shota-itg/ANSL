"""
STEP 1: REPETITA データローダ + 次数中心性による中継点候補
提案手法（TM未知環境での2-SR中継点選択の直接最適化）の基礎評価向け。

このモジュールが提供するもの:
  - load_topology(graph_path)   : .graph をパースし、ノード/有向リンク/容量/IGPウェイトを返す
  - load_tm(demands_path, n)    : .demands をパースし、TM行列 [n, n] を返す
  - degree_centrality_topk(...)  : 次数中心性の上位K個を中継点候補として返す
  - build_dataset(...)          : 上記をまとめてテンソル化した dict を返す（STEP 2以降の入口）

設計上の約束（STEP 2 の評価器が前提にするので固定しておく）:
  - リンクは「有向」。Abilene の .graph は各物理リンクを両方向2エントリで持つため、
    そのまま有向リンク列 edges[e] = (u, v) として扱う。MLU は各有向リンク負荷 / 容量 の最大。
  - リンクIDは .graph の出現順（edge_0, edge_1, ...）に一致させる。
  - 容量・需要の単位は .graph / .demands の生値のまま（Abilene は bw=9953280、需要も同単位系）。
    比 load/cap しか使わないので単位系が揃ってさえいれば MLU はスケール不変。
"""

import torch


def load_topology(graph_path):
    """REPETITA .graph をパース。

    Returns dict:
      n_nodes : int
      edges   : LongTensor [E, 2]  各行 (src, dst)  ※有向、.graph の出現順
      cap     : FloatTensor [E]    各有向リンクの容量 bw
      weight  : FloatTensor [E]    IGP ウェイト（SPF比較で使用）
      delay   : FloatTensor [E]    伝搬遅延（今回は未使用だが読んでおく）
    """
    with open(graph_path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    # --- NODES ---
    assert lines[i].startswith("NODES"), f"expected NODES header, got: {lines[i]}"
    n_nodes = int(lines[i].split()[1])
    i += 1
    assert lines[i].split()[:1] == ["label"], f"expected node column header, got: {lines[i]}"
    i += 1
    # ノード行を n_nodes 個読み飛ばす（座標は今回使わない）
    i += n_nodes

    # 空行スキップ
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # --- EDGES ---
    assert lines[i].startswith("EDGES"), f"expected EDGES header, got: {lines[i]}"
    n_edges = int(lines[i].split()[1])
    i += 1
    assert lines[i].split()[0] == "label", f"expected edge column header, got: {lines[i]}"
    i += 1

    src, dst, cap, weight, delay = [], [], [], [], []
    for _ in range(n_edges):
        # 形式: label src dest weight bw delay
        _, s, d, w, bw, dl = lines[i].split()
        src.append(int(s)); dst.append(int(d))
        weight.append(float(w)); cap.append(float(bw)); delay.append(float(dl))
        i += 1

    edges = torch.tensor(list(zip(src, dst)), dtype=torch.long)  # [E, 2]
    return {
        "n_nodes": n_nodes,
        "edges": edges,
        "cap": torch.tensor(cap, dtype=torch.float32),
        "weight": torch.tensor(weight, dtype=torch.float32),
        "delay": torch.tensor(delay, dtype=torch.float32),
    }


def load_tm(demands_path, n_nodes):
    """REPETITA .demands をパースし TM 行列 [n_nodes, n_nodes] を返す（対角0）。"""
    tm = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    with open(demands_path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    assert lines[0].startswith("DEMANDS"), f"expected DEMANDS header, got: {lines[0]}"
    assert lines[1].split()[:1] == ["label"], f"expected demand column header, got: {lines[1]}"
    for ln in lines[2:]:
        if not ln.strip():
            continue
        # 形式: label src dest bw
        _, s, d, bw = ln.split()
        tm[int(s), int(d)] = float(bw)
    return tm


def degree_centrality_topk(edges, n_nodes, k):
    """次数中心性の上位K個ノードを中継点候補として返す。

    有向グラフだが Abilene は対称なので in-deg == out-deg。
    ここでは「無向次数」= そのノードに接続する有向リンク本数の総和を用いる
    （src または dst に現れた回数）。同点は node id 昇順で決定的に選ぶ。

    Returns: LongTensor [k]  中継点候補ノードID（次数降順）
    """
    deg = torch.zeros(n_nodes, dtype=torch.long)
    for e in range(edges.shape[0]):
        u, v = edges[e, 0].item(), edges[e, 1].item()
        deg[u] += 1
        deg[v] += 1
    # 有向で両方向が別エントリなので、上のカウントは各物理リンクを4回
    # (u->v で u,v +1 / v->u で v,u +1) 数えるが、全ノード一律なので順位は不変。
    # 順位のみ使うため正規化は省略。

    # 次数降順・同点は id 昇順 → (-deg, id) でソート
    order = sorted(range(n_nodes), key=lambda x: (-deg[x].item(), x))
    topk = torch.tensor(order[:k], dtype=torch.long)
    return topk, deg


def build_dataset(graph_path, demands_paths, k):
    """STEP 2 以降の入口。トポロジ + TM系列 + 中継点候補をまとめて返す。

    demands_paths: list[str]  複数の .demands = TM系列（時刻順）
    Returns dict with tensors.
    """
    topo = load_topology(graph_path)
    n = topo["n_nodes"]
    tms = torch.stack([load_tm(p, n) for p in demands_paths], dim=0)  # [T, n, n]
    relays, deg = degree_centrality_topk(topo["edges"], n, k)
    return {
        **topo,
        "tms": tms,          # [T, n, n]
        "relays": relays,    # [k]
        "degree": deg,       # [n]
    }


if __name__ == "__main__":
    import glob, os

    here = os.path.dirname(os.path.abspath(__file__))
    graph = os.path.join(here, "Abilene.graph")
    demands = sorted(glob.glob(os.path.join(here, "Abilene.*.demands")))
    K = 4

    ds = build_dataset(graph, demands, k=K)

    print(f"n_nodes = {ds['n_nodes']}")
    print(f"n_edges (directed) = {ds['edges'].shape[0]}")
    print(f"capacity: uniform={bool((ds['cap'] == ds['cap'][0]).all())}, value={ds['cap'][0].item():.0f}")
    print(f"IGP weight: uniform={bool((ds['weight'] == ds['weight'][0]).all())}, value={ds['weight'][0].item():.0f}")
    print(f"TM series shape = {tuple(ds['tms'].shape)}  (T, n, n)")
    print(f"num nonzero OD pairs in TM[0] = {int((ds['tms'][0] > 0).sum())}  (expected {ds['n_nodes']*(ds['n_nodes']-1)} if full mesh)")
    print(f"degree per node = {ds['degree'].tolist()}")
    print(f"relay candidates (top-{K} by degree) = {ds['relays'].tolist()}")
    # 参考: TM総量の時間変動（level shift/スパイクの確認材料）
    totals = ds['tms'].sum(dim=(1, 2))
    print(f"TM total demand over time = {[round(t.item()) for t in totals]}")
