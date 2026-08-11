"""
STEP 3: NNなし直接最適化（デバッグ／到達目標の確認）
提案手法の NN（STEP 4）が「TMを見ずに」目指す MLU の、TMごとの到達目標を
「TMを見てよい」直接最適化で求める。評価器 evaluator.py が正しく学習可能かの検証も兼ねる。

やること:
  各 TM について、中継点分配比 alpha（= softmax(logits)）を Adam で直接勾配降下し
  MLU を最小化する。NN は使わず logits を直接パラメータにする。
  記録するもの（TMごと）:
    - SPF     : 全ODを直接ECMP最短に流したときの MLU（ベースライン）
    - 連続最適 : 直接最適化後の連続 alpha の MLU（この relay モデルでの到達下限の目安）
    - 丸め後   : argmax で単一SRパスに丸めた後の MLU（推論時に相当。丸めコストが見える）

実行:
  master/ をカレントにして
    python -m scripts.direct_optimize
  主要ハイパラは下の CONFIG で調整。
"""

import torch

from src.data_loader import build_dataset
from src.routing import ecmp_unit_flows
from src.evaluator import (
    MLUEvaluator, build_path_load, default_od_pairs,
)


# --- 設定（当面はここを直接いじる。後で configs/*.yaml に移す想定）-------------
CONFIG = {
    "K": 4,                # 中継点候補数（次数中心性 top-K）
    "include_direct": True,  # 選択肢に直接 s->t を含める（J = K+1）
    "iters": 1000,         # 各TMの最適化反復数
    "lr": 0.1,             # Adam 学習率
    "seed": 0,             # 乱数固定（logits 初期化）
}


def spf_mlu(ev, demand):
    """全ODを直接(列0)に置いた SPFベースライン MLU。"""
    alpha = torch.zeros(ev.OD, ev.J)
    alpha[:, 0] = 1.0
    return ev.mlu(alpha, demand).item()


def round_argmax(alpha):
    """連続 alpha を各ODで argmax の単一選択肢に丸める（推論時の単一SRパス化）。"""
    hard = torch.zeros_like(alpha)
    hard[torch.arange(alpha.shape[0]), alpha.argmax(dim=1)] = 1.0
    return hard


def optimize_tm(ev, demand, iters, lr, seed):
    """1つのTMについて logits を直接最適化して MLU を最小化。

    Returns dict:
      cont_mlu   : 連続 alpha での最良 MLU
      round_mlu  : その alpha を argmax 丸めした MLU
      alpha      : 最良の連続 alpha [OD, J]
    """
    torch.manual_seed(seed)
    logits = torch.zeros(ev.OD, ev.J, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    best_mlu = float("inf")
    best_alpha = None
    for _ in range(iters):
        opt.zero_grad()
        alpha = torch.softmax(logits, dim=1)
        mlu = ev.mlu(alpha, demand)
        mlu.backward()
        opt.step()
        # Adam は行き過ぎることがあるので最良値を保持
        with torch.no_grad():
            cur = ev.mlu(torch.softmax(logits, dim=1), demand).item()
        if cur < best_mlu:
            best_mlu = cur
            best_alpha = torch.softmax(logits, dim=1).detach().clone()

    round_mlu = ev.mlu(round_argmax(best_alpha), demand).item()
    return {"cont_mlu": best_mlu, "round_mlu": round_mlu, "alpha": best_alpha}


def main():
    import os, glob

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "Repetita", "data", "2016TopologyZooUCL_inverseCapacity")
    graph = os.path.join(data_dir, "Abilene.graph")
    demands_paths = sorted(glob.glob(os.path.join(data_dir, "Abilene.*.demands")))

    ds = build_dataset(graph, demands_paths, k=CONFIG["K"])
    F, _ = ecmp_unit_flows(ds)                       # トポロジのみ依存。一度だけ計算。
    tms = ds["tms"]                                   # [T, n, n]
    n = ds["n_nodes"]

    print(f"K={CONFIG['K']}  relays={ds['relays'].tolist()}  "
          f"include_direct={CONFIG['include_direct']}  "
          f"iters={CONFIG['iters']}  lr={CONFIG['lr']}\n")

    header = f"{'TM':>3} | {'SPF':>8} | {'連続最適':>8} | {'丸め後':>8} | {'SPF比(丸め)':>10}"
    print(header)
    print("-" * len(header))

    agg = {"spf": [], "cont": [], "round": []}
    for ti in range(tms.shape[0]):
        tm = tms[ti]
        od_pairs, demand = default_od_pairs(tm, n)
        path_load, _ = build_path_load(F, od_pairs, ds["relays"],
                                       include_direct=CONFIG["include_direct"])
        ev = MLUEvaluator(path_load, ds["cap"])

        spf = spf_mlu(ev, demand)
        res = optimize_tm(ev, demand, CONFIG["iters"], CONFIG["lr"], CONFIG["seed"])
        cont, rnd = res["cont_mlu"], res["round_mlu"]
        ratio = rnd / spf  # 1未満なら SPF より改善

        agg["spf"].append(spf); agg["cont"].append(cont); agg["round"].append(rnd)
        print(f"{ti:>3} | {spf:>8.4f} | {cont:>8.4f} | {rnd:>8.4f} | {ratio:>10.3f}")

    def mean(x):
        return sum(x) / len(x)
    print("-" * len(header))
    print(f"{'平均':>3} | {mean(agg['spf']):>8.4f} | {mean(agg['cont']):>8.4f} | "
          f"{mean(agg['round']):>8.4f} | {mean([r/s for r,s in zip(agg['round'],agg['spf'])]):>10.3f}")

    print("\n[読み方]")
    print("  連続最適 < SPF なら、2-SR中継モデルで直接最適化すれば SPF より改善する余地がある。")
    print("  丸め後 - 連続最適 が、単一SRパス化(argmax)で失う分（丸めコスト）。")
    print("  ここでの『連続最適』は TM を見て最適化した到達目標。STEP 4 の NN は")
    print("  TM を見ずにこの値へどれだけ近づけるかで評価する。")


if __name__ == "__main__":
    main()
