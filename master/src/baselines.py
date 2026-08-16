"""
STEP 5 (一部): 比較手法・理論下限
まずは MCF（Multi-Commodity Flow）による理論最適 MLU を線形計画で解く。
これは「経路を自由に分割してよい」場合に到達できる最小 MLU で、あらゆる
ルーティング手法の下限（誰もこれより良くはできない）。提案手法の到達度を
「最適との差」で語るための基準線。

optimal_mlu_mcf(edges, cap, tm, n_nodes) -> float

定式化（宛先集約型 LP。変数 f[e,t] = 宛先 t 向けトラフィックのリンク e 上の量）:
  min  theta
  s.t. 各ノード v・各宛先 t で流量保存:
         (v から出る f[e,t] の和) - (v に入る f[e,t] の和) = b[v,t]
         b[v,t] = tm[v,t]           (v != t, 発地)
         b[t,t] = -sum_v tm[v,t]    (t, 集約先)
       各リンク e で容量: sum_t f[e,t] <= theta * cap[e]
       f >= 0
注意:
  経路分割を自由に許すので、これは ECMP や単一SRパスより必ず小さい（か等しい）。
  よって 2-SR連続最適(STEP3) がこの値に近ければ、中継モデルが最適に迫れている証拠。
"""

import numpy as np
from scipy.optimize import linprog


def optimal_mlu_mcf(edges, cap, tm, n_nodes):
    """MCF 理論最適 MLU をLPで解いて返す（float）。

    edges : LongTensor/ndarray [E, 2]
    cap   : FloatTensor/ndarray [E]
    tm    : FloatTensor/ndarray [n, n]
    """
    E = len(edges)
    n = n_nodes
    edges = np.asarray(edges)
    cap = np.asarray(cap, dtype=float)
    tm = np.asarray(tm, dtype=float)

    # 変数: x = [ f[e,t] (e=0..E-1, t=0..n-1) ..., theta ]
    nvar = E * n + 1
    theta_idx = E * n

    def fidx(e, t):
        return e * n + t

    # 目的: min theta
    c = np.zeros(nvar)
    c[theta_idx] = 1.0

    # 等式: 流量保存  各 (v, t) で1本  → n*n 本
    A_eq = np.zeros((n * n, nvar))
    b_eq = np.zeros(n * n)
    row = 0
    for t in range(n):
        for v in range(n):
            for e in range(E):
                s_e, d_e = int(edges[e, 0]), int(edges[e, 1])
                if s_e == v:
                    A_eq[row, fidx(e, t)] += 1.0   # v から出る
                if d_e == v:
                    A_eq[row, fidx(e, t)] -= 1.0   # v に入る
            if v != t:
                b_eq[row] = tm[v, t]
            else:
                b_eq[row] = -tm[:, t].sum()
            row += 1

    # 不等式: 各リンク容量  sum_t f[e,t] - cap[e]*theta <= 0  → E 本
    A_ub = np.zeros((E, nvar))
    b_ub = np.zeros(E)
    for e in range(E):
        for t in range(n):
            A_ub[e, fidx(e, t)] = 1.0
        A_ub[e, theta_idx] = -cap[e]

    bounds = [(0, None)] * nvar   # f>=0, theta>=0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"MCF LP failed: {res.message}")
    return float(res.x[theta_idx])


if __name__ == "__main__":
    import os, glob
    from src.data_loader import build_dataset

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "Repetita", "data", "2016TopologyZooUCL_inverseCapacity")
    ds = build_dataset(os.path.join(data_dir, "Abilene.graph"),
                       sorted(glob.glob(os.path.join(data_dir, "Abilene.*.demands"))),
                       k=4)

    print("MCF 理論最適 MLU（経路自由分割の下限）:")
    for ti in range(ds["tms"].shape[0]):
        opt = optimal_mlu_mcf(ds["edges"], ds["cap"], ds["tms"][ti], ds["n_nodes"])
        print(f"  TM{ti}: {opt:.4f}")
