"""
STEP 2 (後半): 微分可能な MLU 評価器
提案手法の中核。中継点への分配比 alpha を入力に、2-SR 経路をECMPで敷いた
ときの最大リンク利用率 (MLU) を返す。alpha について微分可能で、勾配が MLU まで流れる。

構造（前回すり合わせた通り）:
  alpha[od, j]  ─(線形)→ リンク負荷 ─(/容量)→ 利用率 ─(max)→ MLU
  F, 需要 d, 容量 cap は定数。勾配が流れるのは alpha だけ。

前計算 path_load[od, j, e]:
  ODペア od=(s,t) を「選択肢 j」で流したときの、需要1単位あたりのリンク e 負荷。
    - j が中継点 r のとき: s->r->t なので  F[s,r,e] + F[r,t,e]
    - j が「直接」のとき  : s->t 最短なので F[s,t,e]        （include_direct=True の場合）
  これは routing.ecmp_unit_flows(F) から作る定数テンソル。

2-SR の意味:
  各 ODペアの需要を「中継点集合（+ 直接）」に alpha の比率で分割し、各断片は
  s->r 最短(ECMP) と r->t 最短(ECMP) を継いだ経路をたどる。argmax 丸め（単一SRパス化）は
  推論時にのみ行い、学習時は連続の alpha のまま勾配を通す（確定方針）。
"""

import torch


def build_path_load(F, od_pairs, relays, include_direct=True):
    """前計算テンソル path_load [OD, J, E] を作る。

    Args:
      F         : [n, n, E]  ecmp_unit_flows の出力（定数）
      od_pairs  : LongTensor [OD, 2]  評価対象の (s, t) 一覧（s != t）
      relays    : LongTensor [K]      中継点候補ノードID（次数中心性 top-K）
      include_direct : True なら選択肢 j=0 に「直接 s->t」を加える（合計 J=K+1）

    Returns:
      path_load : FloatTensor [OD, J, E]
      option_relay : LongTensor [J]  各選択肢が対応する中継点ノードID
                     （直接の列は便宜上 -1）
    """
    OD = od_pairs.shape[0]
    E = F.shape[2]
    K = relays.shape[0]
    J = K + (1 if include_direct else 0)

    path_load = torch.zeros(OD, J, E)
    option_relay = torch.empty(J, dtype=torch.long)

    s = od_pairs[:, 0]  # [OD]
    t = od_pairs[:, 1]  # [OD]

    col = 0
    if include_direct:
        path_load[:, col, :] = F[s, t]          # [OD, E]  直接 s->t
        option_relay[col] = -1
        col += 1
    for r in relays.tolist():
        # s->r->t = F[s,r] + F[r,t]。r==s / r==t の退化は F[x,x]=0 で自然に処理される。
        path_load[:, col, :] = F[s, r] + F[r, t]  # [OD, E]
        option_relay[col] = r
        col += 1

    return path_load, option_relay


class MLUEvaluator:
    """定数（path_load, 容量）を保持し、alpha と需要から MLU を返す。

    forward は微分可能。学習ループ(STEP3/4)からは loss = evaluator(alpha, demand) と使う。
    """

    def __init__(self, path_load, cap):
        """
        path_load : [OD, J, E]
        cap       : [E]  各有向リンク容量
        """
        self.path_load = path_load          # 定数
        self.cap = cap                       # 定数
        self.OD, self.J, self.E = path_load.shape

    def link_load(self, alpha, demand):
        """リンク負荷ベクトル [E] を返す（微分可能）。

        alpha  : [OD, J]  各 ODペアの選択肢分配比（行方向に和=1、非負）
        demand : [OD]     各 ODペアの需要量（このTMの値）
        """
        # 各 ODペアが各リンクに与える負荷 = demand * sum_j alpha[od,j]*path_load[od,j,e]
        # weighted[od, e] = sum_j alpha[od,j] * path_load[od,j,e]
        weighted = torch.einsum("oj,oje->oe", alpha, self.path_load)  # [OD, E]
        load = torch.einsum("o,oe->e", demand, weighted)             # [E]
        return load

    def utilization(self, alpha, demand):
        """リンク利用率ベクトル [E]（= 負荷/容量）。"""
        return self.link_load(alpha, demand) / self.cap

    def mlu(self, alpha, demand):
        """最大リンク利用率（スカラ）。これが提案手法の損失。"""
        return self.utilization(alpha, demand).max()

    # 学習では logits を出す想定なので softmax ヘルパも用意
    @staticmethod
    def softmax_alpha(logits):
        """logits [OD, J] -> 行方向 softmax で正規化した alpha。"""
        return torch.softmax(logits, dim=1)


def default_od_pairs(tm, n_nodes, thresh=0.0):
    """TM から需要が thresh より大きい ODペア一覧と需要ベクトルを作る。

    Returns:
      od_pairs : LongTensor [OD, 2]
      demand   : FloatTensor [OD]
    """
    pairs, dem = [], []
    for s in range(n_nodes):
        for t in range(n_nodes):
            if s != t and tm[s, t].item() > thresh:
                pairs.append((s, t))
                dem.append(tm[s, t].item())
    od_pairs = torch.tensor(pairs, dtype=torch.long)
    demand = torch.tensor(dem, dtype=torch.float32)
    return od_pairs, demand


if __name__ == "__main__":
    import os, glob
    from src.data_loader import build_dataset
    from src.routing import ecmp_unit_flows

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "Repetita", "data", "2016TopologyZooUCL_inverseCapacity")
    ds = build_dataset(os.path.join(data_dir, "Abilene.graph"),
                       sorted(glob.glob(os.path.join(data_dir, "Abilene.*.demands"))),
                       k=4)
    F, dist = ecmp_unit_flows(ds)

    # TM[0] を使ってセットアップ
    tm0 = ds["tms"][0]
    od_pairs, demand = default_od_pairs(tm0, ds["n_nodes"])
    path_load, option_relay = build_path_load(F, od_pairs, ds["relays"], include_direct=True)
    ev = MLUEvaluator(path_load, ds["cap"])

    OD, J, E = path_load.shape
    print(f"OD={OD}, J={J} (option_relay={option_relay.tolist()}, -1=直接), E={E}")

    # --- サニティ1: 全需要を「直接」に置いた場合の MLU = ECMP最短経路(SPF)ベースライン ---
    alpha_direct = torch.zeros(OD, J)
    alpha_direct[:, 0] = 1.0  # 列0 = 直接
    mlu_spf = ev.mlu(alpha_direct, demand)
    util = ev.utilization(alpha_direct, demand)
    e_max = util.argmax().item()
    u, v = ds["edges"][e_max, 0].item(), ds["edges"][e_max, 1].item()
    print(f"\n[SPFベースライン] 全ODを直接ECMP最短経路: MLU = {mlu_spf.item():.4f}")
    print(f"  ボトルネックリンク = edge{e_max} ({u}->{v}), 利用率 {util[e_max].item():.4f}")

    # --- サニティ2: 全需要を中継点候補に均等分配した場合 ---
    alpha_unif = torch.full((OD, J), 1.0 / J)
    print(f"[参考] 全選択肢に均等分配: MLU = {ev.mlu(alpha_unif, demand).item():.4f}")

    # --- 検証: alpha についての勾配が正しく計算できるか（gradcheck, double精度）---
    torch.manual_seed(0)
    logits = torch.randn(OD, J, dtype=torch.double, requires_grad=True)
    ev_d = MLUEvaluator(path_load.double(), ds["cap"].double())
    demand_d = demand.double()

    def f(logits_):
        alpha_ = torch.softmax(logits_, dim=1)
        return ev_d.mlu(alpha_, demand_d)

    ok = torch.autograd.gradcheck(f, (logits,), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"\n[gradcheck] alpha->MLU の解析勾配 == 数値勾配 : {ok}")

    # --- 勾配が実際に流れることの確認: 1ステップだけ勾配降下して MLU が下がるか ---
    logits = torch.randn(OD, J, requires_grad=True)
    alpha = torch.softmax(logits, dim=1)
    mlu_before = ev.mlu(alpha, demand)
    mlu_before.backward()
    with torch.no_grad():
        logits2 = logits - 5.0 * logits.grad  # 適当なステップ
    mlu_after = ev.mlu(torch.softmax(logits2, dim=1), demand)
    print(f"[勾配降下1歩] MLU {mlu_before.item():.4f} -> {mlu_after.item():.4f} "
          f"({'減少' if mlu_after < mlu_before else '非減少'})")
