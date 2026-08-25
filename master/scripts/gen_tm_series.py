"""
合成TM系列の生成エントリ。
実Abileneトポロジ(容量・IGPウェイトは本物)の上に、時間相関を持つTM系列を
合成して .demands 群として書き出す。提案手法NN(train.py)の学習・評価データにする。

実行:
  master/ をカレントにして
    python -m scripts.gen_tm_series

出力:
  master/data/synth_abilene/ に
    Abilene.graph                （実トポロジのコピー。自己完結用）
    Abilene.00000.demands …      （時刻順の合成TM、既定 T=3000）

train.py / direct_optimize.py で使うには data_dir を上記へ向けるだけ:
    data_dir = os.path.join(here, "..", "data", "synth_abilene")
  （実データの5TMディレクトリはそのまま残る。混在させないため別フォルダにしている）

診断出力:
  - 隣接TMの平均相対変化（連続性の確認。小さいほど滑らか）
  - 校正後SPF-MLUの min/中央値/max（混雑レンジの確認。中央値≒target になる）
  - TM総量の時間推移サンプル（日内変動・level shift の目視材料）
"""

import os

import torch

from src.data_loader import load_topology
from src.routing import ecmp_unit_flows
from src.tm_generator import (
    generate_series, calibrate_to_spf_mlu, write_demands_series, spf_mlu_of_tm,
)


CONFIG = {
    "T": 3000,            # 生成するTM枚数（samples = T - H。学習/検証/テストに分ける）
    "seed": 0,
    "target_spf_mlu": 1.2,   # 校正目標（実Abilene SPF平均1.221 に近い水準）
    "period": 96,         # 日内周期ステップ数
    "amp_diurnal": 0.35,
    "drift_std": 0.03,
    "reversion": 0.02,
    "noise_std": 0.05,
    "spike_prob": 0.01,
    "spike_scale": 2.5,
    "size_sigma": 0.6,
    "out_subdir": os.path.join("data", "synth_abilene"),
    "prefix": "Abilene",
}


def adjacent_relative_change(tms):
    """隣接TMの平均相対変化 mean(||T_{t+1}-T_t|| / ||T_t||)。連続性の指標。"""
    diffs = []
    for t in range(tms.shape[0] - 1):
        num = (tms[t + 1] - tms[t]).norm()
        den = tms[t].norm() + 1e-12
        diffs.append((num / den).item())
    return float(sum(diffs) / len(diffs))


def main():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])

    here = os.path.dirname(os.path.abspath(__file__))
    real_dir = os.path.join(here, "..", "Repetita", "data", "2016TopologyZooUCL_inverseCapacity")
    graph = os.path.join(real_dir, "Abilene.graph")
    topo = load_topology(graph)
    n = topo["n_nodes"]
    print(f"トポロジ: Abilene  n_nodes={n}, 有向リンク={topo['edges'].shape[0]}")

    # --- 生成 ---
    tms = generate_series(
        n, cfg["T"], seed=cfg["seed"],
        period=cfg["period"], amp_diurnal=cfg["amp_diurnal"],
        drift_std=cfg["drift_std"], reversion=cfg["reversion"], noise_std=cfg["noise_std"],
        spike_prob=cfg["spike_prob"], spike_scale=cfg["spike_scale"],
        size_sigma=cfg["size_sigma"],
    )
    print(f"生成: TM系列 shape={tuple(tms.shape)}  (T, n, n)")

    # --- 校正（SPF-MLU中央値を target に）---
    tms, factor, spf_before = calibrate_to_spf_mlu(tms, topo, target=cfg["target_spf_mlu"])
    F, _ = ecmp_unit_flows(topo)
    spf_after = torch.tensor([spf_mlu_of_tm(F, topo["cap"], tms[t]) for t in range(tms.shape[0])])
    print(f"校正倍率 = {factor:.4g}")
    print(f"SPF-MLU(校正後): min={spf_after.min():.3f}  中央値={spf_after.median():.3f}  "
          f"max={spf_after.max():.3f}")

    # --- 診断 ---
    arc = adjacent_relative_change(tms)
    totals = tms.sum(dim=(1, 2))
    print(f"隣接TM平均相対変化 = {arc:.4f}  (小さいほど連続的)")
    print(f"TM総量の推移(先頭10) = {[round(float(x)) for x in totals[:10]]}")
    print(f"1未満のSPF-MLU割合 = {(spf_after < 1.0).float().mean().item():.2%}  "
          f"(0%だと常時混雑で改善余地は大、100%だと非混雑)")

    # --- 書き出し ---
    out_dir = os.path.join(here, "..", cfg["out_subdir"])
    T = write_demands_series(tms, out_dir, prefix=cfg["prefix"], graph_src=graph)
    print(f"\n書き出し完了: {out_dir}")
    print(f"  {cfg['prefix']}.graph + {T} 個の .demands")
    print("\n次の手順:")
    print("  train.py / direct_optimize.py の data_dir を次に変更:")
    print(f'    data_dir = os.path.join(here, "..", "{cfg["out_subdir"].replace(os.sep, "/")}")')
    print("  その後  python -m scripts.train  で提案手法NNを学習・評価。")


if __name__ == "__main__":
    main()
