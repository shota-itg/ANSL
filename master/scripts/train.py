"""
STEP 4 (後半): 提案手法 NN の学習（DOTE方式）
過去のTM（履歴）を入力にNNが中継点分配比 alpha を出力し、それを「次時刻の
（未知の）TM」に適用したときの MLU を損失として勾配法で直接学習する。
予測とMLUの目的不整合を避ける DOTE の枠組みを 2-SR 中継点空間で行う。

実行:
  master/ をカレントにして
    python -m scripts.train

【重要な注意：データ量】
  REPETITA同梱のAbilene需要は5時刻(TM)のみ。DOTE系は本来数百〜数千のTMで
  学習・評価する。5つでは「履歴からの予測」は統計的にほぼ意味を持たない。
  ここでは提案手法のパイプラインが端から端まで動くことのデモとして回している。
  基礎結果として数値を主張するには、AbileneのフルなTM系列(TOTEM/SNDlib等)の
  取得が別途必要になる可能性が高い（戻ってから判断）。

【設計判断：NNに何を入力するか → make_features を差し替える】
  デフォルト: 過去 H 時刻の TM を平坦化して入力（DOTE準拠）。
  将来: TM履歴＋リンク利用率のマルチモーダル入力（提案手法2）に差し替え可能。
"""

import os
import glob
import torch

from src.data_loader import build_dataset
from src.routing import ecmp_unit_flows
from src.evaluator import MLUEvaluator, build_path_load
from src.model import RelayAllocMLP

# MCF比較は scipy 導入後に自動で有効化（未導入でも train.py 自体は動く）
try:
    from src.baselines import optimal_mlu_mcf
    _HAS_MCF = True
except Exception:
    _HAS_MCF = False


CONFIG = {
    "K": 4,                 # 中継点候補数
    "include_direct": True,   # 選択肢に直接 s->t を含める（J=K+1）
    "history": 1,           # 入力に使う過去TM数 H
    "hidden": (256, 256),   # MLP隠れ層
    "epochs": 2000,         # 学習エポック
    "lr": 1e-3,             # Adam 学習率
    "seed": 0,
    "n_test": 1,            # 系列末尾の何時刻をテストに回すか
}


# --- 固定ODペア（全 s!=t）。TM間で alpha の並びを固定するため -------------------
def all_od_pairs(n):
    pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
    return torch.tensor(pairs, dtype=torch.long)


def demand_vector(tm, od_pairs):
    """TM から od_pairs 順の需要ベクトル [OD] を作る。"""
    s = od_pairs[:, 0]
    t = od_pairs[:, 1]
    return tm[s, t].clone()


# --- 入力特徴（★設計判断の中心。ここを差し替えると提案手法の性格が変わる）--------
def make_features(tm_window, scale):
    """過去TMの窓 -> 入力特徴ベクトル [in_dim]。

    tm_window : list[Tensor[n,n]]  直近 H 時刻の TM（古い→新しい）
    scale     : float  正規化スケール（学習安定化。学部研究で未正規化が課題だった点への対処）
    デフォルト: 各TMを平坦化して連結し、scale で割る。
    """
    feats = [tm.flatten() / scale for tm in tm_window]
    return torch.cat(feats)


def spf_mlu(ev, demand):
    alpha = torch.zeros(ev.OD, ev.J)
    alpha[:, 0] = 1.0
    return ev.mlu(alpha, demand).item()


def round_argmax(alpha):
    hard = torch.zeros_like(alpha)
    hard[torch.arange(alpha.shape[0]), alpha.argmax(dim=1)] = 1.0
    return hard


def main():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "Repetita", "data", "2016TopologyZooUCL_inverseCapacity")
    graph = os.path.join(data_dir, "Abilene.graph")
    demands_paths = sorted(glob.glob(os.path.join(data_dir, "Abilene.*.demands")))

    ds = build_dataset(graph, demands_paths, k=cfg["K"])
    n = ds["n_nodes"]
    tms = ds["tms"]                                  # [T, n, n]
    T = tms.shape[0]

    # 評価器（path_load は固定ODで一度だけ構築。cap も固定）
    F, _ = ecmp_unit_flows(ds)
    od_pairs = all_od_pairs(n)
    path_load, _ = build_path_load(F, od_pairs, ds["relays"],
                                   include_direct=cfg["include_direct"])
    ev = MLUEvaluator(path_load, ds["cap"])
    OD, J = ev.OD, ev.J

    # 正規化スケール（容量値。特徴を利用率オーダーに）
    scale = float(ds["cap"].max().item())

    # サンプル生成: (過去H時刻の窓) -> (次時刻TMでMLU)
    H = cfg["history"]
    samples = []
    for i in range(H, T):
        window = [tms[i - H + k] for k in range(H)]   # 古い→新しい
        x = make_features(window, scale)
        target_demand = demand_vector(tms[i], od_pairs)
        samples.append((x, target_demand, i))
    if len(samples) == 0:
        raise RuntimeError(f"TMが少なすぎて履歴H={H}のサンプルが作れない (T={T})")

    # 系列末尾 n_test 個をテスト、残りを学習に
    n_test = cfg["n_test"]
    train_samples = samples[:-n_test] if n_test < len(samples) else samples[:1]
    test_samples = samples[-n_test:]
    in_dim = samples[0][0].numel()

    print(f"T={T} TMs, history H={H} -> samples={len(samples)} "
          f"(train={len(train_samples)}, test={len(test_samples)})")
    print(f"OD={OD}, J={J}, in_dim={in_dim}, relays={ds['relays'].tolist()}")
    print("※ データ5TMは提案手法パイプラインのデモ。数値主張には要フルTM系列。\n")

    model = RelayAllocMLP(in_dim, OD, J, hidden=cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # --- 学習: 損失 = 学習サンプルの平均MLU（DOTEと同じくMLUそのもの）----------
    for ep in range(cfg["epochs"]):
        opt.zero_grad()
        loss = 0.0
        for x, dem, _ in train_samples:
            alpha = model(x)                     # [OD, J]
            loss = loss + ev.mlu(alpha, dem)
        loss = loss / len(train_samples)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, cfg["epochs"] // 5) == 0:
            print(f"  epoch {ep+1:>5}: train平均MLU = {loss.item():.4f}")

    # --- 評価: テストTMで NN / SPF / STEP3到達目標(=そのTMを直接最適化) / MCF ----
    print("\n=== テスト評価 ===")
    header = f"{'TM':>3} | {'SPF':>7} | {'NN(連続)':>8} | {'NN(丸め)':>8}"
    if _HAS_MCF:
        header += f" | {'MCF最適':>7}"
    print(header); print("-" * len(header))

    model.eval()
    with torch.no_grad():
        for x, dem, i in test_samples:
            alpha = model(x)
            nn_cont = ev.mlu(alpha, dem).item()
            nn_round = ev.mlu(round_argmax(alpha), dem).item()
            spf = spf_mlu(ev, dem)
            row = f"{i:>3} | {spf:>7.4f} | {nn_cont:>8.4f} | {nn_round:>8.4f}"
            if _HAS_MCF:
                mcf = optimal_mlu_mcf(ds["edges"], ds["cap"], tms[i], n)
                row += f" | {mcf:>7.4f}"
            print(row)

    print("\n[読み方]")
    print("  NN(連続) が SPF より小さければ、TMを見ずに出したαでも混雑を減らせている。")
    print("  NN(丸め) は単一SRパス化後（推論時相当）。MCF最適が理論下限。")
    if not _HAS_MCF:
        print("  （scipy未導入のためMCF列は省略。pip install scipy で表示される）")
    print("  ※ 5TMではテスト1点のみ。傾向を語るにはフルTM系列が必要。")


if __name__ == "__main__":
    main()
