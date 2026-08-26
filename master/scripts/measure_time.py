"""
STEP 5 (追加): 計算時間の計測  ― 図2（提案手法 vs MCF理論下限）用
============================================================================
「1つの経路設定を計算するのに要する時間」を手法ごとに測る。過去チャットで
確定した計測方針をそのまま実装している。

  提案手法（丸めのみ）:
    学習済みNNに『過去 H=5 時刻の TM』を入力 → 順伝播1回で分配比 alpha を得て
    → argmax 丸めで単一SRパス化、までの時間。
    ＝ make_features(window) + model(x) + round_argmax(alpha)

  MCF理論下限:
    そのTMを既知として LP（scipy.linprog, HiGHS）を解いて最小MLUを求める時間。
    ＝ optimal_mlu_mcf(edges, cap, tm, n)

計測に含めない（＝タイマ区間の外で一度だけ実行）もの:
  - 中継点候補選択（次数中心性）などの前処理  … トポロジ固定で使い回すオフライン処理
  - path_load / F / 評価器 / ODペアの構築       … 定数、TMごとに再計算しない
  - 学習済みモデルのロード                      … 1TMの推論コストではない
  - 最初のウォームアップ数回                    … ライブラリ初期化・キャッシュ整備の外れ値

計測の作法:
  - 複数のテストTMについて各手法の時間を測り、平均（と標準偏差）を報告する。
  - NN推論は sub-ms オーダーでタイマ分解能ノイズを受けやすいので、TM1つあたり
    REPEAT_NN 回まわして中央値を「そのTMの1回分」とする（合計時間÷回数）。
  - MCF は1回が重く決定論的なので、TM1つあたり1回でよい。
  - torch は set_num_threads(1) で単スレッド固定（再現性のため）。HiGHS は内部で
    マルチスレッドの可能性があり、この非対称性は §で明記する前提。

実行:  master/ をカレントにして
    python -m scripts.measure_time
出力:  results/timing.dat        （method  mean_ms  std_ms  … gnuplot 棒グラフ用）
       results/timing_perTM.dat  （TMごとの生値。ばらつき確認・エラーバー用）
"""

import os
import glob
import time
import statistics as stats

import torch

from src.data_loader import build_dataset
from src.routing import ecmp_unit_flows
from src.evaluator import MLUEvaluator, build_path_load
from src.model import RelayAllocMLP
from src.baselines import optimal_mlu_mcf

# train.py と同一の純粋ヘルパを再利用（挙動を完全に一致させるため import する）
from scripts.train import (
    all_od_pairs, demand_vector, make_features, round_argmax,
)

# ---------------------------------------------------------------------------
CONFIG = {
    "K": 4,                 # 中継点候補数（train と一致させる）
    "include_direct": True,   # 選択肢に直接 s->t を含める（J=K+1）
    "history": 5,           # 入力に使う過去TM数 H（提案手法は H=5）
    "hidden": (256, 256),
    "seed": 0,
    # データ位置: 合成系列があればそれを、無ければ REPETITA 同梱5TMを使う
    "data_dir_candidates": [
        os.path.join("data", "synth_abilene"),
        os.path.join("Repetita", "data", "2016TopologyZooUCL_inverseCapacity"),
    ],
    "graph_name": "Abilene.graph",
    "demands_glob": "Abilene.*.demands",
    "n_measure": 20,        # 計測に使うテストTM数（10〜20目安。あるだけ使う）
    "warmup": 5,            # 各手法のウォームアップ回数（平均から除外）
    "repeat_nn": 50,        # NN1TMあたりの反復回数（中央値を1回分とする）
    "ckpt": "results/model.pt",  # 学習済み重みがあれば読む（無ければ新規初期化）
}


def _resolve_data_dir(here):
    for rel in CONFIG["data_dir_candidates"]:
        d = os.path.join(here, "..", rel)
        if os.path.isdir(d) and glob.glob(os.path.join(d, CONFIG["demands_glob"])):
            return d
    raise FileNotFoundError(
        "Abilene の .demands が見つかりません。data/synth_abilene か "
        "Repetita/data/... を用意してください。"
    )


def main():
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(1)   # 単スレッド固定（計測の再現性）

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = _resolve_data_dir(here)
    graph = os.path.join(data_dir, CONFIG["graph_name"])
    demands_paths = sorted(glob.glob(os.path.join(data_dir, CONFIG["demands_glob"])))
    print(f"[data] {data_dir}  ({len(demands_paths)} TMs)")

    ds = build_dataset(graph, demands_paths, k=CONFIG["K"])
    n = ds["n_nodes"]
    tms = ds["tms"]                       # [T, n, n]
    T = tms.shape[0]

    # ---- 前処理（計測区間の外。一度だけ）--------------------------------------
    F, _ = ecmp_unit_flows(ds)
    od_pairs = all_od_pairs(n)
    path_load, _ = build_path_load(F, od_pairs, ds["relays"],
                                   include_direct=CONFIG["include_direct"])
    ev = MLUEvaluator(path_load, ds["cap"])
    OD, J = ev.OD, ev.J
    scale = float(ds["cap"].max().item())
    H = CONFIG["history"]

    # ---- サンプル（過去H時刻の窓）。末尾側を計測対象に ------------------------
    windows = []
    for i in range(H, T):
        window = [tms[i - H + k] for k in range(H)]     # 古い→新しい
        windows.append((window, i))
    if len(windows) == 0:
        raise RuntimeError(
            f"TMが少なすぎて履歴H={H}の窓が作れない (T={T})。"
            "計時には合成TM系列（数百TM以上）を使うこと。"
        )
    m = min(CONFIG["n_measure"], len(windows))
    measure = windows[-m:]
    print(f"[setup] OD={OD}, J={J}, in_dim={H*n*n}, "
          f"relays={ds['relays'].tolist()}, 計測TM数={m}")

    # ---- モデル（計測区間の外でロード。順伝播の時間は重みの中身に依らない）----
    in_dim = H * n * n
    model = RelayAllocMLP(in_dim, OD, J, hidden=CONFIG["hidden"])
    ckpt = os.path.join(here, "..", CONFIG["ckpt"])
    if os.path.isfile(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[model] loaded checkpoint: {CONFIG['ckpt']}")
    else:
        print("[model] no checkpoint → 新規初期化で計測（順伝播の時間は重み値に非依存）")
    model.eval()

    # ================= 提案手法（NN順伝播＋丸め）の計時 =====================
    nn_ms = []      # TMごとの「1回分」時間 [ms]
    with torch.no_grad():
        # ウォームアップ（外れ値を捨てる）
        for window, _ in measure[: min(CONFIG["warmup"], m)]:
            x = make_features(window, scale)
            _ = round_argmax(model(x))
        # 本計測
        for window, _ in measure:
            r = CONFIG["repeat_nn"]
            t0 = time.perf_counter()
            for _ in range(r):
                x = make_features(window, scale)      # 過去H時刻TM -> 入力ベクトル
                alpha = model(x)                      # 順伝播1回
                _ = round_argmax(alpha)               # argmax丸め（単一SRパス化）
            t1 = time.perf_counter()
            nn_ms.append((t1 - t0) / r * 1e3)         # 1回あたり[ms]

    # ================= MCF理論下限（LP求解）の計時 =========================
    mcf_ms = []
    edges, cap = ds["edges"], ds["cap"]
    # ウォームアップ（HiGHS 初期化など）
    for _, i in measure[: min(CONFIG["warmup"], m)]:
        _ = optimal_mlu_mcf(edges, cap, tms[i], n)
    for _, i in measure:
        t0 = time.perf_counter()
        _ = optimal_mlu_mcf(edges, cap, tms[i], n)
        t1 = time.perf_counter()
        mcf_ms.append((t1 - t0) * 1e3)

    # ---- 集計 -----------------------------------------------------------------
    def summarize(name, xs):
        mean = stats.fmean(xs)
        sd = stats.pstdev(xs) if len(xs) > 1 else 0.0
        med = stats.median(xs)
        print(f"  {name:22s} mean={mean:10.4f} ms  median={med:10.4f} ms  "
              f"sd={sd:9.4f}  (n={len(xs)})")
        return mean, sd

    print("\n=== 計算時間（1経路設定あたり）===")
    nn_mean, nn_sd = summarize("提案手法(NN+丸め)", nn_ms)
    mcf_mean, mcf_sd = summarize("MCF理論下限(LP)", mcf_ms)
    if nn_mean > 0:
        print(f"\n  速度比 MCF/提案手法 = {mcf_mean / nn_mean:,.1f} 倍")

    # ---- 出力（gnuplot 用）---------------------------------------------------
    out_dir = os.path.join(here, "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    # 棒グラフ用: method  mean_ms  std_ms
    with open(os.path.join(out_dir, "timing.dat"), "w") as f:
        f.write("# method  mean_ms  std_ms\n")
        f.write(f"Proposed(NN+round)  {nn_mean:.6f}  {nn_sd:.6f}\n")
        f.write(f"MCF(lower-bound)    {mcf_mean:.6f}  {mcf_sd:.6f}\n")
    # 生値: idx  nn_ms  mcf_ms
    with open(os.path.join(out_dir, "timing_perTM.dat"), "w") as f:
        f.write("# idx  nn_ms  mcf_ms\n")
        for k, ((_, i), a, b) in enumerate(zip(measure, nn_ms, mcf_ms)):
            f.write(f"{k}  {a:.6f}  {b:.6f}\n")
    print(f"\n[out] {os.path.join('results', 'timing.dat')} / timing_perTM.dat 書き出し完了")
    print("[note] 縦軸は桁差が大きいので対数軸を推奨。torch=単スレッド, HiGHS=既定スレッド。")


if __name__ == "__main__":
    main()
