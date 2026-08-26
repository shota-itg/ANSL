"""
STEP 4 (後半): 提案手法 NN の学習（DOTE方式）
過去のTM（履歴）を入力にNNが中継点分配比 alpha を出力し、それを「次時刻の
（未知の）TM」に適用したときの MLU を損失として勾配法で直接学習する。
予測とMLUの目的不整合を避ける DOTE の枠組みを 2-SR 中継点空間で行う。
 
実行:
  master/ をカレントにして
    python -m scripts.train
 
【データ】
  scripts.gen_tm_series が生成した合成TM系列（既定 data/synth_abilene）を使う。
  実Abilene同梱の5TMではDOTE式の学習は成立しない（サンプル不足）ため、
  時間相関を持つ合成系列（既定3000枚）で学習・評価する。合成データでの結果は
  「パイプラインと般化の基礎確認」であり、実トラフィックへの一般性の主張ではない。
  data_subdir を "Repetita/data/2016TopologyZooUCL_inverseCapacity" に戻せば
  元の5TMデモに切り替わる。
 
【設計判断：NNに何を入力するか → make_features を差し替える】
  デフォルト: 過去 H 時刻の TM を平坦化して入力（DOTE準拠）。in_dim = H*n*n。
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
    "history": 5,           # 入力に使う過去TM数 H（in_dim = H*n*n）
    "hidden": (256, 256),   # MLP隠れ層
    "epochs": 300,          # 学習エポック（ミニバッチSGD）
    "batch_size": 64,       # ミニバッチサイズ（大量サンプルでのメモリ・速度対策）
    "lr": 1e-3,             # Adam 学習率
    "seed": 0,
    "n_test": 200,          # 系列末尾の何時刻をテストに回すか
    # データ場所（master/ からの相対）。合成系列を既定にする。
    "data_subdir": os.path.join("data", "synth_abilene"),
    "prefix": "Abilene",
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
    data_dir = os.path.join(here, "..", cfg["data_subdir"])
    graph = os.path.join(data_dir, f"{cfg['prefix']}.graph")
    demands_paths = sorted(glob.glob(os.path.join(data_dir, f"{cfg['prefix']}.*.demands")))
    if not demands_paths:
        raise RuntimeError(
            f".demands が見つからない: {data_dir}\n"
            f"  先に python -m scripts.gen_tm_series で合成系列を生成してください。"
        )
 
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
    n_test = min(cfg["n_test"], len(samples) - 1)
    train_samples = samples[:-n_test]
    test_samples = samples[-n_test:]
    in_dim = samples[0][0].numel()
 
    print(f"T={T} TMs, history H={H} -> samples={len(samples)} "
          f"(train={len(train_samples)}, test={len(test_samples)})")
    print(f"OD={OD}, J={J}, in_dim={in_dim} (=H*n*n={H}*{n}*{n}), relays={ds['relays'].tolist()}")
    print(f"batch_size={cfg['batch_size']}, epochs={cfg['epochs']}\n")
 
    model = RelayAllocMLP(in_dim, OD, J, hidden=cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
 
    # --- 学習: 損失 = バッチ平均MLU（DOTEと同じくMLUそのもの）。ミニバッチSGD ----
    #   元は全学習サンプルを1エポックで一括加算していたが、数千サンプルでは
    #   計算グラフが膨れて遅い/メモリ過大。バッチ毎にbackwardするよう変更。
    B = cfg["batch_size"]
    n_train = len(train_samples)
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n_train)
        running = 0.0
        for b0 in range(0, n_train, B):
            idx = perm[b0:b0 + B]
            opt.zero_grad()
            loss = 0.0
            for j in idx.tolist():
                x, dem, _ = train_samples[j]
                alpha = model(x)                     # [OD, J]
                loss = loss + ev.mlu(alpha, dem)
            loss = loss / len(idx)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        if (ep + 1) % max(1, cfg["epochs"] // 10) == 0:
            print(f"  epoch {ep+1:>4}: train平均MLU = {running / n_train:.4f}")
 
    # --- 評価: テストTMで NN / SPF / MCF ----------------------------------------
    print("\n=== テスト評価（末尾{}点の要約）===".format(len(test_samples)))
    model.eval()
    rows = []
    with torch.no_grad():
        for x, dem, i in test_samples:
            alpha = model(x)
            nn_cont = ev.mlu(alpha, dem).item()
            nn_round = ev.mlu(round_argmax(alpha), dem).item()
            spf = spf_mlu(ev, dem)
            mcf = optimal_mlu_mcf(ds["edges"], ds["cap"], tms[i], n) if _HAS_MCF else float("nan")
            rows.append((spf, nn_cont, nn_round, mcf))
 
    r = torch.tensor(rows)                 # [n_test, 4] 列: SPF, NN連続, NN丸め, MCF
    spf_m, cont_m, round_m, mcf_m = r.mean(dim=0).tolist()
    # NN(連続)がSPFを下回った割合、SPFに対する平均改善率
    win = (r[:, 1] < r[:, 0]).float().mean().item()
    impr = ((r[:, 0] - r[:, 1]) / r[:, 0]).mean().item()
    # 到達度: (平均SPF-平均NN)/(平均SPF-平均MCF) … MCFを1、SPFを0とした相対到達
    #   ※per-TMで割るとSPF≈MCFのTMで分母が0に近づき発散するため、集計値で計算する
    if _HAS_MCF:
        denom = spf_m - mcf_m
        reach = (spf_m - cont_m) / denom if abs(denom) > 1e-9 else float("nan")
 
    print(f"  平均MLU  SPF={spf_m:.4f}  NN(連続)={cont_m:.4f}  "
          f"NN(丸め)={round_m:.4f}" + (f"  MCF={mcf_m:.4f}" if _HAS_MCF else ""))
    print(f"  NN(連続)がSPFを下回った割合 = {win:.1%}")
    print(f"  SPFに対する平均改善率(連続) = {impr:.1%}")
    if _HAS_MCF:
        print(f"  SPF→MCF間の平均到達度(連続) = {reach:.1%}  (1.0でMCF最適、0でSPF並み)")

    # --- 図1（平均MLU）と図2（計算時間）用の成果物を書き出す --------------------
    out_dir = os.path.join(here, "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    # (a) 学習済みモデル → measure_time.py が読む checkpoint（順伝播時間の計測用）
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # (b) 図1の棒グラフ用サマリ: method  mean_mlu  std_mlu
    #     提案手法は「丸めのみ」(= NN丸め)。列順は [SPF, NN連続, NN丸め, MCF]
    sd = r.std(dim=0, unbiased=False).tolist()
    with open(os.path.join(out_dir, "mlu_summary.dat"), "w") as f:
        f.write("# method  mean_mlu  std_mlu\n")
        f.write(f"SPF                 {spf_m:.6f}  {sd[0]:.6f}\n")
        f.write(f"Proposed(round)     {round_m:.6f}  {sd[2]:.6f}\n")
        if _HAS_MCF:
            f.write(f"MCF(lower-bound)    {mcf_m:.6f}  {sd[3]:.6f}\n")

    # (c) 図1のエラーバー・点検用: テストTMごとの生値
    with open(os.path.join(out_dir, "mlu_perTM.dat"), "w") as f:
        f.write("# idx  SPF  NN_cont  NN_round  MCF\n")
        for k, (spf_i, cont_i, round_i, mcf_i) in enumerate(rows):
            f.write(f"{k}  {spf_i:.6f}  {cont_i:.6f}  {round_i:.6f}  {mcf_i:.6f}\n")

    print("\n[out] results/model.pt, mlu_summary.dat, mlu_perTM.dat を書き出しました")
 
    print("\n[読み方]")
    print("  NN(連続) が SPF より小さければ、TMを見ずに出したαでも混雑を減らせている。")
    print("  NN(丸め) は単一SRパス化後（推論時相当）。MCF最適が理論下限。")
    if not _HAS_MCF:
        print("  （scipy未導入のためMCF列は省略。pip install scipy で表示される）")
 
 
if __name__ == "__main__":
    main()
 
