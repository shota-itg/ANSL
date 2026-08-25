"""
時系列トラフィック行列(TM)の合成生成。
提案手法(DOTE方式・TM未知環境での2-SR中継点選択)の学習・評価に足る量の
「時間相関を持つTM系列」を人工生成するためのモジュール。

背景（なぜ必要か）:
  REPETITA同梱のAbilene需要は5枚の独立スナップショットで、時系列性が無く、
  DOTE方式(過去TM履歴→次時刻の経路)を学習・般化させるにはサンプルが足りない。
  実測の全系列(GEANT 10,772枚等)は形式・トポロジ突き合わせのコストが高いので、
  ここでは「空間構造(重力モデル)＋時間変動」で時間相関を持つ系列を合成し、
  提案手法のパイプラインと般化の基礎挙動を確認できるデータを作る。

位置づけ（予稿・発表で正直に書くべき点）:
  これは実データではない。合成系列でNNが良い結果を出しても、それは
  「作り込んだ時間変動則をNNが学べた」ことの確認であり、実トラフィックへの
  一般性を主張するものではない。基礎評価＝パイプラインと般化の成立確認、
  実データ検証は今後の課題、という線引きを明示する前提で使う。

モデル（Modulated Gravity Model 系）:
  1) 重力モデルで空間構造を作る。ノード o の流出規模 out[o]、ノード d の流入規模
     in[d] を与え、基準TM  T0[o,d] ∝ out[o]*in[d]（対角0）。大拠点同士が多くの
     トラフィックを交換する、という空間的偏りを与える。
  2) 時間変動を重ねて系列にする。各時刻 t について
       - 日内変動(diurnal): 全体量に滑らかな正弦変動  g(t)=1+A_d*sin(2πt/P+φ)
       - 緩やかなドリフト  : ノード規模の対数を平均回帰AR(1)(Ornstein-Uhlenbeck型)で
                              初期水準へ引き戻しつつ揺らす。隣接TMは近い(連続性)が、
                              純ランダムウォークと違い長期的に発散せず定常に保たれる
                              → 学習分布とテスト分布が系統的にずれない(末尾外挿を避ける)
       - 短時間ノイズ      : OD毎・時刻毎に乗算的な対数正規ノイズ（小さなジッタ）
       - スパイク(任意)    : 稀に一部ODまたは全体を短時間だけ増幅（突発トラフィック）
     これにより「隣接TMの連続性は高いが、level shift とスパイクも混じる」という
     実TM系列の定性的特徴（学部研究時の観察と整合）を再現する。

スケール校正:
  絶対値には意味が無く、MLU は load/cap の比なので単位系に依存しない。
  ただし生成量が小さすぎると全リンク利用率が1未満に収まり、SPFでも混雑せず、
  2-SR/MCF に改善余地が生まれない（全手法が同じに見える）。そこで
  「SPFのMLUの時間中央値が目標値(既定1.2、実Abilene 1.221 に近い水準)になる」
  ように系列全体を定数倍する calibrate_to_spf_mlu を用意する。
"""

import math
import os
import shutil

import torch


# ---------------------------------------------------------------------------
# 1. 空間構造（重力モデル）
# ---------------------------------------------------------------------------
def sample_node_sizes(n, seed, sigma=0.6):
    """各ノードの流出/流入規模を対数正規で引く（拠点規模の偏りを表現）。

    Returns: (out_w [n], in_w [n])  いずれも正、平均1付近。
    """
    g = torch.Generator().manual_seed(seed)
    out_w = torch.exp(sigma * torch.randn(n, generator=g))
    in_w = torch.exp(sigma * torch.randn(n, generator=g))
    return out_w, in_w


def gravity_matrix(out_w, in_w):
    """重力モデルの基準TM  T0[o,d] ∝ out_w[o]*in_w[d]（対角0、総和1に正規化）。"""
    n = out_w.shape[0]
    T = out_w.unsqueeze(1) * in_w.unsqueeze(0)      # [n, n]
    T = T.clone()
    T.fill_diagonal_(0.0)
    T = T / T.sum()
    return T


# ---------------------------------------------------------------------------
# 2. 時間変動を重ねて系列にする
# ---------------------------------------------------------------------------
def generate_series(
    n,
    T,
    seed=0,
    period=96,            # 日内周期のステップ数（15分刻みなら 96=1日）
    amp_diurnal=0.35,     # 日内変動の振幅（全体量の±35%）
    drift_std=0.03,       # ノード規模の対数ドリフトの1ステップ揺らぎ標準偏差
    reversion=0.02,       # 平均回帰の強さ κ（0で純ランダムウォーク=非定常、大きいほど初期水準に固定）
    noise_std=0.05,       # OD毎・時刻毎の乗算ノイズ（対数正規）標準偏差
    spike_prob=0.01,      # 各時刻でスパイクが起きる確率
    spike_scale=2.5,      # スパイク時の増幅率
    size_sigma=0.6,       # 拠点規模の偏りの大きさ
):
    """時間相関を持つTM系列 [T, n, n] を生成する。

    - period, amp_diurnal : 滑らかな日内変動（連続性の主因）
    - drift_std, reversion: 空間構造の緩やかな移り変わり。平均回帰付きなので
                            level shift は起きるが長期的には初期水準へ戻り定常
    - noise_std           : 短時間のジッタ
    - spike_prob/scale    : 稀な突発トラフィック
    絶対スケールは未校正（calibrate_to_spf_mlu で後段調整）。
    """
    g = torch.Generator().manual_seed(seed)

    out_w0, in_w0 = sample_node_sizes(n, seed, sigma=size_sigma)
    log_out0 = torch.log(out_w0).clone()   # 平均回帰の目標水準
    log_in0 = torch.log(in_w0).clone()
    log_out = log_out0.clone()
    log_in = log_in0.clone()

    tms = torch.zeros(T, n, n)
    for t in range(T):
        # (a) 規模を平均回帰AR(1)で動かす（緩やかなドリフト、長期は定常）
        #     Δlog = -κ(log - log0) + drift_std·ε   → 初期水準へ引き戻しつつ揺らぐ
        log_out += -reversion * (log_out - log_out0) + drift_std * torch.randn(n, generator=g)
        log_in += -reversion * (log_in - log_in0) + drift_std * torch.randn(n, generator=g)
        out_w = torch.exp(log_out)
        in_w = torch.exp(log_in)

        # (b) その時刻の空間構造（重力モデル）
        base = gravity_matrix(out_w, in_w)          # 総和1

        # (c) 日内変動（全体量の滑らかな増減）
        diurnal = 1.0 + amp_diurnal * math.sin(2 * math.pi * t / period)

        # (d) OD毎の乗算ノイズ（対数正規）
        noise = torch.exp(noise_std * torch.randn(n, n, generator=g))
        noise.fill_diagonal_(0.0)

        tm = base * diurnal * noise

        # (e) スパイク（稀に一部ODを短時間増幅）
        if torch.rand(1, generator=g).item() < spike_prob:
            mask = (torch.rand(n, n, generator=g) < 0.1).float()
            mask.fill_diagonal_(0.0)
            tm = tm * (1.0 + (spike_scale - 1.0) * mask)

        tm.fill_diagonal_(0.0)
        tms[t] = tm

    return tms


# ---------------------------------------------------------------------------
# 3. SPFのMLUに合わせてスケール校正
# ---------------------------------------------------------------------------
def spf_mlu_of_tm(F, cap, tm):
    """全ODを直接ECMP最短で流したときのMLU（SPFベースライン）。

    F   : [n, n, E]  ecmp_unit_flows の出力（単位需要リンクフロー）
    cap : [E]
    tm  : [n, n]
    """
    load = torch.einsum("ste,st->e", F, tm)   # [E]
    util = load / cap
    return util.max().item()


def calibrate_to_spf_mlu(tms, topo, target=1.2):
    """系列全体を定数倍し、SPFのMLUの時間中央値を target に合わせる。

    MLU は tm に線形なので、時間中央値のSPF-MLUに対する倍率 = target/median。
    Returns: (校正後 tms, 使った倍率, 校正前のSPF-MLU系列[T])
    """
    from src.routing import ecmp_unit_flows

    F, _ = ecmp_unit_flows(topo)
    cap = topo["cap"]
    mlus = torch.tensor([spf_mlu_of_tm(F, cap, tms[t]) for t in range(tms.shape[0])])
    med = mlus.median().item()
    if med <= 0:
        raise RuntimeError("SPF-MLUの中央値が0。生成が縮退している可能性。")
    factor = target / med
    return tms * factor, factor, mlus


# ---------------------------------------------------------------------------
# 4. REPETITA .demands 形式で書き出し
# ---------------------------------------------------------------------------
def write_demands_series(tms, out_dir, prefix="Abilene", pad=5, graph_src=None):
    """TM系列を REPETITA .demands 形式で1時刻1ファイルに書き出す。

    ファイル名: {prefix}.{00000}.demands …（ゼロ詰めで sorted() が時刻順になる）
    形式:
        DEMANDS <本数>
        label src dest bw
        demand_0 <s> <d> <bw>
        ...
    graph_src を渡すと、その .graph を out_dir に {prefix}.graph としてコピーし、
    合成データだけで自己完結したディレクトリにする（train 側は data_dir を
    ここへ向けるだけで済む）。
    """
    os.makedirs(out_dir, exist_ok=True)
    T, n, _ = tms.shape

    if graph_src is not None:
        shutil.copyfile(graph_src, os.path.join(out_dir, f"{prefix}.graph"))

    for t in range(T):
        tm = tms[t]
        # 全 s!=t を書き出す（重力モデルは全ODが正）。0のODは書かない。
        lines = []
        idx = 0
        body = []
        for s in range(n):
            for d in range(n):
                if s == d:
                    continue
                bw = tm[s, d].item()
                if bw <= 0:
                    continue
                body.append(f"demand_{idx} {s} {d} {bw:.6g}")
                idx += 1
        lines.append(f"DEMANDS {idx}")
        lines.append("label src dest bw")
        lines.extend(body)
        fname = os.path.join(out_dir, f"{prefix}.{t:0{pad}d}.demands")
        with open(fname, "w") as f:
            f.write("\n".join(lines) + "\n")

    return T
