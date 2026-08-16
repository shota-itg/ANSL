"""
STEP 4 (前半): 提案手法の NN 本体
TM（履歴）を入力に、各ODペアの中継点分配比 alpha [OD, J] を出力する MLP。
DOTE 方式を 2-SR 中継点空間へ持ち込んだもの。出力は行方向 softmax で
「各ODの選択肢（直接＋中継点K個）への分配比」になる。

設計:
  入力次元 in_dim は「NNに何を見せるか」で決まる（train.py の make_features 参照）。
  デフォルトは過去 H 時刻の TM を平坦化した H*n*n 次元。
  出力は OD*J 次元 → [OD, J] に整形 → softmax。

なぜ MLP か:
  基礎評価のスコープでは GNN は対象外（2026-08-10）。まず DOTE と同じく
  全分配比を1本の出力ベクトルで出す素直な MLP で通す。
"""

import torch
import torch.nn as nn


class RelayAllocMLP(nn.Module):
    """TM特徴 -> 中継点分配比 alpha [.., OD, J]。

    forward は単一サンプル [in_dim] でもバッチ [B, in_dim] でも通る。
    """

    def __init__(self, in_dim, od, j, hidden=(256, 256)):
        super().__init__()
        self.od = od
        self.j = j
        dims = [in_dim] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers += [nn.Linear(dims[-1], od * j)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)                                   # [.., OD*J]
        logits = logits.view(*logits.shape[:-1], self.od, self.j)
        return torch.softmax(logits, dim=-1)                   # [.., OD, J]
