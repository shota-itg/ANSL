import torch
import torch.nn as nn
import torch.nn.functional as F

# 3 ノード → 2 ノードの全結合層 (fully-connected layer: fc)
fc = nn.Linear(3, 2)
print(fc)
print(fc.weight)
print(fc.bias)
print()

# 乱数のシードを固定
torch.manual_seed(0)
fc = nn.Linear(3, 2)
print("乱数のシードを固定")
print(fc.weight)
print(fc.bias)
print()

# バージョンの確認
print("バージョンの確認")
print(torch.__version__)
print()


# 線形変換
# データ型をテンソル型に変える
print("線形変換")
x  =torch.tensor([[1., 2., 3.]])
print("テンソル型であることを確認")
print(type(x))
print()

# 線形変換の計算
print("線形変換の計算")
u = fc(x)
print(u)
print()

# 非線形変換の計算
# ReLU関数
    # h = f(u) = max(0, u)
print("非線形変換の計算")
h = F.relu(u)
print(h)
print()

# 目的関数（損失関数）
# 目標値
t = torch.tensor([[1.], [3.]])

# 予測値
y = torch.tensor([[2.], [4.]])

# 平均二乗誤差の算出
print("平均二乗誤差の算出")
F.mse_loss(y, t)
print(F.mse_loss(y, t))
print()