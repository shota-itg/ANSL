import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris

iris = load_iris()

# 入力値と目標値を抽出
x = iris['data']    # 入力値
t = iris['target']  # 目標値
print(f"irisの中身: \n{iris}")
print()

print(f"x: \n{x}")
print()

print(f"t: \n{t}")
print

print("入力値 x と目標値 t のデータ型を確認")
print(type(x), type(t))
print()

# Pytorch の Tensor 型へ変換
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

print("入力値 x と目標値 t のデータ型を確認")
print(type(x), type(t))
print()

print("入力値 x と目標値 t のサイズを確認")
print(x.shape, t.shape)
print()


## DataLoader ##
# 入力値と目標値をまとめる
dataset = torch.utils.data.TensorDataset(x, t)
print("入力値と目標値をまとめた dataset の長さを確認")
print(len(dataset))
print()

print("入力値と目標値をまとめた dataset の 1 つ目を確認")
print(dataset[0])
print()


## データセット分割 ##
    # 学習データ: モデルのパラメータの最適化
    # 検証データ: モデルのハイパーパラメータの最適化
    # テストデータ: 学習済みモデルの評価
print("## データセット分割 ##")
# 各データのサンプル数を決定
# train : val : test = 60% : 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val

print("train : val : test = 60% : 20% : 20%")
print(f"(train, val, test) = ({n_train}, {n_val}, {n_test})")
print()

torch.manual_seed(0)

# データセットの分割
print("# データセットの分割")
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
print("len(train, val, test) = (" + str(len(train)) + ", " + str(len(val)) + ", " + str(len(test)) + ")")
print()


## ミニバッチ学習 ##
# バッチサイズの定義
batch_size = 10

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True) # 学習の時はshuffle=True    # drop_last: 最後の余りを除去
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)

x, t = next(iter(train_loader))
print("trainの入力値 x: ")
print(x)
print("trainの目標値 t: ")
print(t)
print()


## ネットワークの定義 ##
# 4 → 4 → 3 の全結合層を定義
class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)

    # 順伝播
    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        return h




torch.manual_seed(0)
# インスタンス化
net = Net()
print("net: ")
print(net)
print()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

batch = next(iter(train_loader))
print("batch: ")
print(batch)
print()

x, t = batch
print("trainの入力値 x: ")
print(x)
print("trainの目標値 t: ")
print(t)
print()

# 予測値 y の算出
print("# 予測値 y 算出")
y = net.forward(x)
print(f"y: \n{y}")
print()

loss = F.cross_entropy(y, t)
print(f"loss: \n{loss}")
print()

# 勾配を算出
loss.backward()

# パラメータの更新
optimizer.step()

torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
x = x.to(device)
t = t.to(device)

# 勾配情報の初期化
optimizer.zero_grad()





# エポック数
max_epoch = 1

torch.manual_seed(0)

# モデルのインスタンス化とデバイスへの転送
net = Net().to(device)

# 最適化手法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

## 学習のループ ##
print("学習のループ")
for epoch in range(max_epoch):
    for batch in train_loader:
        x, t = batch

        x = x.to(device)
        t = t.to(device)

        y = net(x)

        loss = F.cross_entropy(y, t)

        print(f'loss: {loss}')

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
print()





## 評価指標の追加 ##
x, t = next(iter(train_loader))
x = x.to(device)
t = t.to(device)
y = net(x)
print(f'y: \n{y}')
print()

y_label = torch.argmax(y, dim=1)
print(f'y_label: \n{y_label}')
print()

print(f'y_label == t: \n{y_label == t}')
print()

print(f'(y_label == t).sum(): \n{(y_label == t).sum()}')
print()

accuracy = (y_label == t).sum().float() / len(t)
print(f'accuracy: \n{accuracy}')
print()





# モデルの初期化
torch.manual_seed(0)

# モデルのインスタンス化とデバイスへの転送
net = Net().to(device)

# 最適化手法の選択
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

## 学習のループ ##
print("学習のループ")
for epoch in range(max_epoch):
    for batch in train_loader:
        x, t = batch

        x = x.to(device)
        t = t.to(device)

        y = net(x)

        loss = F.cross_entropy(y, t)

        # 正解率追加
        y_laber = torch.argmax(y, dim=1)
        accuracy = (y_label == t).sum().float() / len(t)
        print(f'accuracy: {accuracy:.2f}')

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
print()