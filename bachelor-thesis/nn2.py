import csv
from sklearn.utils import Bunch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

### データの取得 ###
x = []
t = []

with open("traffic_log_one_hot.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader) # ヘッダーをスキップ

    # データの長さを推定（ヘッダーから）
    data_len = len([h for h in header if h.startswith("data_")])
    target_len = len([h for h in header if h.startswith("target_")])

    for row in reader:
        data = list(map(int, row[:data_len]))   # 必要なら int を float などに変更

        target = list(map(int, row[data_len:data_len + target_len]))
        x.append(data)
        t.append(target)

# PyTorch の Tensor 型へ変換
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)


### DataLoader の定義 ###
    # ミニバッチ学習に必要な処理
# 入力値 x と目標値 t をまとめる
dataset = torch.utils.data.TensorDataset(x, t)

# データセットを分割
    # 学習データtrain: モデルのパラメータの最適化
    # 検証データval: モデルのハイパーパラメータの最適化
    # テストデータtest: 学習済みモデルの評価
# 各データのサンプル数を決定
# train : val : test = 60% : 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val
print("### データセットの分割数を決定 ###")
print("train : val : test = 60% : 20% : 20%")
print(f'(train, val, test) = ({n_train}, {n_val}, {n_test})')
print()

# シードを固定し，再現性を確保
torch.manual_seed(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
print("### データセットの分割 ###")
print(f'(train, val, test) = ({str(len(train))}, {str(len(val))}, {str(len(test))})')
print()


### ミニバッチ学習 ###
print("### ミニバッチ学習 ###")

# バッチサイズの定義
batch_size = 64
print(f'バッチサイズ: {batch_size}')

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True) # shuffle: 局所解を抜け出すために，各 epoch でデータをランダムにシャフル    # drop_last: データの最後の余りを除外
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)


### ネットワークの定義 ###
class Net(nn.Module):
    # 入力層: P(N, 2) * (N+1)
    # 中間層: (P(N, 2) * (N+1)) * P(N, 2)
    # 出力層: P(N, 2) * (N+1) * N

    # 使用するオブジェクトを定義
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(60, 720)
        self.fc2 = nn.Linear(720, 240)

    # 順伝播
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)

        return h

# 評価関数（マルチラベル制度）
def evaluate(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, t in loader:
            x, t = x.to(device), t.to(device)
            y = model(x)
            pred = (0.5 < torch.sigmoid(y)).float()
            correct += (pred == t).sum().item()
            total += t.numel()

    return correct / total

def main(mode='train'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # シードを固定し，再現性を確保
    torch.manual_seed(0)
    
    # インスタンス化
    net = Net().to(device)

    if mode == 'train':
        # 最適化手法
            # 最急降下法のSGD
        optimizer = torch.optim.SGD(net.parameters(), lr=0.05)  # lr: 学習係数ρ
        
        # エポック数
        max_epoch = 50


        ### 学習ループ ###
        train_losses = []
        val_accuracies = []

        for epoch in range(max_epoch):
            for batch in train_loader:
                # batch を入力値 x と目標値 t に分ける
                x, t = batch

                x = x.to(device)
                t = t.to(device)

                y = net(x)

                # 損失関数（目的関数）
                loss = F.binary_cross_entropy_with_logits(y, t.float())

                # 勾配の初期化
                optimizer.zero_grad()

                loss.backward()

                # パラメータを更新
                optimizer.step()
    
            train_losses.append(loss.item())
            val_acc = evaluate(val_loader, net, device)
            val_accuracies.append(val_acc)
            print(f'Epoch {epoch+1}, Loss: {loss.item(): .4f}, Val Accuracy: {val_acc: .4f}')

        torch.save(net.state_dict(), 'model.pth')
        print('モデルを "model.pth" に保存完了')


        ### 学習曲線の可視化 ###
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    elif mode == 'eval':
        if not os.path.exists('model.pth'):
            print('学習済みモデル "model.pth" が見つかりません。')

            return

        net.load_state_dict(torch.load('model.pth'))    # 保存したパラメータを読み込む
        net.eval()  # 評価モードに切り替え
        test_acc = evaluate(test_loader, net, device)


        ### テスト精度の表示 ###
        test_acc = evaluate(test_loader, net, device)
        print(f'Test Accuracy: {test_acc: .4f}')

if __name__ == '__main__':
    which_mode = input('選択肢: 学習(train) or 評価(eval)\nmode: ')
    main(mode=which_mode)  # 'train' または 'eval' に切り替え可能