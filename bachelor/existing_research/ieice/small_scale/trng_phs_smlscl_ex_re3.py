# trng_phs_smlscl_ex_re.py

"""

"""

import csv
from sklearn.utils import Bunch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


### データの取得 ###
inputs, labels, labels_reshaped = [], [], []

with open("traffic_one_hot_log.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader) # ヘッダーをスキップ

    # データの長さを推定（ヘッダーから）
    data_len = len([h for h in header if h.startswith("data_")])
    target_len = len([h for h in header if h.startswith("target_")])

    for row in reader:
        data = list(map(int, row[:data_len]))   # 必要なら int を float などに変更

        target = list(map(int, row[data_len:data_len + target_len]))
        inputs.append(data)
        labels.append(target)

labels = np.array(labels)   # shape: [10000, 240]
labels_reshaped = labels.reshape(10000, 12, 4, 5)
labels_argmax = np.argmax(labels_reshaped, 3)   # shape: [10000, 12, 4]

# PyTorch の Tensor 型へ変換
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels_argmax, dtype=torch.int64)

print(inputs.shape)
print(labels.shape)



### DataLoader の定義 ###
    # ミニバッチ学習に必要な処理
# 入力値 x と目標値 t をまとめる
dataset = torch.utils.data.TensorDataset(inputs, labels)

# データセットを分割
    # 学習データtrain: モデルのパラメータの最適化
    # 検証データval: モデルのハイパーパラメータの最適化
    # テストデータtest: 学習済みモデルの評価
# 各データのサンプル数を決定
# train : val : test = 60% : 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - (n_train + n_val)

# シードの固定
    # 再現性を確保
torch.manual_seed(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])



### ミニバッチ学習 ###
# バッチサイズの定義
batch_size = 64
print(f'バッチサイズ: {batch_size}')

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True) # shuffle: 局所解を抜け出すために，各 epoch でデータをランダムにシャフル    # drop_last: データの最後の余りを除外
val_loader = torch.utils.data.DataLoader(val, batch_size, drop_last=True)
test_loader = torch.utils.data.DataLoader(test, batch_size, drop_last=True)



####################################################
### ネットワークの定義 ###
class MultiSequentialNet(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()

        # nn.ModuleList で複数の Sequential を管理
        self.blocks = nn.ModuleList([build_sequential() for _ in range(num_blocks)])

    def forward(self, inputs):
        outputs = []
        for block in self.blocks:
            outputs.append(block(inputs))   # 各 Sequential に共通の入力 inputs を通す

        return torch.stack(outputs, dim=1)  # shape: [batch_size, 20] -> shape: [batch_size, num_blocks, 20]

def build_sequential():
    return nn.Sequential(
        nn.Linear(60, 60), 
        nn.ReLU(), 
        nn.Linear(60, 60), 
        nn.ReLU(), 
        nn.Linear(60, 20)
    )


class EarlyStopping:

    def __init__(self, patience=5, verbose=False, path='chekpoint_model.pth'):
        # patience: 最小値の非更新数カウンタ    # verbose: 表示設定    # path: モデルの格納path
        """
        Args: 
            patience (int): 
            verbose (bool): 
            delta (float): 
            path (str): 
            trace_func
        """

        self.patience = patience    # 設定ストップカウンタ
        self.verbose = verbose  # 表示の有無
        self.counter = 0    # 現在のカウンタ値
        self.best_val_loss = None  # ベストスコア
        self.early_stop = False # ストップフラグ
        self.val_loss_min = np.inf  # 前回のベストスコア記録用
        self.path = path    # ベストモデル格納path

    """
    特殊 (call) メソッド
    実際に学習ループ内で最小 loss を更新したか否かを計算させる部分
    """
    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss.item()):
            print("Validation loss is NaN. Ignoring this epoch.")
            self.early_stop = True
            return

        if self.best_val_loss is None: # 1Epoch 目の処理
            self.best_val_loss = val_loss # 1Epoch 目はそのままベストスコアとして記録
            self.save_checkpoint(val_loss, model) # 記録後にモデルを保存してスコアを表示
        elif val_loss < self.best_val_loss:   # ベストスコアを更新した場合
            self.best_val_loss = val_loss # ベストスコアを上書き
            self.save_checkpoint(val_loss, model)    # モデルを保存してスコアを表示
            self.counter = 0    # ストップカウンタリセット
        else:   # ベストスコアを更新できなかった場合
            self.counter += 1   # ストップカウンタを +1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # 現在のカウンタを表示
            if self.patience <= self.counter:   # 設定カウンタを上回ったらストップフラグを True に変更
                self.early_stop = True

    """
    ベストスコア更新時に実行されるチェックポイント関数
    """
    def save_checkpoint(self, val_loss, model):
        if self.verbose:    # 表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)   # ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss    # その時の loss を記録


# 評価関数（マルチラベル精度）
    # 目的: 精度 (accuracy) を計算する
    # 仕組み: 
        # sigmoid 出力を 0.5 で丸めて予測ラベルにする
        # t（正解ラベル）と比較し，正しい要素数をカウント
        # 全要素数で割って割合を返す
    # 戻り値: [0,1] の範囲の accuracy
def element_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # shape: [64, 240]

            labels_to_one_hot = torch.nn.functional.one_hot(labels, num_classes=5)  # shape: [64, 48, 5]
            labels_reshaped = labels_to_one_hot.reshape(64, 240)
            
            pred = (0.5 < torch.sigmoid(outputs)).int()
            
            correct += (pred == labels_reshaped).sum().item()
            total += labels.numel()
    model.train()

    return correct / total

# 評価関数
def path_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # shape: [64, 12, 20]
            outputs_reshaped = outputs.reshape(64, 12, 4, 5)
            outputs_argmax = np.argmax(outputs_reshaped, 3) # shape: [64, 12, 4]

            correct += (outputs_argmax == labels).all(dim=2).sum().item()
            total += labels.numel() // labels.shape[2]
    model.train()

    return correct / total

# エポック数
max_epoch = 500



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# モデルのインスタンス化
net = MultiSequentialNet(num_blocks=12)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

early_stopping = EarlyStopping(patience=7, verbose=True)



### train ###
net.train()

train_losses = []
val_accuracies = []

for epoch in range(max_epoch):
    for each_batch in train_loader:
        inputs, labels = each_batch
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)    # shape: [64, 12, 20]
        outputs_reshaped = outputs.reshape(64, 12, 4, 5)
        outputs_reshaped = outputs_reshaped.reshape(64, 48, 5)
        outputs_reshaped = outputs_reshaped.permute(0, 2, 1) # shape: [64, 5, 48]

        labels_reshaped = labels.reshape(64, 48)

        loss = criterion(outputs_reshaped, labels_reshaped)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # epoch ごとに validation accuracy を計算
    val_acc = path_accuracy(val_loader, net, device)
    val_accuracies.append(val_acc)

    train_losses.append(loss.item())

    early_stopping(loss, net)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break


# ==== test精度を算出 ====
test_acc = path_accuracy(test_loader, net, device)
print(f"Test Accuracy: {test_acc:.4f}")

# ==== モデルを保存 ====
torch.save(net.state_dict(), "best_model.pth")
print("モデルを保存: best_model.pth")



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