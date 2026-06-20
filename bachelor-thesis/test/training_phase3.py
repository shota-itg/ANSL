# training_phase3.py

"""
train : val : test = 6 : 2 : 2
BS=64
NNの構成
    input : hidden : outpur = 64 : 60~1440 : 240
EarlyStopping
"""

import csv
from sklearn.utils import Bunch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import matplotlib.pyplot as plt


### データの取得 ###
print("### データの取得 ###")
print()
x = []
t = []
before_label, label, label_argmax = [], [], []

with open("traffic_one_hot_log.csv", "r") as f:
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
print("### DataLoader の定義 ###")
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

# シードを固定し，再現性を確保
torch.manual_seed(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])


### ミニバッチ学習 ###
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layer):
        super().__init__()
        layers = []

        # 入力層
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 中間層を num_hidden_layer 個を追加
        for _ in range(num_hidden_layer):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # 出力層
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Sequential でまとめる
        self.model = nn.Sequential(*layers)

    # 順伝播
    def forward(self, x):
        return self.model(x)


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
    model.train()

    return correct / total


# エポック数
max_epoch = 500

### ハイパーパラメータ自動調節 ###
def objective(trial):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Optuna が探索するハイパーパラメータ
    num_hidden_layer = trial.suggest_int("num_hidden_layer", 1, 5)
    hidden_dim = trial.suggest_int("hidden_dim", 60, 1440, 60)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, step=1e-5)

    # モデル構築
        # インスタンス化
    net = Net(input_dim=60, hidden_dim=hidden_dim, output_dim=240, num_hidden_layer=num_hidden_layer).to(device)

    # 損失関数
    criterion = nn.CrossEntropyLoss()
    
    # 最適化手法（オプティマイザ）
    optimizer = getattr(torch.optim, optimizer)(net.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=7, verbose=True)

    ### train ###
    net.train()
    
    train_losses = []
    val_accuracies = []

    # 学習ループ
    for epoch in range(max_epoch):
        for batch in train_loader:
            # batch を入力値 x と目標値 t に分ける
            x, t = batch
            
            x, t = x.to(device), t.to(device)

            y = net(x)

            # 損失関数（目的関数）
            loss = criterion(y, t.float())

            # 勾配の初期化
            optimizer.zero_grad()

            loss.backward()

            # パラメータの更新
            optimizer.step()
        
        # epoch ごとに validation accuracy を計算
        val_acc = evaluate(val_loader, net, device)
        val_accuracies.append(val_acc)
        train_losses.append(loss.item())

        early_stopping(loss, net)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
    # trial に学習履歴を記録（後で可視化できるように）
    trial.set_user_attr("train_losses", train_losses)
    trial.set_user_attr("val_accuracies", val_accuracies)

    # Optuna に返す指標（最終 epoch の validation accuracy を最大化する）
    return val_accuracies[-1]


    """
    ### validation ###
        # 目的: 損失 (loss) を計算する
        # 仕組み: 
            # binary_cross_entropy_with_logits で損失を算出
            # バッチごとの損失を合計し，データ数で割る
        # 戻り値: スカラーの loss 値
    net.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            y = net(x)
            loss = F.binary_cross_entropy_with_logits(y, t.float())
            validation_loss += loss.item() * x.size(0)

        validation_loss = validation_loss / len(test_loader.dataset)
    """


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_trial = study.best_trial
print('best_trial: \n', best_trial)
print('=== ハイパーパラメータ ===\n', best_trial.params)
for key, value in best_trial.params.items():
    print(f'{key}: {value}')
print(f'Best validation accuracy: {best_trial.value:.4f}')


### 最良パラメータによるモデルの再学習 ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = Net(input_dim=60, hidden_dim=best_trial.params["hidden_dim"], output_dim=240, num_hidden_layer=best_trial.params["num_hidden_layer"]).to(device)

optimizer = getattr(torch.optim, best_trial.params["optimizer"])(net.parameters(), lr=best_trial.params["learning_rate"])

# train + val をまとめて再学習用に使う
full_train = torch.utils.data.ConcatDataset([train, val])
full_loader = torch.utils.data.DataLoader(full_train, batch_size=64, shuffle=True)

# trains
net.train()
for epoch in range(max_epoch):
    for x, t in full_loader:
        x, t = x.to(device), t.to(device)
        y = net(x)
        loss = F.binary_cross_entropy_with_logits(y, t.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ==== test精度を算出 ====
test_acc = evaluate(test_loader, net, device)
print(f"Test Accuracy: {test_acc:.4f}")

# ==== モデルを保存 ====
torch.save(net.state_dict(), "best_model.pth")
print("モデルを保存: best_model.pth")


### 学習曲線の可視化 ###
train_losses = best_trial.user_attrs["train_losses"]
val_accuracies = best_trial.user_attrs["val_accuracies"]

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





"""

# シードを固定し，再現性を確保
torch.manual_seed(0)

early_stopping = EarlyStopping()

"""