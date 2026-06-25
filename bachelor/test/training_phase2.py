# training_phase2.py

"""
評価関数を変えたやつ
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


# 評価関数（マルチラベル精度）
    # 目的: 精度 (accuracy) を計算する
    # 仕組み: 
        # sigmoid 出力を 0.5 で丸めて予測ラベルにする
        # t（正解ラベル）と比較し，正しい要素数をカウント
        # 全要素数で割って割合を返す
    # 戻り値: [0,1] の範囲の accuracy
def evaluate(loader, model, device, num_paris=12, hops=4, slot_dim=5):
    """
    loader: DataLoader yielding (x, t)
    model: PyTorch model
    Returns: dict with keys
        - element_accuracy: 全要素 (240) ベースのラベル精度
        - per_hop_accuracy: 各ホップのノード ID 一致率（ホップ単位）
        - per_traffic_exact_rate: トラフィック（経路）単位の完全一致率 <-- 推奨最適化指標
    Assumptions: 
        - 出力次元 = num_paris * hops * slot_dim（例: 12*4*5 = 240）
        - ターゲット t は one-hot の float/int 配列（同じ形状）
    """

    model.eval()
    total_elements = 0
    correct_elements = 0

    total_hop_positions = 0
    correct_hop_positions = 0

    total_traffics = 0
    correct_traffics = 0

    with torch.no_grad():
        for x, t in loader:
            x, t = x.to(device), t.to(device)

            y = model(x)

            B = x.size(0)

            # reshape to (B, num_paris, hops, slot_dim)
            per_traffic_len = hops * slot_dim
            out_dim = num_paris * per_traffic_len
            assert y.size(1) == out_dim, f'モデル出力次元が期待と異なる: {y.size(1)} vs {out_dim}'

            y_resh = y.view(B, num_paris, hops, slot_dim)
            t_resh = t.view(B, num_paris, hops, slot_dim)

            # 1) element-wise accuracy
            pred_bin = (0.5 < torch.sigmoid(y)).float()
            correct_elements += (pred_bin == t).sum().item()
            total_elements += t.numel()

            # 2) per-hop node ID (argmax を使う)
            #    -> shape (B, num_pairs, hops)
            pred_ids = torch.argmax(y_resh, dim=3)
            true_ids = torch.argmax(t_resh, dim=3)

            # hop 単位の一致数
            correct_hop_positions += (pred_ids == true_ids).sum().item()
            total_hop_positions += pred_ids.numel()  # B * num_pairs * hops

            # 3) per-traffic (route) exact match: 全ホップが一致したかを判定
            #    per_traffic_exact: shape (B, num_pairs) bool
            per_traffic_exact = (pred_ids == true_ids).all(dim=2)
            correct_traffics += per_traffic_exact.sum().item()
            total_traffics += per_traffic_exact.numel()

    element_accuracy = correct_elements / total_elements
    per_hop_accuracy = correct_hop_positions / total_hop_positions
    per_traffic_exact_rate = correct_traffics / total_traffics

    model.train()
    return {
        "element_accuracy": element_accuracy,
        "per_hop_accuracy": per_hop_accuracy,
        "per_traffic_exact_rate": per_traffic_exact_rate
    }


# エポック数
max_epoch = 20

### ハイパーパラメータ自動調節 ###
def objective(trial):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Optuna が探索するハイパーパラメータ
    num_hidden_layer = trial.suggest_int("num_hidden_layer", 1, 5)
    hidden_dim = trial.suggest_int("hidden_dim", 8, 1024, 8)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, step=1e-5)

    # モデル構築
        # インスタンス化
    net = Net(input_dim=60, hidden_dim=hidden_dim, output_dim=240, num_hidden_layer=num_hidden_layer).to(device)
    
    # 最適化手法（オプティマイザ）
    optimizer = getattr(torch.optim, optimizer)(net.parameters(), lr=learning_rate)


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
            loss = F.binary_cross_entropy_with_logits(y, t.float())

            # 勾配の初期化
            optimizer.zero_grad()

            loss.backward()

            # パラメータの更新
            optimizer.step()
        

        # epoch ごとに validation accuracy を計算
        val_acc = evaluate(val_loader, net, device)
        val_accuracies.append(val_acc)
        train_losses.append(loss.item())
        

    metrics = evaluate(val_loader, net, device, num_paris=12, hops=4, slot_dim=5)
    val_route_exact = metrics["per_traffic_exact_rate"]

    # trial に学習履歴を記録（後で可視化できるように）
    trial.set_user_attr("train_losses", train_losses)
    trial.set_user_attr("val_route_exact", val_accuracies)

    # Optuna に返す指標（最終 epoch の validation accuracy を最大化する）
    return val_route_exact


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
study.optimize(objective, n_trials=20)

best_trial = study.best_trial
print('best_trial: \n', best_trial)
print('=== 最良ハイパーパラメータ ===\n', best_trial.params)
for key, value in best_trial.params.items():
    print(f'{key}: {value}')
print(f'最良の Validation 精度: {best_trial.value:.4f}')


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
metrics = evaluate(test_loader, net, device, num_paris=12, hops=4, slot_dim=5)
print("Test Element-wise accuracy: ", metrics["element_accuracy"])
print("Test Per-hop accuracy:     ", metrics["per_hop_accuracy"])
print("Test Per-traffic exact:    ", metrics["per_traffic_exact_rate"])

"""
test_acc = evaluate(test_loader, net, device)
print(f"Test Accuracy: {test_acc:.4f}")
"""

# ==== モデルを保存 ====
torch.save(net.state_dict(), "best_model.pth")
print("モデルを保存: best_model.pth")


### 学習曲線の可視化 ###
train_losses = best_trial.user_attrs["train_losses"]
val_accuracies = best_trial.user_attrs["val_route_exact"]

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