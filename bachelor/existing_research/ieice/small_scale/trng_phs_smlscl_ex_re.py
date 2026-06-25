# trng_phs_smlscl_ex_re.py

"""
モデルの最適化
    隠れ層と学習率の自動最適化
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
writer = SummaryWriter(log_dir='./logs')    # ログ保存先



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



### ネットワークの定義 ###
class MultiSequentialNet(nn.Module):
    def __init__(self, num_hidden_layer, num_blocks):
        super().__init__()

        # nn.ModuleList で複数の Sequential を管理
        self.blocks = nn.ModuleList([build_sequential(num_hidden_layer) for _ in range(num_blocks)])

    def forward(self, inputs):
        outputs = []
        for block in self.blocks:
            outputs.append(block(inputs))   # 各 Sequential に共通の入力 inputs を通す

        return torch.stack(outputs, dim=1)  # shape: [batch_size, 20] -> shape: [batch_size, num_blocks, 20]

def build_sequential(num_hidden_layer):
    layers = [nn.Linear(60, 60)]
    for _ in range(num_hidden_layer):
        layers.append(nn.Linear(60, 60))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(60, 20))
    return nn.Sequential(*layers)



### EarlyStopping ###
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
            print(f'Validation loss decreased ({self.val_loss_min:.12f} --> {val_loss:.12f}). Saving model ...')
        torch.save(model.state_dict(), self.path)   # ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss    # その時の loss を記録



### 評価関数 ###
# 要素ごとの評価
def element_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # shape: [64, 12, 20]
            outputs_reshaped = outputs.reshape(batch_size, 12, 4, 5)
            outputs_softmax = F.softmax(outputs_reshaped, dim=3)
            outputs_argmax = np.argmax(outputs_softmax, 3)  # shape: [64, 12, 4]
            
            correct += (outputs_argmax == labels).sum().item()
            total += labels.numel()
    model.train()

    return correct / total *100

# 経路単位の評価
def path_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # shape: [64, 12, 20]
            outputs_reshaped = outputs.reshape(batch_size, 12, 4, 5)
            outputs_softmax = F.softmax(outputs_reshaped, dim=3)
            outputs_argmax = np.argmax(outputs_softmax, 3) # shape: [64, 12, 4]

            correct += (outputs_argmax == labels).all(dim=2).sum().item()
            total += labels.numel() // labels.shape[2]
    model.train()

    return correct / total *100


# エポック数
max_epoch = 500



### 最適化 ###
def objective(trial):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Optuna が探索するハイパーパラメータ
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, step=1e-5)

    # モデルのインスタンス化
    model = MultiSequentialNet(num_hidden_layer=3, num_blocks=12)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=7, verbose=True)


    ## train ##
    model.train()

    # ログ
    trial_results_train, trial_results_val = {}, {}
    trial_results_train['loss'], trial_results_train['path_accuracy'], trial_results_train['element_accuracy'] = [], [], []
    trial_results_val['loss'], trial_results_val['path_accuracy'], trial_results_val['element_accuracy'] = [], [], []

    for epoch in range(max_epoch):
        print(f'epoch: {epoch}/{max_epoch}')

        model.train()
        for each_batch in train_loader:
            inputs, labels = each_batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)    # shape: [64, 12, 20]
            outputs_reshaped = outputs.reshape(batch_size, 12, 4, 5)
            outputs_reshaped = outputs_reshaped.reshape(batch_size, 48, 5)
            outputs_reshaped = outputs_reshaped.permute(0, 2, 1) # shape: [64, 5, 48]

            labels_reshaped = labels.reshape(batch_size, 48)

            loss_train = criterion(outputs_reshaped, labels_reshaped)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for each_batch in val_loader:
                inputs, labels = each_batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)    # shape: [64, 12, 20]
                outputs_reshaped = outputs.reshape(batch_size, 12, 4, 5)
                outputs_reshaped = outputs_reshaped.reshape(batch_size, 48, 5)
                outputs_reshaped = outputs_reshaped.permute(0, 2, 1) # shape: [64, 5, 48]

                labels_reshaped = labels.reshape(batch_size, 48)

                loss_val = criterion(outputs_reshaped, labels_reshaped)

        # epoch ごとに validation accuracy を計算
        # train
        trial_results_train['loss'].append(loss_train.item())
        train_path_acc = path_accuracy(train_loader, model, device)
        trial_results_train['path_accuracy'].append(train_path_acc)
        train_elem_acc=element_accuracy(train_loader, model, device)
        trial_results_train['element_accuracy'].append(train_elem_acc)
        # val
        trial_results_val['loss'].append(loss_val.item())
        val_path_acc=path_accuracy(val_loader, model, device)
        trial_results_val['path_accuracy'].append(val_path_acc)
        val_elem_acc=element_accuracy(val_loader, model, device)
        trial_results_val['element_accuracy'].append(val_elem_acc)

        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


    ## trial に学習履歴を記録（後で可視化できるように） ##
    # train
    trial.set_user_attr("trial_train_losses", trial_results_train['loss'])
    trial.set_user_attr("trial_train_path_accuracies", trial_results_train['path_accuracy'])
    trial.set_user_attr("trial_train_element_accuracies", trial_results_train['element_accuracy'])
    # val
    trial.set_user_attr("trial_val_losses", trial_results_val['loss'])
    trial.set_user_attr("trial_val_path_accuracies", trial_results_val['path_accuracy'])
    trial.set_user_attr("trial_val_element_accuracies", trial_results_val['element_accuracy'])

    # Optuna に返す指標（最終 epoch の validation accuracy を最大化する）
    return trial_results_val['path_accuracy'][-1]



### main ###
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)  # n_trials: 試行回数

best_trial = study.best_trial
print('best_trial: \n', best_trial)
print('=== ハイパーパラメータ ===\n', best_trial.params)
for key, value in best_trial.params.items():
    print(f'{key}: {value}')
print(f'Best Validation Path Accuracy: {best_trial.value:.4f}')



### 最良パラメータによるモデルの再学習 ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = MultiSequentialNet(num_hidden_layer=3, num_blocks=12).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=best_trial.params["learning_rate"])
early_stopping = EarlyStopping(patience=7, verbose=True)

# train + val をまとめて再学習用に使う
full_train = torch.utils.data.ConcatDataset([train, val])
full_loader = torch.utils.data.DataLoader(full_train, batch_size=64, shuffle=True, drop_last=True)



### trains ###
model.train()

# ログ
best_model_train, best_model_test = {}, {}
best_model_train['loss'], best_model_train['path_accuracy'], best_model_train['element_accuracy'] = [], [], []
best_model_test['loss'], best_model_test['path_accuracy'], best_model_test['element_accuracy'] = [], [], []

for epoch in range(max_epoch):
    print(f'epoch: {epoch}/{max_epoch}')

    model.train()
    for inputs, labels in full_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)   # shape: [64, 12, 20]
        outputs_reshaped = outputs.reshape(batch_size, 12, 4, 5)
        outputs_reshaped = outputs_reshaped.reshape(batch_size, 48, 5)
        outputs_reshaped = outputs_reshaped.permute(0, 2, 1)    # shape: [64, 5, 48]

        labels_reshaped = labels.reshape(batch_size, 48)

        loss_train = criterion(outputs_reshaped, labels_reshaped)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    # テスト用データによる評価
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)   # shape: [64, 12, 20]
            outputs_reshaped = outputs.reshape(batch_size, 12, 4, 5)
            outputs_reshaped = outputs_reshaped.reshape(batch_size, 48, 5)
            outputs_reshaped = outputs_reshaped.permute(0, 2, 1)    # shape: [64, 5, 48]

            labels_reshaped = labels.reshape(batch_size, 48)

            loss_test = criterion(outputs_reshaped, labels_reshaped)

    # train
    best_model_train['loss'].append(loss_train.item())
    train_path_acc=path_accuracy(full_loader, model, device)
    best_model_train['path_accuracy'].append(train_path_acc)
    train_elem_acc=element_accuracy(full_loader, model, device)
    best_model_train['element_accuracy'].append(train_elem_acc)
    # test
    best_model_test['loss'].append(loss_test.item())
    test_path_acc=path_accuracy(test_loader, model, device)
    best_model_test['path_accuracy'].append(test_path_acc)
    test_elem_acc=element_accuracy(test_loader, model, device)
    best_model_test['element_accuracy'].append(test_elem_acc)

    early_stopping(loss_train, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break
    


# デバック
model.eval()
inputs, _ = next(iter(test_loader))
inputs = inputs.to(device)
outputs = model(inputs)

print(type(outputs))
print(outputs.shape)

outputs_reshaped = outputs.reshape(batch_size, 48, 5)

print(f'outputs: \n{outputs_reshaped}') # デバック
print(f'outputs_softmax: \n{torch.softmax(outputs_reshaped, dim=2)}')   # デバック



# ==== test精度を算出 ====
test_path_acc = path_accuracy(test_loader, model, device)
print(f"Test Path Accuracy: {test_path_acc:.4f}")

# ==== モデルを保存 ====
torch.save(model.state_dict(), "best_model.pth")
print("モデルを保存: best_model.pth")



### 学習曲線の可視化 ###
trial_train_losses = best_trial.user_attrs["trial_train_losses"]
trial_train_path_accuracies = best_trial.user_attrs["trial_train_path_accuracies"]
trial_train_element_accuracies = best_trial.user_attrs["trial_train_element_accuracies"]
trial_val_losses = best_trial.user_attrs["trial_val_losses"]
trial_val_path_accuracies = best_trial.user_attrs["trial_val_path_accuracies"]
trial_val_element_accuracies = best_trial.user_attrs["trial_val_element_accuracies"]



plt.figure(figsize=(18, 10))

plt.subplot(3,4,1)
plt.plot(trial_train_losses, label='Trial Train Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3,4,2)
plt.plot(trial_val_losses, label='Trial Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3,4,3)
plt.plot(best_model_train['loss'], label='Best Model Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3,4,4)
plt.plot(best_model_test['loss'], label='Best Model Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3,4,5)
plt.plot(trial_train_path_accuracies, label='Trail Train Path Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Path Accuracy')
plt.legend()

plt.subplot(3,4,6)
plt.plot(trial_val_path_accuracies, label='Trial Val Path Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Path Accuracy')
plt.legend()

plt.subplot(3,4,7)
plt.plot(best_model_train['path_accuracy'], label='Best Model Train Path Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Path Accuracy')
plt.legend()

plt.subplot(3,4,8)
plt.plot(best_model_test['path_accuracy'], label='Best Model Test Path Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Path Accuracy')
plt.legend()

plt.subplot(3,4,9)
plt.plot(trial_train_element_accuracies, label='Trail Train element Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Element Accuracy')
plt.legend()

plt.subplot(3,4,10)
plt.plot(trial_val_element_accuracies, label='Trial Val element Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Element Accuracy')
plt.legend()

plt.subplot(3,4,11)
plt.plot(best_model_train['element_accuracy'], label='Best Model Train element Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Element Accuracy')
plt.legend()

plt.subplot(3,4,12)
plt.plot(best_model_test['element_accuracy'], label='Best Model Test element Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Element Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



plt.figure(figsize=(18,10))

plt.subplot(1,3,1)
plt.plot(trial_train_losses, label='Trial Train Loss')
plt.plot(trial_val_losses, label='Trial Val Loss')
plt.plot(best_model_train['loss'], label='Best Model Train Loss')
plt.plot(best_model_test['loss'], label='Best Model Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,3,2)
plt.plot(trial_train_path_accuracies, label='Trail Train Path Accuracy')
plt.plot(trial_val_path_accuracies, label='Trial Val Path Accuracy')
plt.plot(best_model_train['path_accuracy'], label='Best Model Train Path Accuracy')
plt.plot(best_model_test['path_accuracy'], label='Best Model Test Path Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Path Accuracy')
plt.legend()

plt.subplot(1,3,3)
plt.plot(trial_train_element_accuracies, label='Trail Train element Accuracy')
plt.plot(trial_val_element_accuracies, label='Trial Val element Accuracy')
plt.plot(best_model_train['element_accuracy'], label='Best Model Train element Accuracy')
plt.plot(best_model_test['element_accuracy'], label='Best Model Test element Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Element Accuracy')
plt.legend()

plt.tight_layout()
plt.show()