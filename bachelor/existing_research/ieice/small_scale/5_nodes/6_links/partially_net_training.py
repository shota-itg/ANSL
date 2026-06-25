# trng_phs_smlscl_ex_re.py

"""
トポロジー
    5-nodes
    6-links

Network
    部分結合層
    
ハイパーパラメータ
    隠れ層: 3層
    活性化関数: 
        中間層: ReLU 関数
        出力層: SoftMax 関数
    損失関数: 多クラス交差エントロピー
    最適化アルゴリズム: Adamax
        学習率: 0.002 （初期値）
"""

import csv
import math
from sklearn.utils import Bunch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR

import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./logs')    # ログ保存先


# ハイパーパラメータ
N = 5   # トポロジのノード数 N
NP_2 = math.perm(N, 2)  # ノード間ペア数 P(N,2)
num_blocks = NP_2   # NN の構成数 P(N,2)
num_hidden_layer = 3
batch_size = 64    # バッチサイズ

patience = 5     # Early Stopping の回数



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

labels = np.array(labels)   # shape: [10000, 240] <-- shape: [len(dataset), (N+1)*N*P(N,2)]
labels_reshaped = labels.reshape(len(labels), NP_2, N, (N+1))   # shape: [len(dataset), P(N,2), N, (N+1)]
labels_argmax = np.argmax(labels_reshaped, 3)   # shape: [10000, 30, 6] shape: [len(dataset), P(N,2), N]

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
n_train = int(len(dataset) * 0.4)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - (n_train + n_val)

# シードの固定
    # 再現性を確保
torch.manual_seed(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])



### ミニバッチ学習 ###
# バッチサイズの定義
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True) # shuffle: 局所解を抜け出すために，各 epoch でデータをランダムにシャフル    # drop_last: データの最後の余りを除外
val_loader = torch.utils.data.DataLoader(val, batch_size, drop_last=True)
test_loader = torch.utils.data.DataLoader(test, batch_size, drop_last=True)



### ネットワークの定義 ###
class PartiallyConnectedLayerNet(nn.Module):
    """
    input_layer: (N+1) * P(N,2)
    hidden_layer: ((N+1) * P(N,2)) * P(N,2)
    output_layer: ((N+1)*N) * P(N,2)
    num_blocks: P(N,2)
    """
    
    def __init__(self):
        super().__init__()

        # nn.ModuleList で複数の Sequential を管理
        self.blocks = nn.ModuleList([build_sequential(num_hidden_layer) for _ in range(num_blocks)])

    def forward(self, inputs):
        outputs = []
        for block in self.blocks:
            outputs.append(block(inputs))   # 各 Sequential に共通の入力 inputs を通す

        return torch.stack(outputs, dim=1)  # shape: [batch_size, 20] -> shape: [batch_size, num_blocks, 20]

def build_sequential(num_hidden_layer):
    layers = [nn.Linear(((N+1)*NP_2), ((N+1)*NP_2))]  # <--
    for _ in range(num_hidden_layer):
        layers.append(nn.Linear(((N+1)*NP_2), ((N+1)*NP_2)))  # <--
        layers.append(nn.ReLU())
    layers.append(nn.Linear(((N+1)*NP_2), ((N+1)*N)))   # <--
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
# トラフィックデマンド集合単位の評価
def demand_accuracy(loader, model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))
            outputs_softmax = F.softmax(outputs_reshaped, dim=3)
            outputs_argmax = torch.argmax(outputs_softmax, 3)   # shape: [batch_size, P(N,2), N]

            correct += ((outputs_argmax == labels).all(dim=2)).all(dim=1).sum().item()
        model.train()
        print(f'> デバック    == Demand  ==    Demand Correct: {correct}       Demand Accuracy: {correct/(len(loader.dataset)) *100}    データセット数: {len(loader.dataset)} ({len(loader.dataset)})')
        return correct / (batch_size*len(loader)) *100

# 経路単位の評価
def path_accuracy(loader, model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # shape: [128, 30, 42] <-- shape: [batch_size, P(N,2), ((N+1)*N)]
            outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))    # reshape(batch_size, P(N,2), N, (N+1)) <--
            outputs_softmax = F.softmax(outputs_reshaped, dim=3)
            outputs_argmax = torch.argmax(outputs_softmax, 3) # shape: [128, 30, 6] <-- shape: [batch_size, P(N,2), N]

            correct += (outputs_argmax == labels).all(dim=2).sum().item()   ################################# dim の値変えたらなんかそれっぽくなったぞ！
    model.train()
    print(f'> デバック    == Path    ==    Path Correct: {correct}          Path Accuracy: {correct/(NP_2*len(loader.dataset)) *100}    データセット数: {len(loader.dataset)} ({NP_2*len(loader.dataset)})')
    return correct / (NP_2*(batch_size*len(loader))) *100

# 要素ごとの評価
def element_accuracy(loader, model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # shape: [128, 30, 42] <-- shape: [batch_size, P(N,2), ((N+1)*N)]
            outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))    # reshape(batch_size, P(N,2), N, (N+1)) <--
            outputs_softmax = F.softmax(outputs_reshaped, dim=3)
            outputs_argmax = torch.argmax(outputs_softmax, 3)  # shape: [128, 30, 6] <-- shape: [batch_size, P(N,2), N]
            
            correct += (outputs_argmax == labels).sum().item()
    model.train()
    print(f'> デバック    == Element ==    Element Correct: {correct}    Element Accuracy: {correct/(N*NP_2*len(loader.dataset)) *100}    データセット数: {len(loader.dataset)} ({N*NP_2*len(loader.dataset)})')

    return correct / (N*NP_2*(batch_size*len(loader))) *100



### LR Range Test ###
def lr_range_test(model, loader, criterion, device):
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (10 ** (1/5)) ** step)

    model.train()
    losses = []
    lrs = []

    for step, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))
        outputs_permuted = outputs_reshaped.permute(0, 3, 1, 2)

        loss = criterion(outputs_permuted, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])

        check_lr = optimizer.param_groups[0]['lr']
        if 1e-2 <= check_lr:
            break
        
        scheduler.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('LR Range Test')
    plt.grid(True)
    plt.show()



### main ###
# 損失関数
criterion = nn.CrossEntropyLoss()


## LR Range Test
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = PartiallyConnectedLayerNet().to(device)
lr_range_test(model, train_loader, criterion, device)

# エポック数
max_epoch = 1000

# モデルのインスタンス化
model = PartiallyConnectedLayerNet()

# 初期学習率
initial_lr = float(input("初期学習率: "))

# 最適化アルゴリズム: Adamax
optimizer = torch.optim.Adamax(model.parameters(), lr=initial_lr)
"""
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
"""

early_stopping = EarlyStopping(patience, verbose=True)

# ログ
lr_history = []
epoch_log = 0
loss_train_log = 0
loss_val_log = 0
train_log, val_log = {}, {}
train_log['loss'], train_log['demand_accuracy'], train_log['path_accuracy'], train_log['element_accuracy'] = [], [], [], []
val_log['loss'], val_log['demand_accuracy'], val_log['path_accuracy'], val_log['element_accuracy'] = [], [], [], []


## 学習ループ
start = time.time()
print(">>> Beginning Model Training")
for epoch in range(max_epoch):
    print(f'>> epoch: {epoch}/{max_epoch}')

    # train
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)  # shape: [batch_size, P(N,2), ((N+1)*N)]
        outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))    # reshape(batch_size, P(N,2), N, (N+1)) <--
        outputs_permuted = outputs_reshaped.permute(0, 3, 1, 2) # shape: [128, 7, 180] <-- shape: [batch_size, N+1, N*P(N,2)]

        loss_train = criterion(outputs_permuted, labels)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        loss_train_log = loss_train

    # validate
    model.eval()
    with torch.no_grad():
        for each_batch in val_loader:
            inputs, labels = each_batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)    # shape: [128, 30, 42] <-- shape: [batch_size, P(N,2), ((N+1)*N)]
            outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))    # shape: [batch_size, P(N,2), N, (N+1)]
            outputs_permuted = outputs_reshaped.permute(0, 3, 1, 2) # shape: [64, 5, 48] <-- shape: [batch_size, (N+1), N*P(N,2)]

            loss_val = criterion(outputs_permuted, labels)

            loss_val_log = loss_val


    ## ログ保存
    # 学習率
    lr = optimizer.param_groups[0]['lr']
    lr_history.append(lr)
    # train
    print(">>> train")
    train_log['loss'].append(loss_train.item())
    train_demand_acc = demand_accuracy(train_loader, model, device)
    train_log['demand_accuracy'].append(train_demand_acc)
    train_path_acc = path_accuracy(train_loader, model, device)
    train_log['path_accuracy'].append(train_path_acc)
    train_elem_acc = element_accuracy(train_loader, model, device)
    train_log['element_accuracy'].append(train_elem_acc)
    
    # val
    print(">>> val")
    val_log['loss'].append(loss_val.item())
    val_demand_acc = demand_accuracy(val_loader, model, device)
    val_log['demand_accuracy'].append(val_demand_acc)
    val_path_acc = path_accuracy(val_loader, model, device)
    val_log['path_accuracy'].append(val_path_acc)
    val_elem_acc = element_accuracy(val_loader, model, device)
    val_log['element_accuracy'].append(val_elem_acc)


    ## Early Stopping
    early_stopping(loss_val, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

    """
    # 学習率を更新
    scheduler.step()
    """

    epoch_log += 1
    print("\n")

print(">>> Model Training Finished")
end = time.time()
print()



### test ###
print(">>> Trained Model")
print(f'time: {int((end-start) // 60)}m{(end-start) % 60:.2f}s ({end - start:.4f}s)')

print("=== Imformation ===")
print(f'Batch Size: {batch_size}')
print(f'Hidden Layer: {num_hidden_layer}')
print(f'Max Epoch: {max_epoch}')
print(f'Final Epoch: {epoch_log}')
print(f'Train Loss: {loss_train_log}')
print(f'Val Loss: {loss_val_log}')
print(f'Initial Learning Rate: {initial_lr}')
for param_group in optimizer.param_groups:
    print(f'lr: {param_group['lr']}')

print("=== Final Results ===")
test_demand_acc = demand_accuracy(test_loader, model, device)
print(f'Test Demand Accuracy: {test_demand_acc:.4f}')

test_path_acc = path_accuracy(test_loader, model, device)
print(f"Test Path Accuracy: {test_path_acc:.4f}")

test_elem_acc = element_accuracy(test_loader, model, device)
print(f'Test Element Accuracy: {test_elem_acc:.4f}')



# モデルを保存
torch.save(model.state_dict(), "best_model.pth")
print("モデルを保存: best_model.pth")


## モデルの出力を保存 ##
model.eval()
outputs_list, labels_list = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))
        outputs_softmax = F.softmax(outputs_reshaped, dim=3)
        outputs_argmax = torch.argmax(outputs_softmax, dim=3)

        outputs_list.append(outputs_argmax.cpu())
        labels_list.append(labels.cpu())

outputs_tensor = torch.cat(outputs_list, dim=0)
labels_tensor = torch.cat(labels_list, dim=0)

outputs_np = outputs_tensor.numpy()
labels_np = labels_tensor.numpy()

outputs_flat = outputs_np.reshape(-1, outputs_np.shape[2])
labels_flat = labels_np.reshape(-1, labels_np.shape[2])

with open('outputs.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    for row in outputs_flat:
        writer.writerow(row)

with open('labels.csv', 'w', newline='') as f_label:
    writer = csv.writer(f_label)
    for row in labels_flat:
        writer.writerow(row)



### 学習曲線の可視化 ###
plt.figure(figsize=(18, 8))

plt.subplot(1, 4, 1)
plt.plot(train_log['loss'], label='Train Loss')
plt.plot(val_log['loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(train_log['demand_accuracy'], label='Train Demand Accuracy')
plt.plot(val_log['demand_accuracy'], label='Val Demand Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Demand Accuracy')
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(train_log['path_accuracy'], label='Train Path Accuracy')
plt.plot(val_log['path_accuracy'], label='Val Path Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Path Accuracy')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(train_log['element_accuracy'], label='Train Element Accuracy')
plt.plot(val_log['element_accuracy'], label='Val Element Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Element Accuracy')
plt.legend()

plt.suptitle(f'H={num_hidden_layer}_BS={batch_size}_Initial lr={initial_lr}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 10))
plt.plot(lr_history, label='Learning Rate', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.title('Learning Rate Scheduling')

plt.tight_layout()
plt.show()






















""""

### 最適化 ###
def objective(trial):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Optuna が探索するハイパーパラメータ
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, step=1e-5)

    # モデルのインスタンス化
    model = PartiallyConnectedLayerNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience, verbose=True)


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

            outputs = model(inputs)    # shape: [128, 30, 42] <-- shape: [batch_size, P(N,2), ((N+1)*N)]
            outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))    # reshape(batch_size, P(N,2), N, (N+1)) <--
            outputs_reshaped = outputs_reshaped.reshape(batch_size, (N*NP_2), (N+1))  # reshape(batch_size, N*P(N,2), (N+1)) <--
            outputs_reshaped = outputs_reshaped.permute(0, 2, 1) # shape: [128, 7, 180] <-- shape: [batch_size, N+1, N*P(N,2)]

            labels_reshaped = labels.reshape(batch_size, (N*NP_2))  # shape: [batch_size, N*P(N,2)]

            loss_train = criterion(outputs_reshaped, labels_reshaped)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for each_batch in val_loader:
                inputs, labels = each_batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)    # shape: [128, 30, 42] <-- shape: [batch_size, P(N,2), ((N+1)*N)]
                outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))    # shape: [batch_size, P(N,2), N, (N+1)]
                outputs_reshaped = outputs_reshaped.reshape(batch_size, (N*NP_2), (N+1)) # shape: [batch_size, N*P(N,2), (N+1)]
                outputs_reshaped = outputs_reshaped.permute(0, 2, 1) # shape: [64, 5, 48] <-- shape: [batch_size, (N+1), N*P(N,2)]

                labels_reshaped = labels.reshape(batch_size, (N*NP_2))    # shape: [batch_size, N*P(N,2)]

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
study.optimize(objective, n_trials)  # n_trials: 試行回数

best_trial = study.best_trial
print('best_trial: \n', best_trial)
print('=== ハイパーパラメータ ===\n', best_trial.params)
for key, value in best_trial.params.items():
    print(f'{key}: {value}')
print(f'Best Validation Path Accuracy: {best_trial.value:.4f}')



### 最良パラメータによるモデルの再学習 ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = PartiallyConnectedLayerNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=best_trial.params["learning_rate"])
early_stopping = EarlyStopping(patience, verbose=True)

# train + val をまとめて再学習用に使う
full_train = torch.utils.data.ConcatDataset([train, val])
full_loader = torch.utils.data.DataLoader(full_train, batch_size=batch_size, shuffle=True, drop_last=True)



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

        outputs = model(inputs)   # shape: [128, 30, 42] <-- shape: [batch_size, P(N,2), ((N+1)*N)]
        outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))    # shape: [batch_size, P(N,2), N, (N+1)]
        outputs_reshaped = outputs_reshaped.reshape(batch_size, (N*NP_2), (N+1))  # shape: [batch_size, N*P(N,2), (N+1)]
        outputs_reshaped = outputs_reshaped.permute(0, 2, 1)    # shape: [batch_size, (N+1), N*P(N,2)]

        labels_reshaped = labels.reshape(batch_size, (N*NP_2))    # shape: [batch_size, N*P(N,2)]

        loss_train = criterion(outputs_reshaped, labels_reshaped)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    # テスト用データによる評価
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)   # shape: [64, 12, 20] <-- shape: [batch_size, P(N,2), ((N+1)*N)]
            outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))    # shape: [batch_size, P(N,2), N, (N+1)]
            outputs_reshaped = outputs_reshaped.reshape(batch_size, (N*NP_2), (N+1))  # shape: [batch_size, N*P(N,2), (N+1)]
            outputs_reshaped = outputs_reshaped.permute(0, 2, 1)    # shape: [batch_size, (N+1), N*P(N,2)]

            labels_reshaped = labels.reshape(batch_size, (N*NP_2))    # shape: [batch_size, N*P(N,2)]

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

outputs_reshaped = outputs.reshape(batch_size, (N*NP_2), (N+1))

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

"""