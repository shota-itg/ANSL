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

with open("traffic_log.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data_len = len([h for h in header if h.startswith("data_")])
    target_len = len([h for h in header if h.startswith("target_")])

    for row in reader:
        data = list(map(int, row[:data_len]))
        target = list(map(int, row[data_len:data_len + target_len]))
        x.append(data)
        t.append(target)

traffic_log = Bunch(data=np.array(x), target=np.array(t))
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

### DataLoader の定義 ###
dataset = torch.utils.data.TensorDataset(x, t)
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val
torch.manual_seed(0)
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

batch_size = 128
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)

### ネットワークの定義 ###
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(60, 720)
        self.fc2 = nn.Linear(720, 240)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return h

### 評価関数 ###
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

### メイン処理 ###
def main(mode='train'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    net = Net().to(device)

    if mode == 'train':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        max_epoch = 50
        train_losses = []
        val_accuracies = []

        for epoch in range(max_epoch):
            for x, t in train_loader:
                x, t = x.to(device), t.to(device)
                y = net(x)
                loss = F.binary_cross_entropy_with_logits(y, t.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())
            val_acc = evaluate(val_loader, net, device)
            val_accuracies.append(val_acc)
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}')

        torch.save(net.state_dict(), 'model.pth')
        print("✅ モデルを 'model.pth' に保存しました。")

        # 学習曲線の可視化
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
            print("❌ 学習済みモデル 'model.pth' が見つかりません。先に学習を行ってください。")
            return

        net.load_state_dict(torch.load('model.pth'))    # 保存したパラメータを読み込む
        net.eval()  # 評価モードに切り替え
        test_acc = evaluate(test_loader, net, device)
        print(f"🧪 テスト精度: {test_acc:.4f}")

if __name__ == '__main__':
    main(mode='train')  # 'train' または 'eval' に切り替え可能