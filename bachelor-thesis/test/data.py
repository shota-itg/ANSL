# data.py

import csv
import torch
import numpy as np

### データの取得 ###
print("### データの取得 ###")
print()
x = []
t = []

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

before_label, label, label_argmax = [], [], []
before_label = np.array(t)
for i in range(10000):
    label.append(before_label[i].reshape(-1, 5))

print(before_label)
label_argmax = np.argmax(label, 2)

x = torch.tensor(x, dtype=torch.float32)
label = torch.tensor(label_argmax, dtype=torch.int64)

print(label)

"""
t_label = []
label = []

t_label = np.array(t)
#print(f't_label: \n{t_label}')
print(len(t_label))

for i in range(10000):
    label.append(t_label[i].reshape(-1, 5))

#print(f'label: \n{label}')

label_argmax = np.argmax(label, 2)

print(f'label_argmax: \n{label_argmax}')

# PyTorch の Tensor 型へ変換
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

print(t)
"""