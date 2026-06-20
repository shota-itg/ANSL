# utils /metrics.py

import math
import torch
import torch.nn.functional as F

from utils.config_loader import load_config


## 評価関数
# トラフィックデマンド集合単位の評価
def demand_accuracy(loader, batch_size, model, device, exp_dir=None):
    config = load_config(exp_dir)
    nodes = config["topology"]["nodes"]

    N = len(nodes)
    NP_2 = math.perm(N, 2)

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data) # shape[batch_size *num_traffic *(N+1) *N]
            outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))
            outputs_softmax = F.softmax(outputs_reshaped, dim=3)
            outputs_argmax = torch.argmax(outputs_softmax, 3)   # shape: [batch_size, num_traffic, N]

            # labels = labels.reshape(batch_size, num_traffic, N)

            correct += ((outputs_argmax == labels).all(dim=2)).all(dim=1).sum().item()  # (outputs_argmax == labels): shape[batch_size, NP_2, N]の bool テンソル(True /False)   # .all(dim=2): shape[batch_size, NP_2]  # .all(dim=1): shape[batch_size]    # .sum(): Trueの数を数える  # .item(): 整数に変換
        model.train()
        # debug # print(f'> デバック    == Demand  ==    Demand Correct: {correct}       Demand Accuracy: {correct /(len(loader.dataset)) *100}    データセット数: {len(loader.dataset)} ({len(loader.dataset)})')
        return correct /(batch_size *len(loader)) *100


# 経路単位の評価
    # 評価項目(1a)
def path_accuracy(loader, batch_size, model, device, exp_dir=None):
    config = load_config(exp_dir)
    nodes = config["topology"]["nodes"]

    N = len(nodes)
    NP_2 = math.perm(N, 2)

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data) # shape[batch_size *num_traffic *(N+1) *N]
            outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))
            outputs_softmax = F.softmax(outputs_reshaped, dim=3)
            outputs_argmax = torch.argmax(outputs_softmax, 3) # shape[batch_size, num_traffic, N]

            # labels = labels.reshape(batch_size, num_traffic, N)

            correct += (outputs_argmax == labels).all(dim=2).sum().item()
    model.train()
    # debug # print(f'> デバック    == Path    ==    Path Correct: {correct}          Path Accuracy: {correct /(num_traffic *len(loader.dataset)) *100}    データセット数: {len(loader.dataset)} ({num_traffic *len(loader.dataset)})')
    return correct /(NP_2 *(batch_size *len(loader))) *100


# 要素ごとの評価
def element_accuracy(loader, batch_size, model, device, exp_dir=None):
    config = load_config(exp_dir)
    nodes = config["topology"]["nodes"]

    N = len(nodes)
    NP_2 = math.perm(N, 2)
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data) # shape[batch_size *num_traffic *(N+1) *N]
            outputs_reshaped = outputs.reshape(batch_size, NP_2, N, (N+1))
            outputs_softmax = F.softmax(outputs_reshaped, dim=3)
            outputs_argmax = torch.argmax(outputs_softmax, 3)  # shape[batch_size, (NP_2 /2), N]

            # labels = labels.reshape(batch_size, num_traffic, N)
            
            correct += (outputs_argmax == labels).sum().item()
    model.train()
    # debug # print(f'> デバック    == Element ==    Element Correct: {correct}    Element Accuracy: {correct /(N *num_traffic *len(loader.dataset)) *100}    データセット数: {len(loader.dataset)} ({N *num_traffic *len(loader.dataset)})')

    return correct /(N *NP_2 *(batch_size *len(loader))) *100