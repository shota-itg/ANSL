# kakunin.py

import torch

a = [[[0, 1], [1, 0], [1, 0]], [[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [0, 1]], [[0, 1], [0, 1], [1, 0]]]    # shape: [4, 3, 2]
b = [[[0, 1], [1, 0], [1, 0]], [[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [0, 1]], [[0, 1], [0, 0], [1, 0]]]    # shape: [4, 3, 2]


a = torch.tensor(a, dtype=torch.float32)
b = torch.tensor(b, dtype=torch.float32)
c = torch.sigmoid(b)
pred = (0.5 < torch.sigmoid(b)).int()

correct = (a == b).all(dim=2).sum().item()
print(correct)

print(correct / (3*4))