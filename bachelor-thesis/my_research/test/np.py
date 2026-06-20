import numpy as np

arr = np.array([[1, 100], [3, 450], [2, 150]])

idx = np.argmax(arr[:, 1])
row = arr[idx]

print(row)