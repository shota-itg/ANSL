import numpy as np
import matplotlib.pyplot as plt

tau = 180

s = np.random.exponential(tau, 5000)

#ヒストグラムで表示
plt.hist(s)
plt.show()