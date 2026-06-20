import numpy as np
import matplotlib.pyplot as plt

# 平均λ
lamb = 4

# 平均λ=4のポワソン分布に従うランダムな値を1000件生成
s = np.random.poisson(lamb, 5000)

#ヒストグラムで表示
plt.hist(s)
plt.show()