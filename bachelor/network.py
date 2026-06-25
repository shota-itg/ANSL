import networkx as nx # NetworkXをインポート
import matplotlib.pyplot as plt

# ネットワークの生成
G = nx.Graph() # 空のグラフ
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]) # 空のグラフにエッジを追加

# 描画
nx.draw(G, with_labels=True)
plt.show()