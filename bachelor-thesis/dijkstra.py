"""
2025/08/25
ダイクストラ法
"""

def dijkstra(edges, num_node):
    """ 経路の表現
            [終点, 辺の値]
            A, B, C, D, ... → 0, 1, 2, ...とする"""
    node = [float('inf')] * num_node    # スタート地点以外の値は ∞ で初期化
    node[0] = 0 # スタートは 0 で初期化

    node_name = [i for i in range(num_node)]    # ノードの名前を 0~ ノードの数で表す

    while 0 < len(node_name):
        r = node_name[0]

        # 最もコストが小さい頂点を探す
        for i in node_name:
            if node[i] < node[r]:
                r = i   # コストが小さい頂点が見つかると更新

        # 最もコストが小さい頂点を取り出す
        min_point = node_name.pop(node_name.index(r))

        # 経路の要素を各変数に格納することで，視覚的に見やすくする
        for factor in edges[min_point]:
            goal = factor[0]    # 終点
            cost = factor[1]    # コスト

            # 更新条件
            if node[min_point] + cost < node[goal]:
                node[goal] = node[min_point] + cost # 更新

    return node

if __name__ == '__main__':
    Edges = [
        [[1, 4], [2, 3]],   # 頂点0 からの辺のリスト
        [[2, 1] ,[3, 1], [4, 5]],   # 頂点1 からの辺のリスト
        [[5, 2]],   # 頂点2 からの辺のリスト
        [[4, 3]],   # 頂点3 からの辺のリスト
        [[6, 2]],   # 頂点4 からの辺のリスト
        [[4, 1], [6, 4]],   # 頂点5 からの辺のリスト
        []  # 頂点6からの辺のリスト
    ]

    # 今の目的地の数は 7つ(0~6)
    node_num = 7

    opt_node = dijkstra(Edges, node_num)

    # 以下は結果を整理するためのコード
    node_name = []
    for i in range(node_num):
        node_name.append(chr(ord('A') + i))
    result = []
    for i in range(len(opt_node)):
        result.append(f"{node_name[i]} : {opt_node[i]}")
    print(f"'目的地: そこまでの最小コスト'\n\n{result}")