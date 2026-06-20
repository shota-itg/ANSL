# data_gen/dijkstra.py

import heapq
import random

from utils.config_loader import load_config

config = load_config()
nodes = config["topology"]["nodes"]


## ダイクストラ法で経路探索
def dijkstra(graph, start, end):
    # debug #print(f'＜src={start}, dst={end}の探索＞')

    distances = {node: float('infinity') for node in nodes}
    distances[start] = 0
    queue = [(0, 0, 0, start, [start])]
    finalized = set()

    while queue:
        current_distance, _, _, current_node, current_path = heapq.heappop(queue)

        # debug #print(f'distances={distances}\ncurrent_node={current_node}, current_distance=☆ {current_distance}, current_path={current_path} <-- 探索開始')

        if current_node == end:
            # debug #print(f'current_node={current_node}, current_distance={current_distance} <-- 探索終了\n')

            return current_path
        
        if distances[current_node] < current_distance:
            # debut #print(f'distances[current_node]={distances[current_node]} < current_distance={current_distance} <-- continue\n')

            continue

        finalized.add(current_node)

        for step, (neighbor, weight) in enumerate(random.sample(list(graph.get(current_node, {}).items()), len(graph.get(current_node, {})))):    # neighbor: 隣接ノード  # graph.get(node, {}): 「 node の隣接ノード一覧」を取得 # .items(): 「隣接ノードと重み」のペアを取り出す
            if neighbor not in finalized:
                # debug #print(f'neighbor={neighbor}, weight={weight} <-- 探索開始')

                distance = current_distance + weight

                if distance <= distances[neighbor]:
                    # debug #print(f'distance={distance} < distances[neighbor]={distances[neighbor]} --> distances[neighbor]={distances[neighbor]} を distance={distance} に更新\n')

                    distances[neighbor] = distance
                    path = current_path +[neighbor]
                    heapq.heappush(queue, (distance, len(path)-1, step, neighbor, path))

    return None


## 経路上のリンクを取得
    # 経路に含まれるリンクのリストを取得
    # 経路 path=[0,1,3] なら，リンク (0,1) と (1,3) を取得
def get_links_from_path(path, links):
    link_list = []  # 空のリスト (list) # 経路に含まれるリンクを順番に追加
    """
    if path:
        for i in range(len(path) - 1):
            link = links.get((path[i], path[i+1]))
            if link:
                link_list.append(link)

        return link_list
    else:
        return None
    
    if not path:
        return None    
    """
    
    for i in range(len(path) -1):
        u = path[i]
        v = path[i +1]
        key = (min(u, v), max(u, v))
        link = links.get(key)
        if link:
            link_list.append(link)

    return link_list