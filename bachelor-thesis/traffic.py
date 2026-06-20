import time
import random
import csv
from multiprocessing import Process, Queue

# 帯域幅の選択肢[Mbps]
bandwidth_options = [150, 300, 450, 600, 750, 900, 1000]

# 任意のノードが他ノードに対してトラフィックを発生させる関数
# 発生する確率は 40%
def generate_traffic(node_id, other_nodes, queue, prob=0.4):
    for dst in other_nodes:
        if random.random() < prob:
            bw = random.choice(bandwidth_options)
            queue.put([node_id, dst, bw])
    return graph


# ダイクストラ法による最短経路探索
def dijkstra(graph, start, end):
    queue = [(0, start, [start])]
    visited = set()

    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == end:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node].items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
    return None


if __name__ == "__main__":
    nodes = [0, 1, 2, 3]
    queue = Queue()
    traffic_log = []

    duration = 10 # 実行時間[s]

    start_time = time.time()

    while time.time() - start_time < duration:
        # 並列処理
        processes = []
        for src in nodes:
            dsts = [n for n in nodes if n != src]
            p = Process(target=generate_traffic, args=(src, dsts, queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        
        while not queue.empty():
            traffic_log.append(queue.get())

        time.sleep(0.5) # 少し待機（負荷調整）

    # 通信ログの表示
    print("Traffic log: ")
    for entry in traffic_log:
        print(entry)
    print("収集完了\n件数: ", len(traffic_log))


# CSVファイルへの保存
with open("traffic_log.cs", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["src", "dst", "bandwidth", "path"]) # ヘッダー

    writer.writerows(traffic_log)