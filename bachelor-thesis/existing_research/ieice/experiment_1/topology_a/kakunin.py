import heapq
import random

def dijkstra(graph, start, end):
    # debug #
    print(f'＜src={start}, dst={end}の探索＞')

    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    queue = [(0, 0, 0, start, [start])]

    finalized = set()
    # finalized_shortest_path = {node: [start] for node in graph}
    
    while queue:
        current_distance, _, _, current_node, current_path = heapq.heappop(queue) # 未確定ノードから最小コストを抽出

        # debug #
        print(f'distances={distances}')
        print(f'current_node={current_node}, current_distance=☆ {current_distance}, current_path={current_path} <-- 探索開始')
        
        if current_node == end:
            # debug #
            print(f'current_node={current_node}, current_distance={current_distance} <-- 探索終了？\n')
            return current_path

        if distances[current_node] < current_distance:
            # debut #
            print(f'distances[current_node]={distances[current_node]} < current_distance={current_distance} <-- continue\n')
            continue
        
        finalized.add(current_node)

        # for neighbor, weight in graph[current_node].items():
        for step, (neighbor, weight) in enumerate(random.sample(list(graph[current_node].items()), len(graph[current_node]))):
            if neighbor not in finalized:
                # debug #
                print(f'neighbor={neighbor}, weight={weight} <-- 探索開始')
                
                distance = current_distance + weight

                if distance <= distances[neighbor]:
                    # debug #
                    print(f'distance={distance} < distances[neighbor]={distances[neighbor]} --> distances[neighbor]={distances[neighbor]} を distance={distance} に更新\n')

                    distances[neighbor] = distance
                    path = current_path +[neighbor]
                    print(len(path))
                    heapq.heappush(queue, (distance, len(path)-1, step, neighbor, path))
            

            
        
        # debug #
        print("\n")
                
    return None

# グラフの定義
graph = {
    'a': {'b': 1, 'c': 7, 'd': 1},
    'b': {'a': 1, 'e': 2, 'f': 4},
    'c': {'a': 7, 'f': 2, 'g': 3},
    'd': {'a': 1, 'g': 5},
    'e': {'b': 2, 'f': 3},
    'f': {'b': 4, 'c': 2, 'e': 3, 'h': 2},
    'g': {'c': 3, 'd': 5, 'h': 1},
    'h': {'f': 2, 'g': 1}

}

print(dijkstra(graph, 'a', 'h'))
