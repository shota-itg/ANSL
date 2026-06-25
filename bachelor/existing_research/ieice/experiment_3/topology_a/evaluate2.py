# evaluate.py

"""
既存研究の評価項目(1b)のシミュレーション
"""


import csv
import numpy as np
from collections import deque
import heapq
import threading
import math
import statistics
import random
import time

# データ量の選択肢 [MB]
data_size_options = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

# データ量の選択肢 [MB]
data_size_options = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

path_djk_2 = 0
path_djk_3 = 0
path_djk_4 = 0
path_djk_5 = 0
path_djk_6 = 0



### データの取得 ###
data = []
queue_djk = deque()
queue_ex = deque()

with open("io.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        int_row = [int(float(cell)) for cell in row]    # 実数を整数に変換（切り捨て）
        data.append(int_row)

print(f'len(data): {len(data)}')
for i in range(len(data)):
    queue_djk.append(data[i])
    queue_ex.append(data[i])

"""
data_np = np.array(data)
data_np = data_np.reshape(-1, 30, 9)
print(f'data_np.shape: {data_np.shape}')
for i in range(len(data_np)):
    queue.append(data)
"""





### リンククラス：容量と使用量を管理 ###
class Link:

    ### __init__はクラスのコンストラクタ（初期化関数） ###
        # クラスからオブジェクトを作ったときに最初に呼ばれる関数
    def __init__(self, max_capacity):   # self: クラスの自分自身（インスタンス）を指す変数
        self.max_capacity = max_capacity    # max_capacity: 最大帯域（例: 1000Mbps）
        self.used = 0   # used: 現在使用中の帯域
        self.lock = threading.Lock()    # lock: スレッド安全性を確保するためのロック（複数スレッドが同時に帯域を変更しないように）

    ## 要求帯域 bw を確保できるかをチェック
    def check_links(self, bw):
        with self.lock:
            return self.used + bw <= self.max_capacity
        
    ## 帯域の確保
        # 要求帯域 bw を確保できるか確認し，可能なら used を増やす
    def try_allocate(self, bw):
        with self.lock:
            if self.used + bw <= self.max_capacity:
                self.used += bw

    ## 帯域の解放
        # 使用後に used を減らす
    def release(self, bw):
        with self.lock:
            self.used -= bw

    ## debug
    def status(self):
        return f'used: {self.used} / {self.max_capacity} Mbps'
    


### ネットワークを構築 ###
    # ノード間のリンク構成（6ノード・8リンク）
def build_network():
    links = {}  # 空の辞書 (dictionary)
    # ノード間リンク（双方向）
    links[(0,1)] = Link(10000)   # (0,1) は辞書 (dictionary) のキー（タプル）    # Link(1000) は Link クラスのインスタンス
    links[(1,0)] = links[(0,1)]
    links[(0,5)] = Link(10000)
    links[(5,0)] = links[(0,5)]
    links[(1,2)] = Link(10000)
    links[(2,1)] = links[(1,2)]
    links[(1,5)] = Link(10000)
    links[(5,1)] = links[(1,5)]
    links[(2,3)] = Link(10000)
    links[(3,2)] = links[(2,3)]
    links[(2,4)] = Link(10000)
    links[(4,2)] = links[(2,4)]
    links[(3,4)] = Link(10000)
    links[(4,3)] = links[(3,4)]
    links[(4,5)] = Link(10000)
    links[(5,4)] = links[(4,5)]
    
    return links



# グラフ構造（重みは帯域幅の逆数）
# リンク使用状況に応じた重み付きグラフを生成
    # 重みは「空き帯域の逆数」なので，空いているほど経路として優先される
def generate_graph(links):
    graph = {}  # 空の辞書 (dictionary)
    for (src, dst), link in links.items():  # links.items() は辞書 (dictionary) のすべてのキーと値のペアを取り出す  # 例）(0,1): Link(1000) → src=0, dst=1, link = Link(1000)
        ### print(f"Link ({src}->{dst}): used={link.used}, max_capacity={link.max_capacity}")   ### デバック用
        if src not in graph:    # もし src が graph にまだ登録されていなければ
            graph[src] = {} # 初めて見るノードなら，隣接ノードの情報を入れるための空の辞書 (dictionary) を作る準備
        if link.used == link.max_capacity:
            graph[src][dst] = float('inf')
        else:
            graph[src][dst] = 1 / (link.max_capacity - link.used)    # 使用量に応じて重み変化    # 二次元辞書（辞書の中に辞書）  # 1e-6: ゼロ除算を防ぐための微小値
    
    return graph



### ダイクストラ法で経路探索 ###
def dijkstra(graph, start, end):

    # debug #print(f'＜src={start}, dst={end}の探索＞')

    queue = deque()
    queue.append((0, start, [start]))  # queue: 優先度付きキュー (heapq) でコストの小さい順に探索  # [start]: 経路を記録するためのリストで，最初は「自分自身だけの経路」なので [start]
    shortest = []

    while queue:
        cost, node, path = queue.popleft() # path: 現在の経路  # heapq: 優先度付きキューを扱うモジュール   # heappop: コストが最小の要素を取り出す関数 # queueの中身は（コスト, 現在のノード, 経路）のタプル

        # debug #print(f'cost={cost}, node={node}, path={path} <-- 探索開始')

        if node == end:

            # debug #print(f'cost={cost}, node={node}, path={path} <-- こいつは経路の候補だぞ！\n')
            
            heapq.heappush(shortest, (cost, len(path), node, path))
            continue

        for neighbor, bandwidth in graph.get(node, {}).items():    # neighbor: 隣接ノード  # graph.get(node, {}): 「 node の隣接ノード一覧」を取得 # .items(): 「隣接ノードと重み」のペアを取り出す
            if neighbor not in path:
                queue.append((cost+bandwidth, neighbor, path+[neighbor])) # .heappush: 新しい経路候補をキューに追加   # path + [neighbor]: 今の経路に neighbor を追加した新しい経路

                # debug #print(f'cost={cost+bandwidth}, node={neighbor}, path={path+[neighbor]} <-- 探索結果を追加')

        # debug #print()

    if shortest:
        cost, _, node, path = heapq.heappop(shortest)
        if len(path) == 2:
            global path_djk_2
            path_djk_2 += 1
        elif len(path) == 3:
            global path_djk_3
            path_djk_3 += 1
        elif len(path) == 4:
            global path_djk_4
            path_djk_4 += 1
        elif len(path) == 5:
            global path_djk_5
            path_djk_5 += 1
        elif len(path) == 6:
            global path_djk_6
            path_djk_6 += 1
        return path
    else:
        return None



### 経路上のリンクを取得 ###
    # 経路に含まれるリンクのリストを取得
    # 経路 path=[0,1,3] なら，リンク (0,1) と (1,3) を取得
def get_links_from_path(path, links):
    link_list = []  # 空のリスト (list) # 経路に含まれるリンクを順番に追加
    for i in range(len(path) - 1):
        link = links.get((path[i], path[i+1]))
        if link:
            link_list.append(link)

    return link_list



### メインの処理 ###
if __name__ == "__main__":
    nodes = [0, 1, 2, 3, 4, 5]
    NP_2 = math.perm(len(nodes), 2)

    path_ex_2 = 0
    path_ex_3 = 0
    path_ex_4 = 0
    path_ex_5 = 0
    path_ex_6 = 0


    ## Dijkstra 法
    links = build_network() # links はディクショナリ Link クラス

    # counter
    evaluation_1a = 0
    no_path_counter_djk = 0
    no_path_all_counter_djk = 0
    fukusou_counter_djk = 0
    fukusou_all_counter_djk = 0

    max_load_link_djk = 0

    # debug #
    print(">>> Dijkstra")

    while queue_djk:
        release_threads = []

        # counter
        no_path_sub_counter_djk = 0
        fukusou_sub_counter_djk = 0
        
        all_used_links_djk = 0
        links_djk_list = []

        for _ in range(NP_2):
            graph = generate_graph(links)
            item = queue_djk.popleft()
            src, dst, bw = item[:3]
            path = dijkstra(graph, src, dst)
            data_size = (random.choice(data_size_options) *8)

            # debug #print(f'src={src}, dst={dst}, bw={bw}, path={path}')

            if not path:
                no_path_sub_counter_djk +=1
                continue

            link_path = get_links_from_path(path, links)
            success = all(link.check_links(bw) for link in link_path)

            if success:
                for link in link_path:
                    link.try_allocate(bw)
                evaluation_1a += 1

                def release_later(each_link, bw, data_size):
                    delay = data_size / bw
                    time.sleep(delay *3e-4) ########################################################### ここで調整
                    for l in each_link:
                        l.release(bw)

                # threading.Thread(target=release_later, args=(link_path, bw, data_size)).start()
                t = threading.Thread(target=release_later, args=(link_path, bw, data_size))
                t.start()
                release_threads.append(t)
            else:
                fukusou_sub_counter += 1

        if not (no_path_sub_counter_djk == 0 and fukusou_sub_counter_djk == 0):
            no_path_counter_djk += 1
            no_path_all_counter_djk += no_path_sub_counter_djk
            fukusou_counter_djk += 1
            fukusou_all_counter_djk += fukusou_sub_counter_djk


        ## 最大負荷リンクにおける使用帯域幅の詳細値 [Mbps]
        for link in links.values():
            all_used_links_djk += link.used
            links_djk_list += [link.used]

        max_djk_log = max(links_djk_list)
        min_djk_log = min(links_djk_list)
        average_djk_log = statistics.mean(links_djk_list)
        std_dev_djk_log = statistics.stdev(links_djk_list)
        std_err_djk_log = std_dev_djk_log / (len(links_djk_list) ** 0.5)
        median_djk_log = statistics.median(links_djk_list)

        if max_load_link_djk < all_used_links_djk:
            max_load_link_djk = all_used_links_djk

            max_djk = max_djk_log
            min_djk = min_djk_log
            average_djk = average_djk_log
            std_dev_djk = std_dev_djk_log
            std_err_djk = std_err_djk_log
            median_djk = median_djk_log
            print(links_djk_list)

        for t in release_threads:
            t.join()

        # リンクの一斉解放
        for link in links.values():
            link.used = 0



    ### Existing Proposal
    links = build_network() # links はディクショナリで Link クラス

    # counter
    evaluation_1b = 0
    no_path_counter_ex = 0
    no_path_all_counter_ex = 0
    fukusou_counter_ex = 0
    fukusou_all_counter_ex = 0

    max_load_link_ex = 0

    # debug #
    print(">>> Proposal of Existring Research")

    while queue_ex:
        release_threads = []

        # counter
        no_path_sub_counter_ex = 0
        fukusou_sub_counter_ex = 0

        all_used_links_ex = 0
        links_ex_list = []

        for _ in range(NP_2):
            graph = generate_graph(links)
            item = queue_ex.popleft()
            src, dst, bw = item[:3]
            path = item[3:]
            data_size = (random.choice(data_size_options) *8)

            path_len_check = 0
            for i in range(len(path)):
                if path[i] == 6:
                    path_len_check = i
                    break

            if path_len_check == 2:
                path_ex_2 += 1
            elif path_len_check == 3:
                path_ex_3 += 1
            elif path_len_check == 4:
                path_ex_4 += 1
            elif path_len_check == 5:
                path_ex_5 += 1
            elif path_len_check == 6:
                path_ex_6 += 1

            # debug #print(f'src={src}, dst={dst}, bw={bw}, path={path}')

            if not path:
                no_path_sub_counter_ex +=1
                continue

            link_path = get_links_from_path(path, links)
            success = all(link.check_links(bw) for link in link_path)

            if success:
                for link in link_path:
                    link.try_allocate(bw)
                evaluation_1b += 1

                def release_later(each_link, bw, data_size):
                    delay = data_size / bw
                    time.sleep(delay *3e-4) ########################################################### ここで調整
                    for l in each_link:
                        l.release(bw)

                # threading.Thread(target=release_later, args=(link_path, bw, data_size)).start()
                t = threading.Thread(target=release_later, args=(link_path, bw, data_size))
                t.start()
                release_threads.append(t)
            else:
                fukusou_sub_counter_ex += 1

        if not (no_path_sub_counter_ex == 0 and fukusou_sub_counter_ex == 0):
            no_path_counter_ex += 1
            no_path_all_counter_ex += no_path_sub_counter_ex
            fukusou_counter_ex += 1
            fukusou_all_counter_ex += fukusou_sub_counter_ex


        ## 最大負荷リンクにおける使用帯域幅の詳細値 [Mbps]
        for link in links.values():
            all_used_links_ex += link.used
            links_ex_list += [link.used]

        max_ex_log = max(links_ex_list)
        min_ex_log = min(links_ex_list)
        average_ex_log = statistics.mean(links_ex_list)
        std_dev_ex_log = statistics.stdev(links_ex_list)
        std_err_ex_log = std_dev_ex_log / (len(links_ex_list) ** 0.5)
        median_ex_log = statistics.median(links_ex_list)

        if max_load_link_ex < all_used_links_ex:
            max_load_link_ex = all_used_links_ex

            max_ex = max_ex_log
            min_ex = min_ex_log
            average_ex = average_ex_log
            std_dev_ex = std_dev_ex_log
            std_err_ex = std_err_ex_log
            median_ex = median_ex_log
            print(links_ex_list)

        for t in release_threads:
            t.join()
            
        # リンクの一斉解放
        for link in links.values():
            link.used = 0



### 結果の表示 ###
print("=== Imformation ===")
print(f'データセット数: {len(data)}')
print(f'トポロジーのノード数: {len(nodes)} (NP_2={NP_2})')
print()


print("=== Results of Dijkstra ===")
print(f'デマンド集合ごとのno path件数: {no_path_counter_djk}')
print(f'トラフィックごとのno path件数: {no_path_all_counter_djk}')
print(f'デマンド集合ごとの輻輳件数: {fukusou_counter_djk}')
print(f'トラフィックごとの輻輳件数: {fukusou_all_counter_djk}')
print(f'path_djk_2={path_djk_2}, path_djk_3={path_djk_3}, path_djk_4={path_djk_4}, path_djk_5={path_djk_5}, path_djk_6={path_djk_6}')
print(" == Detailed values ​​of bandwidth used on the most loaded link ==")
print(f'Max: {max_djk}, Min: {min_djk}, Average: {average_djk}, SD: {std_dev_djk}, SE: {std_err_djk}, Median: {median_djk}')


print(f'評価項目 (1b): {evaluation_1a/(len(data)) *100} ({evaluation_1a}/{len(data)} トラフィック)')
print()


print("=== Results of Exsitring Proposal ===")
print(f'デマンド集合ごとのno path件数: {no_path_counter_ex}')
print(f'トラフィックごとのno path件数: {no_path_all_counter_ex}')
print(f'デマンド集合ごとの輻輳件数: {fukusou_counter_ex}')
print(f'トラフィックごとの輻輳件数: {fukusou_all_counter_ex}')
print(f'path_ex_2={path_ex_2}, path_ex_3={path_ex_3}, path_ex_4={path_ex_4}, path_ex_5={path_ex_5}, path_ex_6={path_ex_6}')
print(" == Detailed values ​​of bandwidth used on the most loaded link ==")
print(f'Max: {max_ex}, Min: {min_ex}, Average: {average_ex}, SD: {std_dev_ex}, SE: {std_err_ex}, Median: {median_ex}')

print(f'評価項目 (1b): {evaluation_1b/(len(data)) *100} ({evaluation_1b}/{len(data)} トラフィック)')
print()