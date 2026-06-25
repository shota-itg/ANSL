# data /network.py

"""
既存研究の図7(a)のトポロジ
    6ノード，8リンク

全ノード間でトラフィックを1つずつ発生
    P(6, 2) = 30個
    30個を1つのトラフィックデマンド集合とする

解放処理
    トラフィックデマンド集合（30個）ごとに全リンクを解放
    
# 全ノードからの拠点間トラフィックが 1 データ × 任意の回数
"""

import time
import random
import csv
import heapq
from collections import deque
import threading
from sklearn.utils import Bunch
import numpy as np
import statistics

csv_name = "test"
nodes = [0, 1, 2, 3, 4, 5]    # ノードの定義
num_datasets = 10000 # ほしいデータセット数を設定

# 帯域幅の選択肢 [Mbps]
bandwidth_options = [150, 300, 450, 600, 750, 900, 1000]

"""
path_2 = 0
path_3 = 0
path_4 = 0
path_5 = 0
path_6 = 0
"""



### リンククラス：容量と使用量を管理 ###
class Link:

    ### __init__はクラスのコンストラクタ（初期化関数） ###
        # クラスからオブジェクトを作ったときに最初に呼ばれる関数
    def __init__(self, max_capacity):   # self: クラスの自分自身（インスタンス）を指す変数
        self.max_capacity = max_capacity    # max_capacity: 最大帯域（例: 1000Mbps）
        self.used = 0   # used: 現在使用中の帯域
        self.lock = threading.Lock()    # lock: スレッド安全性を確保するためのロック（複数スレッドが同時に帯域を変更しないように）

    """いらないかも
    ## 要求帯域 bw を確保できるかをチェック
    def check_links(self, bw):
        with self.lock:
            return self.used + bw <= self.max_capacity
    """
        
    ## 帯域の確保
        # 要求帯域 bw を確保できるか確認し，可能なら used を増やす
    def try_allocate(self, bw):
        with self.lock:
            if self.used + bw <= self.max_capacity:
                self.used += bw
                return self.used <= self.max_capacity

    ## 帯域の解放
        # 使用後に used を減らす
    def release(self, bw):
        with self.lock:
            self.used -= bw
            """debug
            if self.used < 0:
                print(f"[WARNING] Link over-released: used={self.used}, bw={bw}")
            """
            

    """debug
    
    """
    def status(self):
        return f'used: {self.used} /{self.max_capacity} Mbps'



### ネットワークを構築 ###
    # ノード間のリンク構成（4ノード・6リンク）
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



### 経路上のリンクを取得 ###
    # 経路に含まれるリンクのリストを取得
    # 経路 path=[0,1,3] なら，リンク (0,1) と (1,3) を取得
def get_links_from_path(path, links):
    link_list = []  # 空のリスト (list) # 経路に含まれるリンクを順番に追加

    if path:
        for i in range(len(path) - 1):
            link = links.get((path[i], path[i+1]))
            if link:
                link_list.append(link)

        return link_list
    else:
        return None



### メインの処理 ###
if __name__ == "__main__":  # Python ファイルを直接実行したときだけこのブロックを実行するという意味 # 他のファイルからインポートされたときは実行されない
    queue = deque() # 通信要求キュー

    # list
    links_log = []
    data_log, data_one_hot_log = [], []
    target_log, target_one_hot_log = [], []
    traffic_log, traffic_one_hot_log = [], []

    # counter
    fukusou_counter, fukusou_all_counter = 0, 0
    no_path_counter, no_path_all_counnter = 0, 0

    max_load_link_ex = 0
    max_load_link_list_ex = []

    i = 0

    links = build_network() # links はディクショナリで Link クラス

    def data_to_one_hot(src, dst, bw, num_nodes):   # num_nodes: ネットワークのノード数
        one_hot = []
        one_hot = [0] * num_nodes + [bw]
        one_hot[src] = -1
        one_hot[dst] = 1

        return one_hot

    def target_to_one_hot(path, num_nodes):
        total_one_hot = []
        for i in range(num_nodes):
            one_hot = []
            one_hot = [0] * (num_nodes+1)
            if not path[i] == -1:
                one_hot[path[i]] = 1
            else:
                one_hot[num_nodes] = 1

            total_one_hot += one_hot

        return total_one_hot

    while len(target_log) < num_datasets:
        i += 1
        
        links_ex_list = []
        traffic_list = []
        data_list, data_one_hot_list = [], []
        target_list, target_one_hot_list = [], []
        links_list = []
        release_threads = []

        fukusou_sub_counter = 0
        no_path_sub_counter = 0
        no_path_checker = True

        all_used_links_ex = 0

        # nodes をランダムにシャフル
        random.shuffle(nodes)

        # 全ノード間トラフィック（1つのトラフィックデマンド集合）を生成
        for src in nodes:
            dsts = [n for n in nodes if n != src]
            dst = random.choice(dsts)
            for dst in dsts:
                bw = random.choice(bandwidth_options)   # bandwidth_options: ランダムで帯域を選択
                traffic_list.append((src, dst, bw))

        # queue に追加
        for src, dst, bw in traffic_list:
            queue.append((src, dst, bw))


        ## トラフィック処理
        for _ in range(int(len(traffic_list) /2)):
            graph = generate_graph(links)   # graph はディクショナリ
            src, dst, bw = queue.popleft()  # キューから [src, dst, bw] のリストを取り出し，それぞれの変数に代入
            path = dijkstra(graph, src, dst)          

            """ debug
            print(f'試行回数: {i}')
            print(f'ログ数: {len(target_log)}')
            # リンク使用量
            for (s, d), link in links.items():
                if s < d:
                    print(f'Link {s} --> {d} ({d} --> {s}): {link.status()}')
            print(f'< {t+1}/{len(traffic_list)} > src={src}, dst={dst}, bw={bw}, path={path}')
            """
            

            if not path:
                no_path_checker = False
                no_path_sub_counter += 1
                
                # debug #print("--------------------------------------------------> not path")

                continue


            ## 帯域確保と呼損判定
                # 経路上のすべてのリンクで帯域を確保できれば成功
                # どれか 1 つでも失敗すれば呼損
            link_path = get_links_from_path(path, links)
            # success = all(link.check_links(bw) for link in link_path)
            allocated_links = []
            for link in link_path:
                if link.try_allocate(bw):
                    allocated_links.append(link)
                
            # debug #print(f'debug: len(allocated_links)={len(allocated_links)}, len(link_path)={len(link_path)}')

            # 成功した通信は1秒後に帯域を開放（スレッドで非同期実行）
            if len(allocated_links) == len(link_path):
                continue
            # 失敗した通信は呼損
            else:
                for l in allocated_links:
                    l.release(bw)
                    
                fukusou_sub_counter += 1

                # debug #print("--------------------------------------------------> 輻輳")

            """ debug
            # リンク使用量
            for (s, d), link in links.items():
                if s < d:
                    print(f'Link {s} --> {d} ({d} --> {s}): {link.status()}')

            print(f'デマンド集合ごとの輻輳件数: {fukusou_sub_counter}/{len(traffic_list)}')
            if 0 < fukusou_sub_counter:
                print(f'トラフィックごとの輻輳件数: {fukusou_all_counter+fukusou_sub_counter}/{i*len(traffic_list)}')
            else:
                print(f'トラフィックごとの輻輳件数: {fukusou_all_counter}/{i*len(traffic_list)}')
            print()
            """   

        # modality2
        if no_path_checker and fukusou_sub_counter == 0:
            # 全ノード間トラフィックの半分のパスを張った状態のリンク情報を保存
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if (i,j) in links:
                        links_list += [links[(i,j)].used]
                    else:
                        links_list += [-1]
        
        traffic_list = []   # リセット
        while queue:
            traffic_list.append(queue.popleft())

        # bw 降順でソート
        traffic_list.sort(key=lambda x: x[2], reverse=True)

        # queue に追加
        for src, dst, bw in traffic_list:
            queue.append((src, dst, bw))


        ## トラフィック処理
        while queue:
            graph = generate_graph(links)   # graph はディクショナリ
            src, dst, bw = queue.popleft()  # キューから [src, dst, bw] のリストを取り出し，それぞれの変数に代入
            path = dijkstra(graph, src, dst)          

            """ debug
            print(f'試行回数: {i}')
            print(f'ログ数: {len(target_log)}')
            # リンク使用量
            for (s, d), link in links.items():
                if s < d:
                    print(f'Link {s} --> {d} ({d} --> {s}): {link.status()}')
            print(f'< {t+1}/{len(traffic_list)} > src={src}, dst={dst}, bw={bw}, path={path}')
            """
            

            if not path:
                no_path_checker = False
                no_path_sub_counter += 1
                
                # debug #print("--------------------------------------------------> not path")

                continue


            ### 帯域確保と呼損判定
                # 経路上のすべてのリンクで帯域を確保できれば成功
                # どれか 1 つでも失敗すれば呼損
            link_path = get_links_from_path(path, links)
            # success = all(link.check_links(bw) for link in link_path)
            allocated_links = []
            for link in link_path:
                if link.try_allocate(bw):
                    allocated_links.append(link)
                
            # debug #print(f'debug: len(allocated_links)={len(allocated_links)}, len(link_path)={len(link_path)}')

            # 成功した通信は1秒後に帯域を開放（スレッドで非同期実行）
            if len(allocated_links) == len(link_path):
                # data
                data_list += [src, dst, bw]
                data_one_hot_list += data_to_one_hot(src, dst, bw, len(nodes))  # one-hot 形式に変換後， data_one_hot_list に追加

                # target
                padded_path = path + [-1] * (len(nodes)-len(path)) # path の残りを -1 で埋める
                target_list += padded_path
                target_one_hot = target_to_one_hot(padded_path, len(nodes))  # one-hot 形式に変換
                target_one_hot_list += target_one_hot  # one-hot 形式に変換後，target_one_hot_list に追加

                """ 解放処理
                def release_later(each_link, delay_time, bandwidth):
                    for l in each_link:
                        time.sleep(delay_time)
                        l.release(bandwidth)

                # threading.Thread(target=release_later, args=(link_path)).start()
                t = threading.Thread(target=release_later, args=(link_path, delay, bw))
                t.start()
                release_threads.append(t)
                """
                
                            
            # 失敗した通信は呼損
            else:
                for l in allocated_links:
                    l.release(bw)
                    
                fukusou_sub_counter += 1

                # debug #print("--------------------------------------------------> 輻輳")

            """ debug
            # リンク使用量
            for (s, d), link in links.items():
                if s < d:
                    print(f'Link {s} --> {d} ({d} --> {s}): {link.status()}')

            print(f'デマンド集合ごとの輻輳件数: {fukusou_sub_counter}/{len(traffic_list)}')
            if 0 < fukusou_sub_counter:
                print(f'トラフィックごとの輻輳件数: {fukusou_all_counter+fukusou_sub_counter}/{i*len(traffic_list)}')
            else:
                print(f'トラフィックごとの輻輳件数: {fukusou_all_counter}/{i*len(traffic_list)}')
            print()
            """        


        if no_path_checker:
            ## 最大負荷リンクにおける使用帯域幅の詳細値 [Mbps]
            for link in links.values():
                links_ex_list += [link.used]

                # debug #print(link.used)

            # debug #print(links_ex_list)

            if fukusou_sub_counter == 0:
                links_log.append(links_list)
                data_log.append(data_list)
                data_one_hot_log.append(data_one_hot_list)
                target_log.append(target_list)
                target_one_hot_log.append(target_one_hot_list)
                
                max_load_link_list_ex += [max(links_ex_list)]
            elif 0 < fukusou_sub_counter:
                fukusou_counter += 1
                fukusou_all_counter += fukusou_sub_counter
        else:
            no_path_counter += 1
            no_path_all_counnter += no_path_sub_counter

        # 全リンクを解放
        for link in links.values():
            link.used = 0

        # debug #print()

    
    max_ex = max(max_load_link_list_ex)
    min_ex = min(max_load_link_list_ex)
    average_ex = statistics.mean(max_load_link_list_ex)
    std_dev_ex = statistics.stdev(max_load_link_list_ex)
    std_err_ex = std_dev_ex / (len(max_load_link_list_ex) ** 0.5)
    median_ex = statistics.median(max_load_link_list_ex)



    ### 結果表示 ###
    traffic_log = Bunch(
        data=np.array(data_log), 
        target=np.array(target_log)
       )
    # debug #print(f'traffic_log: \n{traffic_log}')

    traffic_one_hot_log = Bunch(
        data_one_hot=np.array(data_one_hot_log), 
        target_one_hot=np.array(target_one_hot_log)
    )
    # debug #print(f'traffic_one_hot_log: \n{traffic_one_hot_log}')
    print(f'{csv_name}データセット収集完了\nデータセット数: {len(target_log)}\nデマンド集合ごとの輻輳件数: {fukusou_counter}\nトラフィックごとの輻輳件数: {fukusou_all_counter}\nデマンド集合ごとのno path件数: {no_path_counter}\nトラフィックごとのno path件数: {no_path_all_counnter}')
    # print(f'path_2={path_2}, path_3={path_3}, path_4={path_4}, path_5={path_5}, path_6={path_6}')
    # debug #print(f'max_load_link_list: {max_load_link_list_ex}')
    print(f'Max: {max_ex}, Min: {min_ex}, Average: {average_ex}, SD: {std_dev_ex}, SE: {std_err_ex}, Median: {median_ex}')
    print()



    ### CSV保存 ###
    

    if csv_name == "train":
        with open("train_traffic_log.csv", "w", newline="") as f:
            writer = csv.writer(f)

            # ヘッダーの自動生成（例: src0 ~ dst3, bw, path0 ~ pathN）
            data_len = len(data_log[0]) if data_log else 0
            target_len = len(target_log[0]) if target_log else 0
            header = [f"data_{i}" for i in range(data_len)] + [f"target_{i}" for i in range(target_len)]
            writer.writerow(header)

            # 各サンプルを 1 行にまとめて保存
            for d, t in zip(data_log, target_log):
                writer.writerow(d + t)

        with open("train_dataset.csv", "w", newline="") as f:
            writer = csv.writer(f)

            # ヘッダーの自動生成（例: src0 ~ dst3, bw, path0 ~ pathN）
            data_len = len(data_one_hot_log[0]) if data_one_hot_log else 0
            links_len = len(links_log[0]) if links_log else 0
            target_len = len(target_one_hot_log[0]) if target_one_hot_log else 0
            header = [f"data_{i}" for i in range(data_len)] + [f"links_{i}" for i in range(links_len)] + [f"target_{i}" for i in range(target_len)]
            writer.writerow(header)

            # 各サンプルを 1 行にまとめて保存
            for d, l, t in zip(data_one_hot_log, links_log, target_one_hot_log):
                writer.writerow(d + l + t)
    elif csv_name == "test":
        with open("test_traffic_log.csv", "w", newline="") as f:
            writer = csv.writer(f)

            # ヘッダーの自動生成（例: src0 ~ dst3, bw, path0 ~ pathN）
            data_len = len(data_log[0]) if data_log else 0
            target_len = len(target_log[0]) if target_log else 0
            header = [f"data_{i}" for i in range(data_len)] + [f"target_{i}" for i in range(target_len)]
            writer.writerow(header)

            # 各サンプルを 1 行にまとめて保存
            for d, t in zip(data_log, target_log):
                writer.writerow(d + t)

        with open("test_dataset.csv", "w", newline="") as f:
            writer = csv.writer(f)

            # ヘッダーの自動生成（例: src0 ~ dst3, bw, path0 ~ pathN）
            data_len = len(data_one_hot_log[0]) if data_one_hot_log else 0
            links_len = len(links_log[0]) if links_log else 0
            target_len = len(target_one_hot_log[0]) if target_one_hot_log else 0
            header = [f"data_{i}" for i in range(data_len)] + [f"links_{i}" for i in range(links_len)] + [f"target_{i}" for i in range(target_len)]
            writer.writerow(header)

            # 各サンプルを 1 行にまとめて保存
            for d, l, t in zip(data_one_hot_log, links_log, target_one_hot_log):
                writer.writerow(d + l + t)
