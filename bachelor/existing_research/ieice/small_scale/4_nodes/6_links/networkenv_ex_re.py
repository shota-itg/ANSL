#networkenv_ex_re.py

"""
既存研究の図7(a)のトポロジ
任意の回数で全ノードからトラフィックを一つずつ発生させる
全ノードからの拠点間トラフィックが 1 データ × 任意の回数
リンクは一つのデマンド集合の処理後にまとめて解放
"""

import time
import random
import csv
import heapq
from multiprocessing import Process, Queue
import threading
from sklearn.utils import Bunch
import numpy as np

# 帯域幅の選択肢 [Mbps]
bandwidth_options = [150, 300, 450, 600, 750, 900, 1000]
data_size_options = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]




### リンククラス：容量と使用量を管理 ###
class Link:

    ### __init__はクラスのコンストラクタ（初期化関数） ###
        # クラスからオブジェクトを作ったときに最初に呼ばれる関数
    def __init__(self, max_capacity):   # self: クラスの自分自身（インスタンス）を指す変数
        self.max_capacity = max_capacity    # max_capacity: 最大帯域（例: 1000Mbps）
        self.used = 0   # used: 現在使用中の帯域
        self.lock = threading.Lock()    # lock: スレッド安全性を確保するためのロック（複数スレッドが同時に帯域を変更しないように）

    ### 帯域の確保 ###
        # 要求帯域 bw を確保できるか確認し，可能なら used を増やす
    def try_allocate(self, bw):
        with self.lock:
            if self.used + bw <= self.max_capacity:
                self.used += bw

                return True

            return False

    ### 帯域の解放 ###
        # 使用後に used を減らす
    def release(self, bw):
        with self.lock:
            self.used -= bw

    ##デバック用
    def status(self):
        return f'used: {self.used} / {self.max_capacity} Mbps'



### ネットワークを構築 ###
    # ノード間のリンク構成（4ノード・6リンク）
def build_network():
    links = {}  # 空の辞書 (dictionary)
    # ノード間リンク（双方向）
    links[(0,1)] = Link(10000)   # (0,1) は辞書 (dictionary) のキー（タプル）    # Link(1000) は Link クラスのインスタンス
    links[(1,0)] = links[(0,1)]
    links[(0,2)] = Link(10000)
    links[(2,0)] = links[(0,2)]
    links[(0,3)] = Link(10000)
    links[(3,0)] = links[(0,3)]
    links[(1,2)] = Link(10000)
    links[(2,1)] = links[(1,2)]
    links[(1,3)] = Link(10000)
    links[(3,1)] = links[(1,3)]
    links[(2,3)] = Link(10000)
    links[(3,2)] = links[(2,3)]
    
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
        graph[src][dst] = 1 / (link.max_capacity - link.used + 1e-6)    # 使用量に応じて重み変化    # 二次元辞書（辞書の中に辞書）  # 1e-6: ゼロ除算を防ぐための微小値
    
    return graph



### ダイクストラ法で経路探索 ###
def dijkstra(graph, start, end):
    queue = [(0, start, [start])]   # queue: 優先度付きキュー (heapq) でコストの小さい順に探索  # [start]: 経路を記録するためのリストで，最初は「自分自身だけの経路」なので [start]
    visited = set() # visited: 探索済みノードを記録 # Python の集合型 (set) で，重複を許さないデータ構造 # 一度訪れたノードを記録し，再訪問を防ぐ
    while queue:
        cost, node, path = heapq.heappop(queue) # path: 現在の経路  # heapq: 優先度付きキューを扱うモジュール   # heappop: コストが最小の要素を取り出す関数 # queueの中身は（コスト, 現在のノード, 経路）のタプル
        if node == end:
            return path
        if node in visited:
            continue
        visited.add(node)   # 今訪れたノードを visited に追加   # 「このノードはもう探索済み」と記録するため
        for neighbor, weight in graph.get(node, {}).items():    # neighbor: 隣接ノード  # graph.get(node, {}): 「 node の隣接ノード一覧」を取得 # .items(): 「隣接ノードと重み」のペアを取り出す
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor])) # .heappush: 新しい経路候補をキューに追加   # path + [neighbor]: 今の経路に neighbor を追加した新しい経路
    
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
if __name__ == "__main__":  # Python ファイルを直接実行したときだけこのブロックを実行するという意味 # 他のファイルからインポートされたときは実行されない
    nodes = [0, 1, 2, 3]    # ノードの定義
    queue = Queue() # 通信要求キュー
    data_log, data_one_hot_log = [], []
    target_log, target_one_hot_log = [], []
    traffic_log, traffic_one_hot_log = [], []
    num_iterations = 25000 # 任意の回数 # 25000にあとで設定
    fukusou_counter = 0

    i = 1

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

    while len(data_log) < num_iterations:
        print(f'>>> デバック --> {i}回目')
        i += 1
        traffic_list = []
        data_list, data_one_hot_list = [], []
        target_list, target_one_hot_list = [], []

        fukusou_sub_counter = 0

        # nodes をランダムにシャフル
        random.shuffle(nodes)

        # 全ノードから一つずつトラフィックを生成
        for src in nodes:
            dsts = [n for n in nodes if n != src]
            dst = random.choice(dsts)
            for dst in dsts:
                bw = random.choice(bandwidth_options)   # bandwidth_options: ランダムで帯域を選択
                traffic_list.append((src, dst, bw))

        # bw 降順でソート
        traffic_list.sort(key=lambda x: x[2], reverse=True)

        # queue に追加
        for src, dst, bw in traffic_list:
            queue.put((src, dst, bw))

        ### トラフィック処理 ###
        for _ in range(len(traffic_list)):
            graph = generate_graph(links)   # graph はディクショナリ
            src, dst, bw = queue.get()  # キューから [src, dst, bw] のリストを取り出し，それぞれの変数に代入
            path = dijkstra(graph, src, dst)
            data_size = (random.choice(data_size_options) *8)

            if not path:
                fukusou_sub_counter += 1
                print(">>> デバック --> not path")
                """
                #data
                data_list += [src, dst, bw]
                data_one_hot_list += data_to_one_hot(src, dst, bw, len(nodes))  # one-hot 形式に変換後， data_list に追加

                #target
                path = [-1] * len(nodes)
                target_list += path
                target_one_hot_list += target_to_one_hot(path, len(nodes))
                """
                time.sleep(random.uniform(8e-1*1e-3, 53.3e+0*1e-3))

            ### 帯域確保と呼損判定 ###
                # 経路上のすべてのリンクで帯域を確保できれば成功
                # どれか 1 つでも失敗すれば呼損
            link_path = get_links_from_path(path, links)
            success = all(link.try_allocate(bw) for link in link_path)

            # 成功した通信は1秒後に帯域を開放（スレッドで非同期実行）
            if success:
                # data
                data_list += [src, dst, bw]
                data_one_hot_list += data_to_one_hot(src, dst, bw, len(nodes))  # one-hot 形式に変換後， data_one_hot_list に追加

                # target
                padded_path = path + [-1] * (len(nodes)-len(path)) # path の残りを -1 で埋める
                target_list += padded_path
                target_one_hot = target_to_one_hot(padded_path, len(nodes))  # one-hot 形式に変換
                target_one_hot_list += target_one_hot  # one-hot 形式に変換後，target_one_hot_list に追加

                def release_later(each_link, bw, data_size):
                    delay = data_size / bw
                    time.sleep(delay *9e-4)
                    for l in each_link:
                        l.release(bw)

                threading.Thread(target=release_later, args=(link_path, bw, data_size)).start()

                # time.sleep(random.uniform(8e-1*1e-24, 53.3e+0*1e-24))

            # 失敗した通信は呼損
            else:
                fukusou_sub_counter += 1
                print(">>> デバック --> 輻輳")
                """
                # data
                data_list += [src, dst, bw]
                data_one_hot_list += data_to_one_hot(src, dst, bw, len(nodes))  # one-hot 形式に変換後，data_one_hot_list に追加

                # target
                path = [-1] * len(nodes)
                target_list += path
                target_one_hot_list += target_to_one_hot(path, len(nodes))
                """
                time.sleep(random.uniform(8e-1*1e-3, 53.3e+0*1e-3))

            # time.sleep(random.uniform(8e-1*1e-3, 53.3e+0*1e-3))

        if fukusou_sub_counter == 0:
            data_log.append(data_list)
            data_one_hot_log.append(data_one_hot_list)
            target_log.append(target_list)
            target_one_hot_log.append(target_one_hot_list)
            print(f'>>> デバック --> ログ数: {len(data_log)+1}')

        if 0 < fukusou_sub_counter:
            fukusou_counter += 1

        if fukusou_sub_counter == len(traffic_list):
            print(">>> デバック --> リセット")
            for link in links.values():
                link.used = 0

        """"
        # リンク使用量をリセット（デマンド集合終了後にまとめて解放）
        for link in links.values():
            link.used = 0
        """

        for (src, dst), link in links.items():
            if src < dst:
                print(f'Link {src} --> {dst}: {link.status()}')
        print()



    ### 結果表示 ###
    traffic_log = Bunch(
        data=np.array(data_log), 
        target=np.array(target_log)
       )
    print(f'traffic_log: \n{traffic_log}')
    print()

    traffic_one_hot_log = Bunch(
        data_one_hot=np.array(data_one_hot_log), 
        target_one_hot=np.array(target_one_hot_log)
    )
    print(f'traffic_one_hot_log: \n{traffic_one_hot_log}')
    print(f'収集完了\nデータセット数: {len(target_log)}\nデマンド集合ごとの輻輳件数: {fukusou_counter}\n全体での輻輳件数: {fukusou_sub_counter}')



    ### CSV保存 ###
    with open("traffic_log.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # ヘッダーの自動生成（例: src0 ~ dst3, bw, path0 ~ pathN）
        data_len = len(data_log[0]) if data_log else 0
        target_len = len(target_log[0]) if target_log else 0
        header = [f"data_{i}" for i in range(data_len)] + [f"target_{i}" for i in range(target_len)]
        writer.writerow(header)

        # 各サンプルを 1 行にまとめて保存
        for d, t in zip(data_log, target_log):
            writer.writerow(d + t)

    with open("traffic_one_hot_log.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # ヘッダーの自動生成（例: src0 ~ dst3, bw, path0 ~ pathN）
        data_len = len(data_one_hot_log[0]) if data_one_hot_log else 0
        target_len = len(target_one_hot_log[0]) if target_one_hot_log else 0
        header = [f"data_{i}" for i in range(data_len)] + [f"target_{i}" for i in range(target_len)]
        writer.writerow(header)

        # 各サンプルを 1 行にまとめて保存
        for d, t in zip(data_one_hot_log, target_one_hot_log):
            writer.writerow(d + t)
