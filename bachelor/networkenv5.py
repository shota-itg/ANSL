
"""
one-hot データの次元が違う
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


## リンククラス：容量と使用量を管理 ##
class Link:

    ## __init__はクラスのコンストラクタ（初期化関数） ##
        # クラスからオブジェクトを作ったときに最初に呼ばれる関数
    def __init__(self, max_capacity):   # self: クラスの自分自身（インスタンス）を指す変数
        self.max_capacity = max_capacity    # max_capacity: 最大帯域（例: 1000Mbps）
        self.used = 0   # used: 現在使用中の帯域
        self.lock = threading.Lock()    # lock: スレッド安全性を確保するためのロック（複数スレッドが同時に帯域を変更しないように）

    ## 帯域の確保 ##
        # 要求帯域 bw を確保できるか確認し，可能なら used を増やす
    def try_allocate(self, bw):
        with self.lock:
            if self.used + bw <= self.max_capacity:
                self.used += bw

                return True

            return False

    ## 帯域の解放 ##
        # 使用後に used を減らす
    def release(self, bw):
        with self.lock:
            self.used -= bw


## ネットワークを構築 ##
    # ノード間のリンク構成（4ノード・6リンク）
def build_network():
    links = {}  # 空の辞書 (dictionary)
    # ノード間リンク（双方向）
    links[(0,1)] = Link(1000)   # (0,1) は辞書 (dictionary) のキー（タプル）    # Link(1000) は Link クラスのインスタンス
    links[(1,0)] = links[(0,1)]
    links[(0,2)] = Link(1000)
    links[(2,0)] = links[(0,2)]
    links[(0,3)] = Link(1000)
    links[(3,0)] = links[(0,3)]
    links[(1,2)] = Link(1000)
    links[(2,1)] = links[(1,2)]
    links[(1,3)] = Link(1000)
    links[(3,1)] = links[(1,3)]
    links[(2,3)] = Link(1000)
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


## ダイクストラ法で経路探索 ##
def dijkstra(graph, start, end):
    queue = [(0, start, [start])]   # queue: 優先度付きキュー (heapq) でコストの小さい順に探索  # [start]: 経路を記録するためのリストで，最初は「自分自身だけの経路」なので [start]
    visited = set() # visited: 探索済みノードを記録 # Pythonの集合型 (set) で，重複を許さないデータ構造 # 一度訪れたノードを記録し，再訪問を防ぐ
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


## トラフィック生成関数 ##
def generate_traffic(node_id, other_nodes, queue, prob=0.4):    # prob: 通信を発生させる確率 (40%)
    for dst in other_nodes:
        if random.random() < prob:
            bw = random.choice(bandwidth_options)   # bandwidth_options: ランダムに帯域を選択
            queue.put([node_id, dst, bw])   # queue: 他プロセスと共有する通信要求のキュー


## 経路上のリンクを取得 ##
    # 経路に含まれるリンクのリストを取得
    # 経路 path=[0,1,3] なら，リンク (0,1) と (1,3) を取得
def get_links_from_path(path, links):
    link_list = []  # 空のリスト (list) # 経路に含まれるリンクを順番に追加
    for i in range(len(path) - 1):
        link = links.get((path[i], path[i+1]))
        if link:
            link_list.append(link)

    return link_list


## メインの処理 ##
if __name__ == "__main__":  # Python ファイルを直接実行したときだけこのブロックを実行するという意味 # 他のファイルからインポートされたときは実行されない
    nodes = [0, 1, 2, 3]    # ノードの定義
    one_hot = {
        -1: [0, 0, 0, 0], 
        0: [1, 0, 0, 0], 
        1: [0, 1, 0, 0], 
        2: [0, 0, 1, 0], 
        3: [0, 0, 0, 1]
    }
    queue = Queue() # 通信要求キュー
    data_log = []   # 空のリスト (list)
    data_one_hot = []   # 空のリスト (list)
    target_log = [] # 空のリスト (list)
    target_one_hot = [] # 空のリスト (list)
    duration = 10  # 任意の実行時間 [秒]
    start_time = time.time()

    links = build_network()

    while time.time() - start_time < duration:
        
        ## 並列トラフィック生成 ##
            # 各ノードから他ノードへの通信要求を並列生成 (muliprocessing)
        processes = []  # 空のリスト (list) # 各ノードのトラフィック生成プロセスを格納するためのリスト
        for src in nodes:
            dsts = [n for n in nodes if n != src]   # リスト内包表記    # src 以外のノードを dsts に入れている（送信元以外の宛先候補）
            p = Process(target=generate_traffic, args=(src, dsts, queue))   # Process: multiprocessing のモジュールのクラス # target: 実行する関数  # args: その関数に渡す引数
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        ## トラフィック処理 ##
        while not queue.empty():
            graph = generate_graph(links)
            src, dst, bw = queue.get()  # キューから [src, dst, bw] のリストを取り出し，それぞれの変数に代入
            path = dijkstra(graph, src, dst)

            def data_to_one_hot(src, dst, bw, num_nodes):   # num_nodes: ネットワークのノード数
                one_hot = []
                one_hot = [0] * num_nodes + [bw]
                one_hot[src] = -1
                one_hot[dst] = 1

                return one_hot

            if not path:
                #data
                data_log.append(data_to_one_hot(src, dst, bw, len(nodes)))   # one-hot 形式に変換後， data_log に追加

                #target
                target_log.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 1 は call_loss の意
                continue

            ## 帯域確保と呼損判定 ##
                # 経路上のすべてのリンクで帯域を確保できれば成功
                # どれか 1 つでも失敗すれば呼損
            link_path = get_links_from_path(path, links)
            success = all(link.try_allocate(bw) for link in link_path)

            # 成功した通信は1秒後に帯域を開放（スレッドで非同期実行）
            if success:
                # 解放処理を別スレッドで遅延実行
                def release_later(links_to_release, bw, delay=1.0):
                    time.sleep(delay)
                    for l in links_to_release:
                        l.release(bw)

                threading.Thread(target=release_later, args=(link_path, bw)).start()

                # data
                data_log.append(data_to_one_hot(src, dst, bw, len(nodes)))   # one-hot 形式に変換後， data_log に追加

                # target
                padded_path = path + [-1] * (len(nodes)-len(path)+1) # path の残りを -1 で埋める
                target_one_hot = one_hot[padded_path[0]] + one_hot[padded_path[1]] + one_hot[padded_path[2]] + one_hot[padded_path[3]] + [0]  # one-hot 形式に変換
                target_log.append(target_one_hot)   # one-hot 形式に変換後， target_log に追加
            # 失敗した通信は呼損
            else:
                # data
                data_log.append(data_to_one_hot(src, dst, bw, len(nodes)))   # one-hot 形式に変換後， data_log に追加

                # target
                target_log.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])   # 1 は call_loss の意

        time.sleep(0.5)


    """
    ### デバック
    print("### デバック ###")
    print(f'data_log: \n{data_log}')
    print(f'target_log: \n{target_log}')
    print()
    """


    ## 結果表示 ##
    traffic_log = Bunch(
        data=np.array(data_log), 
        target=np.array(target_log)
       )   # ログ保存
    print(traffic_log)
    print("収集完了\n件数:", len(data_log) + len(target_log))


    ## CSV保存 ##
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






"""
    付録
"""
# one-hot に変換
# def path_to_one_hot(path, num_nodes): # num_nodes: ネットワークのノード数
#   one_hot = []
#   for node in path:
#       one_hot = [0] * num_nodes
#       if 0 <= node < num_nodes:
#           one_hot[node] = 1
#       one_hot.append(one_hot)
# 
#   return one_hot