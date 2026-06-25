# data_gen/network_core.py

import threading


## リンククラス：容量と使用量を管理
class Link:

    ### __init__はクラスのコンストラクタ（初期化関数） ###
        # クラスからオブジェクトを作ったときに最初に呼ばれる関数
    def __init__(self, max_capacity):   # self: クラスの自分自身（インスタンス）を指す変数
        self.max_capacity = max_capacity    # max_capacity: 最大帯域（例: 1000Mbps）
        self.used = 0   # used: 現在使用中の帯域
        self.lock = threading.Lock()    # lock: スレッド安全性を確保するためのロック（複数スレッドが同時に帯域を変更しないように）
        
    ## 帯域の確保
        # 要求帯域 bw を確保できるか確認し，可能なら used を増やす
    def try_allocate(self, bw):
        with self.lock:
            if self.used + bw <= self.max_capacity:
                self.used += bw
                return self.used <= self.max_capacity
            else:
                return False

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
        return f'used: {self.used} / {self.max_capacity} Mbps'


## ネットワークを構築
    # ノード間のリンク構成（4ノード・6リンク）
def build_network(config) -> dict:
    links_cfg = config["topology"]["links"]
    links = {}  # 空の辞書 (dictionary)

    for link in links_cfg:
        u = link["u"]
        v = link["v"]
        key = (min(u, v), max(u ,v))
        capa = link["capacity"]

        links[key] = Link(capa)
    
    return links


## グラフ構造（重みは帯域幅の逆数）
# リンク使用状況に応じた重み付きグラフを生成
    # 重みは「空き帯域の逆数」なので，空いているほど経路として優先される
def generate_graph(links):
    """
    Args: 
        links: {(min(u, v), max(u, v): Link)}
        graph: 隣接辞書
    Returns: 
    """
    """
    for (src, dst), link in links.items():  # links.items() は辞書 (dictionary) のすべてのキーと値のペアを取り出す  # 例）(0,1): Link(1000) → src=0, dst=1, link = Link(1000)
        ### print(f"Link ({src}->{dst}): used={link.used}, max_capacity={link.max_capacity}")   ### デバック用
        if src not in graph:    # もし src が graph にまだ登録されていなければ
            graph[src] = {} # 初めて見るノードなら，隣接ノードの情報を入れるための空の辞書 (dictionary) を作る準備

        if link.max_capacity == 0:
            graph[src][dst] = float('inf')
        elif link.used == link.max_capacity:
            graph[src][dst] = float('inf')
        else:
            graph[src][dst] = 1 / (link.max_capacity - link.used)    # 使用量に応じて重み変化    # 二次元辞書（辞書の中に辞書）  # 1e-6: ゼロ除算を防ぐための微小値    
    """

    graph = {}  # 空の辞書 (dictionary)
    for (u, v), link in links.items():
        if u not in graph:
            graph[u] = {}
        if v not in graph:
            graph[v] = {}
        
        if link.max_capacity == 0 or link.max_capacity <= link.used:
            weight_uv = float('inf')
            weight_vu = float('inf')
        else:
            w = 1 /(link.max_capacity -link.used)
            weight_uv = w
            weight_vu = w

        graph[u][v] = weight_uv
        graph[v][u] = weight_vu
        
    return graph