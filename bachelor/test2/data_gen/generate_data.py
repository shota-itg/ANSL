# data_gen/genearate_data3.py

import time
import os
import json
from collections import deque
import random
import statistics
import numpy as np

from utils.config_loader import load_config, load_hyperparameter, load_runtime, save_runtime

from data_gen.network_core import build_network, generate_static_graph, generate_graph
from data_gen.dijkstra import dijkstra, get_links_from_path
from data_gen.data_utils import data_to_one_hot, target_to_one_hot
from data_gen.save_data import save_data
from data_gen.preprocess import preprocess_data

config = load_config()
topo_name = config["topology"]["name"]
nodes = config["topology"]["nodes"]
topo_links = config["topology"]["links"]
bandwidth_options = config["bandwidth_options"]
data_json_path_cfg = config["paths"]["data"]
lf_data_json_path_cfg = config["paths"]["lf_data"]

runtime_cfg = load_runtime()
while True:
    data_name = input("Chose 'train' or 'test': ").strip().lower()
    if data_name in ("train", "test"):
        break
    else:
        print("\nError: 'train' か 'test' を入力してください。")
runtime_cfg["data"]["data_name"] = data_name
if data_name == "train":
    while True:
        try:
            num_traffic = int(input("再経路の対象トラフィック数: "))
            break
        except ValueError:
            print("\nError: 整数を入力してください。")
    while True:
        lf_enabled = input("リンク障害の有無 'true' or 'false': ")
        if lf_enabled in ("true", "false"):
            lf_enabled = (lf_enabled == "true")
            break
        else:
            print("\nError: 'true' か 'false' を入力してください。")
    if lf_enabled:
        while True:
            try:
                num_failure = int(input("リンク障害数: "))
                break
            except ValueError:
                print("\nError: 整数を入力してください。")
    while True:
        try:
            num_train_data = int(input("How many data?: "))   # ほしいデータセット数を設定
            break
        except ValueError:
            print("\nError: 整数を入力してください。")
    num_data = num_train_data
    runtime_cfg["data"]["num_train_data"] = num_train_data
else:
    while True:
        try:
            num_traffic = int(input("再経路の対象トラフィック数: "))
            break
        except ValueError:
            print("\nError: 整数を入力してください。")
    while True:
        lf_enabled = input("リンク障害の有無 'true' or 'false': ")
        if lf_enabled in ("true", "false"):
            lf_enabled = (lf_enabled == "true")
            break
        else:
            print("\nError: 'true' か 'false' を入力してください。")
    if lf_enabled:
        while True:
            try:
                num_failure = int(input("リンク障害数: "))
                break
            except ValueError:
                print("\nError: 整数を入力してください。")
    while True:
        try:
            num_test_data = int(input("How many data?: "))
            break
        except ValueError:
            print("\nError: 整数を入力してください。")
    num_data = num_test_data
    runtime_cfg["data"]["num_test_data"] = num_test_data
save_runtime(runtime_cfg)


queue = deque()

data_log, links_log, target_log, traffic_log = [], [], [], []
data_one_hot_log, target_one_hot_log, traffic_one_hot_log = [], [], []
links_shortest_log, mlu_shortest_log, std_shortest_log, cv_shortest_log = [], [], [], []
links_djk_log, mlu_djk_log, std_djk_log, cv_djk_log = [], [], [], []

fukusou_counter, fukusou_all_counter = 0, 0
no_path_counter, no_path_all_counter = 0, 0

max_load_link = 0
max_load_link_list = []

max_failure = 0

dijkstra_total_times_per_sample = [] # dijkstraの計測時間のリスト

while len(target_log) < num_data:
    links = build_network(config)

    data_list, links_list, target_list, traffic_list = [], [], [], []
    data_one_hot_list, target_one_hot_list = [], []
    links_shortest_list, links_djk_list = [], []

    # links_ex_list = []
    release_threads = []
    failed_links_list = []

    fukusou_sub_counter, no_path_sub_counter = 0, 0
    no_path_checker = True
    queue_checker = True
    no_continue = False

    current_sample_dijkstra_duration = 0.0 # トラフィックデマンド集合ごとの計測用変数を初期化
    # debug #
    


    """
    nodes_shuffled = nodes.copy()
    random.shuffle(nodes_shuffled)

    # 全ノード間トラフィック（1つのトラフィックデマンド集合）を生成
    for src in nodes_shuffled:
        dsts = [n for n in nodes_shuffled if n != src]
        for dst in dsts:
            bw = random.choice(bandwidth_options)
            traffic_list.append((src, dst, bw))
    """
    # 全ノード間トラフィック（1つのトラフィックデマンド集合）を生成
    for src in nodes:
        dsts = [n for n in nodes if n != src]
        for dst in dsts:
            bw = random.choice(bandwidth_options)
            traffic_list.append((src, dst, bw))

    random.shuffle(traffic_list)
    # debug #
    print(traffic_list)

    # queue に追加
    for src, dst, bw in traffic_list:
        queue.append((src, dst, bw))

    traffic_list = [] # リセット


    ## トラフィック処理
    while queue:
        graph = generate_static_graph(links)   # graph はディクショナリ
        src, dst, bw = queue.popleft()  # キューから [src, dst, bw] のリストを取り出し，それぞれの変数に代入
        path = dijkstra(graph, src, dst) # リンクコスト固定で最短経路を選択

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
            break

        ## 帯域確保と呼損判定
            # 経路上のすべてのリンクで帯域を確保できれば成功
            # どれか 1 つでも失敗すれば呼損
        link_path = get_links_from_path(path, links)
        # success = all(link.check_links(bw) for link in link_path)
        allocated_links = []
        for link in link_path:
            # print(f'path: {path}, link_path: {link[0]}')
            if link[1].allocate(bw):
                allocated_links.append(link[1])

        traffic_list.append((src, dst, bw, link_path))

        # debug #print(f'debug: len(allocated_links)={len(allocated_links)}, len(link_path)={len(link_path)}')

        # 成功した通信は1秒後に帯域を開放（スレッドで非同期実行）
        if len(allocated_links) == len(link_path):
            continue

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
    """
    if no_path_checker is False:
        # 全リンクを解放
        for link in links.values():
            link.used = 0
        continue    
    """

    # Dijkstra（最短）で計算した後のネットワーク状態
    for link in links.values():
        links_shortest_list += [link.used]

    # リンク障害を発生
    if lf_enabled:
        if data_name == "train":
            k = random.randint(0, num_failure)
        else:
            k = num_failure
        failed_links = random.sample(topo_links, k)
        for link in failed_links:
            u = link["u"]
            v = link["v"]
            key = (min(u, v), max(u, v))
            links[key].max_capacity = -1
            failed_links_list.append(key)

    # 障害リンクを通るフローの解放と収集
    if failed_links_list:
        for key in failed_links_list:
            if not queue_checker:
                break
            for i in reversed(range(len(traffic_list))):
                if not(queue_checker):
                    break
                src, dst, bw, link_path_list = traffic_list[i]
                for link in link_path_list:
                    if num_traffic <= len(queue):
                        queue_checker = False
                        break
                    # debug #print(f'link[0]: {link[0]}, key: {key}')
                    if link[0] == key: # 当該リンクを経路に含むトラフィックか判定
                        queue.append((src, dst, bw))
                        for l in link_path_list:
                            l[1].release(bw)
                        del traffic_list[i]
                        break

    traffic_list.sort(key=lambda x: x[2], reverse=False) # 要求帯域幅で昇順ソート

    while len(queue) < num_traffic:
        links_list = []
        for key in links:
            if not key in failed_links_list:
                links_list.append((key, links[key].used)) # links_listに障害リンク以外のリンク利用量をappend

        # links_list = np.array(links_list)
        links_list.sort(key=lambda x: x[1], reverse=True) # 全リンク利用量で降順ソート
        # print(links_list)

        # idx = np.argmax(links_list[:, 1])
        """
        idx = max(range(len(links_list)), key=lambda i: links_list[i][1]) # 一番利用量が多いリンクのインデックス
        key, link_all_used = links_list[idx]        
        """
        max_used = max([x[1] for x in links_list])
        max_indices = [i for i, x in enumerate(links_list) if x[1] == max_used]
        idx = random.choice(max_indices)
        key, link_all_used = links_list[idx]   
        append_checker = False
        # debug #print(f'links_list: {links_list}, links_list[idx]: {links_list[idx]}')
        for i in reversed(range(len(traffic_list))):
            if append_checker:
                break
            src, dst, bw, link_path_list = traffic_list[i]
            for link in link_path_list:
                # print(f'link[0]: {link[0]}, key: {key}')
                if link[0] == key: # 当該リンクを経路に含むトラフィックか判定
                    # print("追加")
                    queue.append((src, dst, bw))
                    for l in link_path_list:
                        l[1].release(bw)
                    del traffic_list[i]
                    append_checker = True
                    break

    """
    for key, link_all_used in links_list:
        if not queue_checker:
            break
        for i, (src, dst, bw, link_path_list) in enumerate(traffic_list):
            if not queue_checker:
                break
            for link in link_path_list:
                if num_traffic <= len(queue):
                    queue_checker = False
                    break
                if link == key: # 当該リンクを経路に含むトラフィックか判定
                    queue.append((src, dst, bw))
                    for l in link_path_list:
                        l.release(bw)
                    del traffic_list[i]
                    break    
    """


    for key in links:
        if not key in failed_links_list: # 正常リンクの場合
            if links[key].max_capacity < links[key].used:
                no_continue = True
        elif links[key].used != 0: # 障害リンクかつそのリンクの利用量が0でない場合
            no_continue = True
    if no_continue:
        continue

    links_list = [] # リセット

#######################################################################################
    # modality2
    """ 隣接行列
    if no_path_checker and fukusou_sub_counter == 0:
        # 全ノード間トラフィックの半分のパスを張った状態のリンク情報を保存
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                key = (min(i, j), max(i, j))
                if i == j:
                    links_list += [-1]
                elif key in links:
                    link = links[key]
                    if link.max_capacity == -1: # 障害リンクの場合
                        links_list += [-1]
                    else: # リンクがある場合
                        links_used_rate = links[key].used /links[key].max_capacity
                        links_list += [links_used_rate]
                else: # そもそもリンクがない場合
                    links_list += [-2]
    """



    """ リンク数のみ

    """
    if no_path_checker and fukusou_sub_counter == 0:
        for key in links:
            if links[key].max_capacity == -1: # 障害リンクの場合
                links_list += [-1]
            else: # リンクがある場合
                links_used_rate = links[key].used /links[key].max_capacity
                links_list += [links_used_rate]
    """
    else:
        continue    
    """

#######################################################################################

        
    traffic_list = []   # リセット
    while queue:
        traffic_list.append(queue.popleft())

    # bw 降順でソート
    traffic_list.sort(key=lambda x: x[2], reverse=True)

    # queue に追加
    for src, dst, bw in traffic_list:
        queue.append((src, dst, bw))


    ## 残りのトラフィック処理
    while queue:
        src, dst, bw = queue.popleft()  # キューから [src, dst, bw] のリストを取り出し，それぞれの変数に代入
        start_time = time.perf_counter()
        graph = generate_graph(links)   # graph はディクショナリ
        path = dijkstra(graph, src, dst)
        end_time = time.perf_counter()
        current_sample_dijkstra_duration += (end_time -start_time)

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
            if link[1].try_allocate(bw):
                allocated_links.append(link[1])
                
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
            continue

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
    """
    if fukusou_sub_counter != 0:
        # 全リンクを解放
        for link in links.values():
            link.used = 0
        continue    
    """


    if no_path_checker:
        # Dijkstra法で再計算した時のネットワーク状態
        for link in links.values():
            if not link.max_capacity == -1:
                links_djk_list += [link.used]

            # debug #print(link.used)

        # debug #print(links_djk_list)

        if fukusou_sub_counter == 0:
            links_log.append(links_list)
            data_log.append(data_list)
            data_one_hot_log.append(data_one_hot_list)
            target_log.append(target_list)
            target_one_hot_log.append(target_one_hot_list)
                
            # max_load_link_list += [max(links_ex_list)]

            links_shortest_log.append(links_shortest_list)
            mlu_shortest_log.append(max(links_shortest_list))
            std_shortest_log.append(statistics.stdev(links_shortest_list))
            cv_shortest_log.append(statistics.stdev(links_shortest_list) /statistics.mean(links_shortest_list))

            # 最大の障害リンク数を記録
            if lf_enabled:
                if max_failure < k:
                    # debug #print(f"max_failure < k ⇒ {max_failure} < {k}")
                    max_failure = k

            links_djk_log.append(links_djk_list)
            mlu_djk_log.append(max(links_djk_list))
            std_djk_log.append(statistics.stdev(links_djk_list))
            cv_djk_log.append(statistics.stdev(links_djk_list) /statistics.mean(links_djk_list))

            dijkstra_total_times_per_sample.append(current_sample_dijkstra_duration)
            
        elif 0 < fukusou_sub_counter:
            fukusou_counter += 1
            fukusou_all_counter += fukusou_sub_counter
    else:
        no_path_counter += 1
        no_path_all_counter += no_path_sub_counter

    # 全リンクを解放
    for link in links.values():
        link.used = 0

    # debug #print()

"""
max_link = max(max_load_link_list)
min_link = min(max_load_link_list)
average_link = statistics.mean(max_load_link_list)
std_dev_link = statistics.stdev(max_load_link_list)
std_err_link = std_dev_link / (len(max_load_link_list) ** 0.5)
median_link = statistics.median(max_load_link_list)
"""


# list -> Numpy
links_shortest_log = np.array(links_shortest_log)
mlu_shortest_log = np.array(mlu_shortest_log)
std_shortest_log = np.array(std_shortest_log)
cv_shortest_log = np.array(cv_shortest_log)
if data_name == "test":
    links_djk_log = np.array(links_djk_log)
mlu_djk_log = np.array(mlu_djk_log)
std_djk_log = np.array(std_djk_log)
cv_djk_log = np.array(cv_djk_log)


mlu_shortest_avg = np.mean(mlu_shortest_log)
mlu_shortest_std = np.std(mlu_shortest_log)
mlu_shortest_min = np.min(mlu_shortest_log)
mlu_shortest_max = np.max(mlu_shortest_log)
std_shortest_avg = np.mean(std_shortest_log)
std_shortest_std = np.std(std_shortest_log)
cv_shortest_avg = np.mean(cv_shortest_log)
cv_shortest_std = np.std(cv_shortest_log)

mlu_djk_avg = np.mean(mlu_djk_log)
mlu_djk_std = np.std(mlu_djk_log)
mlu_djk_min = np.min(mlu_djk_log)
mlu_djk_max = np.max(mlu_djk_log)
std_djk_avg = np.mean(std_djk_log)
std_djk_std = np.std(std_djk_log)
cv_djk_avg = np.mean(cv_djk_log)
cv_djk_std = np.std(cv_djk_log)

if dijkstra_total_times_per_sample:
    avg_dijkstra_time = sum(dijkstra_total_times_per_sample) /len(dijkstra_total_times_per_sample)
    print("=== Dijkstra Computation Time ===")
    print(f'Average Dijkstra time per sample ({len(target_log)} paths): {avg_dijkstra_time:.6f} seconds')


"""
traffic_log = Bunch(
    data=np.array(data_log), 
    target=np.array(target_log)
    )
# debug #print(f'traffic_log: \n{traffic_log}')

traffic_one_hot_log = Bunch(
    data_one_hot=np.array(data_one_hot_log), 
    target_one_hot=np.array(target_one_hot_log)
)
"""

# debug #print(f'traffic_one_hot_log: \n{traffic_one_hot_log}')
print(f'{data_name}データセット収集完了\nデータセット数: {len(target_log)}\nデマンド集合ごとの輻輳件数: {fukusou_counter}\nトラフィックごとの輻輳件数: {fukusou_all_counter}\nデマンド集合ごとのno path件数: {no_path_counter}\nトラフィックごとのno path件数: {no_path_all_counter}')
# print(f'path_2={path_2}, path_3={path_3}, path_4={path_4}, path_5={path_5}, path_6={path_6}')
# debug #print(f'max_load_link_list: {max_load_link_list_ex}')
# print(f'Max: {max_link}, Min: {min_link}, Average: {average_link}, SD: {std_dev_link}, SE: {std_err_link}, Median: {median_link}')
print("===shortest===")
print(f'MLU avg: {mlu_shortest_avg}, MLU Std: {mlu_shortest_std}, MLU min: {mlu_shortest_min}, MLU max: {mlu_shortest_max}, STD avg: {std_shortest_avg}, STD Std: {std_shortest_std}, CV avg: {cv_shortest_avg}, CV Std: {cv_shortest_std}')
print("===dijkstra===")
print(f'MLU avg: {mlu_djk_avg}, MLU Std: {mlu_djk_std}, MLU min: {mlu_djk_min}, MLU max: {mlu_djk_max}, STD avg: {std_djk_avg}, STD Std: {std_djk_std}, CV avg: {cv_djk_avg}, CV Std: {cv_djk_std}\n')


data_log = np.array(data_log)
target_log = np.array(target_log)
print(f'data_log: {data_log.shape}, target_log: {target_log.shape}')