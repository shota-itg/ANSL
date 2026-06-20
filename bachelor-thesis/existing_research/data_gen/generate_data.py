# data_gen/genearate_data.py

import time
from collections import deque
import random
import statistics

from utils.config_loader import load_config, load_runtime, save_runtime

from data_gen.network_core import build_network, generate_graph
from data_gen.dijkstra import dijkstra, get_links_from_path
from data_gen.data_utils import data_to_one_hot, target_to_one_hot
from data_gen.save_data import save_data

config = load_config()
nodes = config["topology"]["nodes"]
topo_links = config["topology"]["links"]
bandwidth_options = config["bandwidth_options"]


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
            num_train_data = int(input("How many data?: "))   # ほしいデータセット数を設定
            break
        except ValueError:
            print("\nError: 整数を入力してください。")
    num_data = num_train_data
    runtime_cfg["data"]["num_train_data"] = num_train_data
    ### lf_enabled = config["train"]["train_lf_enabled"]
else:
    while True:
        try:
            num_test_data = int(input("How many data?: "))
            break
        except ValueError:
            print("\nError: 整数を入力してください。")
    num_data = num_test_data
    runtime_cfg["data"]["num_test_data"] = num_test_data
    ### lf_enabled = config["test"]["test_lf_enabled"]
save_runtime(runtime_cfg)


queue = deque()

data_log, target_log, traffic_log = [], [], []
data_one_hot_log, target_one_hot_log, traffic_one_hot_log = [], [], []

fukusou_counter, fukusou_all_counter = 0, 0
no_path_counter, no_path_all_counter = 0, 0

max_load_link = 0
max_load_link_list = []

dijkstra_total_times_per_sample = [] # dijkstraの計測時間のリスト


while len(target_log) < num_data:
    links = build_network(config)

    data_list, links_list, target_list, traffic_list = [], [], [], []
    data_one_hot_list, target_one_hot_list = [], []
    links_ex_list = []
    release_threads = []

    fukusou_sub_counter, no_path_sub_counter = 0, 0
    no_path_checker = True

    current_sample_dijkstra_duration = 0.0 # トラフィックデマンド集合ごとの計測用変数を初期化

    # 全ノード間トラフィック（1つのトラフィックデマンド集合）を生成
    for src in nodes:
        dsts = [n for n in nodes if n != src]
        for dst in dsts:
            bw = random.choice(bandwidth_options)
            traffic_list.append((src, dst, bw))

    random.shuffle(traffic_list)

    # bw 降順でソート
    traffic_list.sort(key=lambda x: x[2], reverse=True)

    # queue に追加
    for src, dst, bw in traffic_list:
        queue.append((src, dst, bw))

    """
    if lf_enabled:
        k = random.randint(0, num_failure)
        failed_links = random.sample(topo_links, k)
        for link in failed_links:
            u = link["u"]
            v = link["v"]
            key = (min(u, v), max(u, v))
            links[key].max_capacity = -1    
    """


    ## トラフィック処理
    ### for _ in range(len(traffic_list) -num_traffic):
    start_time = time.perf_counter()
    for _ in range(len(traffic_list)):
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
    end_time = time.perf_counter()

    current_sample_dijkstra_duration += (end_time -start_time)  


    if no_path_checker:
        ## 最大負荷リンクにおける使用帯域幅の詳細値 [Mbps]
        for link in links.values():
            links_ex_list += [link.used]

            # debug #print(link.used)

        # debug #print(links_ex_list)

        if fukusou_sub_counter == 0:
            ### djk_traffic_log.append(djk_traffic_list)
            ### links_log.append(links_list)
            data_log.append(data_list)
            data_one_hot_log.append(data_one_hot_list)
            target_log.append(target_list)
            target_one_hot_log.append(target_one_hot_list)
                
            max_load_link_list += [max(links_ex_list)]

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

max_link = max(max_load_link_list)
min_link = min(max_load_link_list)
average_link = statistics.mean(max_load_link_list)
std_dev_link = statistics.stdev(max_load_link_list)
std_err_link = std_dev_link / (len(max_load_link_list) ** 0.5)
median_link = statistics.median(max_load_link_list)

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
print(f'Max: {max_link}, Min: {min_link}, Average: {average_link}, SD: {std_dev_link}, SE: {std_err_link}, Median: {median_link}')

# save_data(lf_enabled, data_name, djk_traffic_log, data_log, links_log, target_log, data_one_hot_log, target_one_hot_log)
save_data(data_name, data_log, target_log, data_one_hot_log, target_one_hot_log)