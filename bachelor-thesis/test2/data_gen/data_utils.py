# data_gen/data_utils.py

def data_to_one_hot(src, dst, bw, num_nodes):   # num_nodes: ネットワークのノード数
        one_hot = []
        one_hot = [0] *num_nodes +[bw]
        one_hot[src] = -1
        one_hot[dst] = 1

        return one_hot

def target_to_one_hot(path, num_nodes):
    total_one_hot = []
    for i in range(num_nodes):
        one_hot = []
        one_hot = [0] *(num_nodes+1)
        if not path[i] == -1:
            one_hot[path[i]] = 1
        else:
            one_hot[num_nodes] = 1

        total_one_hot += one_hot

    return total_one_hot