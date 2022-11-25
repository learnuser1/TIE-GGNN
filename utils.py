import numpy as np
import torch
from torch.utils.data import Dataset


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(inputData, num_node,train_len=None):
    # inputdata 全部序列，[[783, 56, 227, 784, 227, 227, 56, 227, 785, 786], [783, 56, 227, 784, 227, 227, 56, 227, 785]...]

    # 添加虚拟节点
    # len_data = []
    # for i in range(len(inputData)):
    #     inputData[i].append(num_node-1)
    #     len_data.append(len(inputData[i]))
    len_data = [len(nowData) for nowData in inputData]
    # train_len = 39
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    # print('handle_data max_len',max_len,len(inputData))
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_pois1 = [list(upois) + [0] * (max_len - le) if le < max_len else list(upois[-max_len:])
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    # print(inputData[1000:1010])
    # print(len_data[1000:1010])
    # print(len(us_pois[0]),us_pois[1000:1010]) #序列的倒数排列
    # print(len(us_msks[0]),us_msks[1000:1010]) # [1,1,1,0,...0]表示序列长度为3
    # aa
    return us_pois, us_msks, max_len,us_pois1


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity


class Data(Dataset):
    def __init__(self, data, train_len=None):
        inputs, mask, max_len ,inputs1= handle_data(data[0],train_len)
        self.inputs = np.asarray(inputs)
        self.inputs1 = np.asarray(inputs1)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len

    def __getitem__(self, index):
        u_input, mask, target,u_input1 = self.inputs[index], self.mask[index], self.targets[index] ,self.inputs1[index] #第 index 个序列的矩阵即矩阵的第index行
        # print('getitem')
        # print(u_input)
        # print('1')
        # print(u_input1)
        max_n_node = self.max_len
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        adj = np.zeros((max_n_node, max_n_node))
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            u1 = np.where(node == u_input1[i])[0][0]
            adj[u][u] = 1
            # adj1[u1][u1] = 1

            # adj[w][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            v1 = np.where(node == u_input1[i + 1])[0][0]
            # adj1[v1][v1] = 1
            # u_A[u1][u1] = 1
            u_A[u1][v1] = 1
            # u_A[v1][v1] = 1
            # adj[u][v] = 1
            # adj[v][v] = 1
            # if u == v or adj[u][v] == 4:
            #     continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
                # l += 1
            else:
                # l += 1
                adj[u][v] = 2
                adj[v][u] = 3
                # adj[u][v] = 1
                # adj[v][u] = 1
        # gru adj
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()

        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        alias_inputs1 = [np.where(node == i)[0][0] for i in u_input1]
        # print(alias_inputs)
        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input),torch.tensor(u_A),torch.tensor(alias_inputs1)]

    def __len__(self):
        return self.length
