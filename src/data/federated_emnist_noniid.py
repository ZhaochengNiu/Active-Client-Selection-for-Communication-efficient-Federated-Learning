import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset

# 这个代码的目的是创建非独立同分布 (non-IID) 的联邦 EMNIST 数据集，
# 其中前 2000 个客户端的训练数据仅包含数字 (0-9)，其余客户端则包含字符，并将其标签随机替换为数字标签。
# 这样模拟了真实场景中客户端数据的异构性 (non-IID)，适合用于联邦学习实验。

class FederatedEMNISTDataset_nonIID:
    def __init__(self, data_dir, args):
        '''
        known class: digits (10)
        unknown class: characters (52) -> label noise
        '''
        # 该类的构造函数初始化了联邦学习环境下的 EMNIST 数据集，但数据集是非独立同分布 (non-IID) 的。
        # self.num_classes = 10 表示该任务只关心 10 个类别（数字 0-9），其余 52 个字符类别被视为标签噪声。
        self.num_classes = 10
        # 初始化了训练和测试集中客户端的数量，默认为 3400 个客户端。如果传入的 args.total_num_clients 不为 None，则使用给定的客户端数目。
        self.train_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        self.test_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        # 调用 _init_data 函数来初始化数据集，即加载或预处理数据。
        self._init_data(data_dir)
        # 打印客户端总数。
        print(f'Total number of users: {self.train_num_clients}')
        # 计算并打印实际参与训练和测试的客户端数量。
        # self.data['train']['data_sizes'] 和 self.data['test']['data_sizes'] 是字典，键为客户端 ID，值为客户端的样本数量。
        self.train_num_clients = len(self.dataset['train']['data_sizes'].keys())
        self.test_num_clients = len(self.dataset['test']['data_sizes'].keys())
        print(f'#TrainClients {self.train_num_clients} #TestClients {self.test_num_clients}')
        # 3383

    def _init_data(self, data_dir):
        # 构建数据集的存储路径，文件名为 FederatedEMNIST_preprocessed_nonIID.pickle。
        file_name = os.path.join(data_dir, 'FederatedEMNIST_preprocessed_nonIID.pickle')
        # 如果文件存在，使用 pickle 从文件中加载已预处理的数据集。
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            # 如果文件不存在，调用 preprocess 函数处理原始数据，生成新的数据集。
            dataset = preprocess(data_dir, self.train_num_clients)
            
            # with open(file_name, 'wb') as f:
            #     pickle.dump(data, f)
        # 最后将加载或预处理好的数据集保存在 self.data 中。
        self.dataset = dataset


def preprocess(data_dir, num_clients=None):
    # 使用 h5py 读取 fed_emnist_train.h5 和 fed_emnist_test.h5 文件，分别包含训练和测试数据。
    train_data = h5py.File(os.path.join(data_dir, 'fed_emnist_train.h5'), 'r')
    test_data = h5py.File(os.path.join(data_dir, 'fed_emnist_test.h5'), 'r')
    # 获取所有客户端的 ID 并计算客户端数量。默认情况下，客户端数量为训练集和测试集中的实际客户端数量；
    # 如果 num_clients 有设置，使用指定的客户端数量。
    train_ids = list(train_data['examples'].keys())
    test_ids = list(test_data['examples'].keys())
    num_clients_train = len(train_ids) if num_clients is None else num_clients
    num_clients_test = len(test_ids) if num_clients is None else num_clients
    print(f'num_clients_train {num_clients_train} num_clients_test {num_clients_test}')
    # 初始化字典存储各客户端的本地训练和测试数据，train_data_local_dict 和 test_data_local_dict 分别用于存储训练和测试数据，
    # train_data_local_num_dict 和 test_data_local_num_dict 记录每个客户端的数据大小
    # local data
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}
    idx = 0
    # 循环遍历每个客户端的 ID。
    for client_idx in range(num_clients_train):
        client_id = train_ids[client_idx]
        # 从 HDF5 文件中获取当前客户端的训练图像数据（train_x）和对应的标签数据（train_y），
        # 并使用 np.expand_dims 增加一个维度，使其适合 PyTorch 的输入格式 (N, 1, 28, 28)。
        # train
        train_x = np.expand_dims(train_data['examples'][client_id]['pixels'][()], axis=1)
        train_y = train_data['examples'][client_id]['label'][()]
        # 如果客户端索引小于 2000，则客户端只包含数字（0-9），所以过滤出数字标签对应的数据。
        digits_index = np.arange(len(train_y))[np.isin(train_y, range(10))]
        if client_idx < 2000:
            # client with only digits
            train_y = train_y[digits_index]
            train_x = train_x[digits_index]
        else:
            # 如果客户端索引大于等于 2000，则客户端包含字符数据（字符标签被视为标签噪声）。
            # 这些标签被替换为随机的数字标签 (0-9)，并使用 np.random.randint 生成随机标签。
            # client with only characters (but it's label noise for digits classification)
            non_digits_index = np.invert(np.isin(train_y, range(10)))
            train_y = train_y[non_digits_index]
            train_y = np.random.randint(10, size=len(train_y))
            train_x = train_x[non_digits_index]
        
        if len(train_y) == 0:
            # 如果客户端没有数据，跳过该客户端。
            continue
        
        # test
        # 同样地，处理测试数据。对于字符数据，将其标签替换为随机生成的数字标签。
        test_x = np.expand_dims(test_data['examples'][client_id]['pixels'][()], axis=1)
        test_y = test_data['examples'][client_id]['label'][()]

        non_digits_index = np.invert(np.isin(test_y, range(10)))
        test_y[non_digits_index] = np.random.randint(10, size=sum(non_digits_index))
        # 如果测试数据为空，跳过该客户端。
        if len(test_x) == 0:
            continue
        # 将客户端的训练数据封装为 PyTorch 的 TensorDataset，并存储在 train_data_local_dict 中，同时记录样本数量。
        local_train_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
        train_data_local_dict[idx] = local_train_data
        train_data_local_num_dict[idx] = len(train_x)
        # 对测试数据进行同样的处理，将其存储在 test_data_local_dict 中。
        local_test_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        test_data_local_dict[idx] = local_test_data
        test_data_local_num_dict[idx] = len(test_x)
        # 每处理完一个客户端的数据后，将索引 idx 增加 1，供下一个客户端使用。
        idx += 1
        
    # 处理完数据后，关闭 HDF5 文件。
    train_data.close()
    test_data.close()
    # 将训练和测试数据及其对应的样本大小存储在字典中，形成完整的数据集。
    dataset = {}
    dataset['train'] = {
        'data_sizes': train_data_local_num_dict,
        'data': train_data_local_dict,
    }
    dataset['test'] = {
        'data_sizes': test_data_local_num_dict,
        'data': test_data_local_dict,
    }
    # 返回处理后的数据集。
    return dataset