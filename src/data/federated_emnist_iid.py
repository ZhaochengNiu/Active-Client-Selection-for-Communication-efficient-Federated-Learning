import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset

# 该代码的目的是加载并处理 Federated EMNIST 数据集，适合用于联邦学习实验。
# 数据以客户端为单位进行组织，每个客户端有自己的本地训练和测试数据集，数据分布符合 IID 假设。


class FederatedEMNISTDatasetIID:
    # 这是类 FederatedEMNISTDatasetIID 的构造函数。它接收两个参数：data_dir 表示数据集的存储路径，args 包含一些额外的配置参数。
    def __init__(self, data_dir, args):
        # self.num_classes = 62 表示数据集有 62 个类别（EMNIST 有数字和字母，共 62 类）。
        self.num_classes = 62
        # self.train_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        # self.test_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        # self.min_num_samples = 100 设定了每个客户端最少需要有 100 个样本，确保每个客户端的训练数据足够。
        self.min_num_samples = 100
        # min_num_samples = 150; num_clients = 2492
        # min_num_samples = 100; num_clients = 
        # 调用内部函数 _init_data 初始化数据集。
        self._init_data(data_dir)
        # 初始化后，计算训练集和测试集中客户端的数量并打印出来。
        # self.dataset['train']['data_sizes'] 和 self.dataset['test']['data_sizes'] 是一个字典，键是客户端的 ID，值是每个客户端的数据大小。
        self.train_num_clients = len(self.dataset['train']['data_sizes'].keys())
        self.test_num_clients = len(self.dataset['test']['data_sizes'].keys())
        print(f'#TrainClients {self.train_num_clients} #TestClients {self.test_num_clients}')

    def _init_data(self, data_dir):
        # 构建存储预处理数据集的路径 file_name，文件名是 FederatedEMNIST_preprocessed_IID.pickle。
        file_name = os.path.join(data_dir, 'FederatedEMNIST_preprocessed_IID.pickle')
        if os.path.isfile(file_name):
            # 如果文件存在，使用 pickle 读取已预处理的数据集。
            # os.path.isfile(file_name) 检查文件是否存在，pickle.load(f) 用于从文件中加载 Python 对象。
            print('> read dataset ...')
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            # 如果文件不存在，则调用 preprocess 函数处理数据集并创建新的数据。
            print('> create dataset ...')
            dataset = preprocess(data_dir, self.min_num_samples)
            # with open(file_name, 'wb') as f:
            #     pickle.dump(dataset, f)
        # 最后，将加载或新创建的数据集保存到 self.dataset 中，供后续使用。
        self.dataset = dataset


def preprocess(data_dir, min_num_samples):
    # 读取 fed_emnist_train.h5 和 fed_emnist_test.h5 文件中的训练和测试数据，h5py.File 用于读取 HDF5 格式的数据。
    train_data = h5py.File(os.path.join(data_dir, 'fed_emnist_train.h5'), 'r')
    test_data = h5py.File(os.path.join(data_dir, 'fed_emnist_test.h5'), 'r')

    # 获取所有训练集和测试集客户端的 ID，并计算客户端的数量。train_data['examples'].keys() 返回训练集中所有客户端的 ID。
    train_ids = list(train_data['examples'].keys())
    test_ids = list(test_data['examples'].keys())
    num_clients_train = len(train_ids)
    num_clients_test = len(test_ids)
    print(f'#TrainClients {num_clients_train} #TestClients {num_clients_test}')

    # local dataset
    # 初始化字典 train_data_local_dict 和 test_data_local_dict 来存储各个客户端的本地数据。
    # train_data_local_num_dict 和 test_data_local_num_dict 用来记录每个客户端的样本数量。idx 是一个全局索引，用于给每个客户端编号。
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}
    idx = 0

    # 循环遍历所有客户端，client_idx 是客户端的索引，client_id 是客户端的 ID。
    for client_idx in range(num_clients_train):
        client_id = train_ids[client_idx]

        # train
        # 从 HDF5 文件中提取当前客户端的训练数据。train_data['examples'][client_id]['pixels'][()] 获取客户端的图像数据，
        # train_y 获取对应的标签数据。
        # np.expand_dims(train_x, axis=1) 将图片数据增加一个维度，使得图片数据变成 (N, 1, 28, 28)，适合 PyTorch 的输入格式。
        train_x = np.expand_dims(train_data['examples'][client_id]['pixels'][()], axis=1)
        train_y = train_data['examples'][client_id]['label'][()]
        # 如果该客户端的样本数小于设定的最小样本数 min_num_samples，则跳过该客户端。
        if len(train_x) < min_num_samples:
            continue
        # 如果数据量足够，则对训练数据进行裁剪，仅保留前 min_num_samples 个样本。
        train_x = train_x[:min_num_samples]
        train_y = train_y[:min_num_samples]
        # 使用 TensorDataset 将客户端的图片和标签数据封装为 PyTorch 数据集，
        # 并将其存储在 train_data_local_dict 字典中，train_data_local_num_dict[idx] 记录样本数。
        local_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
        train_data_local_dict[idx] = local_data
        train_data_local_num_dict[idx] = len(train_x)

        # test
        # 同样地，处理测试数据并将其封装为 TensorDataset，存储在 test_data_local_dict 字典中。
        # 这里也会记录样本数量，如果测试数据为空，打印出客户端的索引。
        test_x = np.expand_dims(test_data['examples'][client_id]['pixels'][()], axis=1)
        test_y = test_data['examples'][client_id]['label'][()]
        local_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        test_data_local_dict[idx] = local_data
        test_data_local_num_dict[idx] = len(test_x)
        if len(test_x) == 0:
            print(client_idx)
        # 每处理一个客户端的数据，idx 自增 1，用于下一个客户端的编号。
        idx += 1
    # 处理完成后，关闭 HDF5 文件。
    train_data.close()
    test_data.close()
    # 将所有训练和测试数据以及对应的样本数量打包到 dataset 字典中，供后续使用。
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
