'''
Reference:
    https://github.com/Accenture/Labs-Federated-Learning/tree/clustered_sampling
'''
import torchvision.datasets as D
import torchvision.transforms as T
from torch.utils.data import TensorDataset
import torch
import numpy as np
import pickle
import os

# 该代码使用 Dirichlet 分布将 CIFAR-10 数据集划分给多个客户端，每个客户端的数据类别比例不同，模拟了联邦学习中数据异构性 (non-IID)。
# 支持对训练集和测试集的自定义分布和样本数。

# 这个代码的目的是使用 Dirichlet 分布将 CIFAR-10 数据集分配给多个客户端，以模拟联邦学习中的数据异构性 (non-IID)。
# 该实现特别针对 CIFAR-10 数据集，并使用 PyTorch 对数据进行处理和存储。下面逐行解释该代码：


class PartitionedCIFAR10Dataset(object):
    def __init__(self, data_dir, args):
        '''
        partitioned CIFAR10 datatset according to a Dirichlet distribution
        '''
        # 这个类将 CIFAR-10 数据集按照 Dirichlet 分布划分到多个客户端。
        # self.num_classes = 10：表示 CIFAR-10 有 10 个类别。
        # self.train_num_clients 和 self.test_num_clients：分别表示训练和测试集的客户端数量，默认值为 100。
        # 如果 args 提供了 total_num_clients，则使用该值。
        # self.balanced：指示是否对每个客户端分配相同数量的样本，默认是 False。
        # self.alpha：表示 Dirichlet 分布的参数，控制分布的异构性。
        self.num_classes = 10
        self.train_num_clients = 100 if args.total_num_clients is None else args.total_num_clients
        self.test_num_clients = 100 if args.total_num_clients is None else args.total_num_clients
        self.balanced = False
        self.alpha = args.dirichlet_alpha
        # 调用 _init_data 方法初始化数据集，并打印总客户端数量。
        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')
    
    def _init_data(self, data_dir):
        # 通过 file_name 检查是否存在预处理的 CIFAR-10 数据集。如果存在，从文件中加载数据。
        file_name = os.path.join(data_dir, 'PartitionedCIFAR10_preprocessed_.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            # 如果没有预处理的文件，使用 Dirichlet 分布生成矩阵 matrix，表示如何将不同类别的数据分配给各个客户端。
            # self.alpha 控制分布的浓度参数，size=self.train_num_clients 表示矩阵的行数（客户端数量）。
            matrix = np.random.dirichlet([self.alpha] * self.num_classes, size=self.train_num_clients)
            # 调用 partition_CIFAR_dataset 方法来分割训练和测试数据集，将数据按照 Dirichlet 分布划分给每个客户端。
            train_data = self.partition_CIFAR_dataset(data_dir, matrix, train=True)
            test_data = self.partition_CIFAR_dataset(data_dir, matrix, train=False)
            # 将处理后的训练和测试数据集保存到字典中。
            dataset = {
                'train': train_data, 
                'test' : test_data
            }
        # 将处理好的数据集保存在 self.data 属性中。
        self.dataset = dataset

    def partition_CIFAR_dataset(self, data_dir, matrix, train):
        """Partition data into `n_clients`.
        Each client i has matrix[k, i] of data of class k"""
        # 该方法根据 Dirichlet 分布矩阵将 CIFAR-10 数据集划分给多个客户端。矩阵 matrix[k, i] 表示第 i 个客户端分配了类别 k 的数据比例。

        # 数据预处理使用了 torchvision.transforms：首先将图像转换为张量 T.ToTensor()，然后归一化图像像素值，使其在 [-1, 1] 范围内。
        # 使用 torchvision.datasets.CIFAR10 下载并加载 CIFAR-10 数据集。
        transform = [
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
        dataset = D.CIFAR10(data_dir, train=train, download=True, transform=transform)
        # 根据数据集类型（训练集或测试集）选择客户端数量。
        n_clients = self.train_num_clients if train else self.test_num_clients
        # 初始化两个列表，分别用于存储每个客户端的数据（图像和标签）。
        list_clients_X = [[] for i in range(n_clients)]
        list_clients_y = [[] for i in range(n_clients)]
        # 如果 self.balanced 为 True，每个客户端分配相同数量的样本（500 个样本）。
        if self.balanced:
            n_samples = [500] * n_clients
        # 如果 self.balanced 为 False，且是训练集或测试集，根据给定的分布分配不同数量的样本。
        elif not self.balanced and train:
            n_samples = [100] * 10 + [250] * 30 + [500] * 30 + [750] * 20 + [1000] * 10
        elif not self.balanced and not train:
            n_samples = [20] * 10 + [50] * 30 + [100] * 30 + [150] * 20 + [200] * 10
        
        # custom
        # if train:
        #     n_samples = [5] * 10 + [10] * 20 + [100] * 10 + [250] * 29 + [500] * 30 + [750] * 20 + [1000] * 10
        # else:
        #     n_samples = [10] * 29 + [20] * 10 + [50] * 30 + [100] * 30 + [150] * 20 + [200] * 10
        # 遍历每个类别 k，使用 np.where 查找 CIFAR-10 数据集中属于该类别的样本索引，并将这些索引存储在 list_idx 中。
        list_idx = []
        for k in range(self.num_classes):

            idx_k = np.where(np.array(dataset.targets) == k)[0]
            list_idx += [idx_k]
        # 依次为每个客户端分配数据。根据 Dirichlet 矩阵 matrix 计算该客户端每个类别应分配的样本数量，并从相应的类别索引中随机选择这些样本。
        # clients_idx_i 存储该客户端的所有样本索引。
        for idx_client, n_sample in enumerate(n_samples):

            clients_idx_i = []
            client_samples = 0

            for k in range(self.num_classes):

                if k < self.num_classes:
                    samples_digit = int(matrix[idx_client, k] * n_sample)
                if k == self.num_classes:
                    samples_digit = n_sample - client_samples
                client_samples += samples_digit

                clients_idx_i = np.concatenate(
                    (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
                )

            clients_idx_i = clients_idx_i.astype(int)
            # 将每个客户端的样本（图像和标签）从数据集中提取出来，并存储在 list_clients_X 和 list_clients_y 中。
            for idx_sample in clients_idx_i:

                list_clients_X[idx_client] += [dataset.data[idx_sample]]
                list_clients_y[idx_client] += [dataset.targets[idx_sample]]

            list_clients_X[idx_client] = np.array(list_clients_X[idx_client])
        # 最终将每个客户端的数据转换为 PyTorch 的 TensorDataset 格式，并记录每个客户端的数据大小，返回一个包含数据和样本大小的字典。
        return {
            'data': {idx: TensorDataset(torch.Tensor(list_clients_X[idx]).permute(0, 3, 1, 2), torch.tensor(list_clients_y[idx])) for idx in range(len(list_clients_X))}, # (list_clients_X, list_clients_y),
            'data_sizes': {idx: len(list_clients_y[idx]) for idx in range(len(list_clients_X))}
        }
