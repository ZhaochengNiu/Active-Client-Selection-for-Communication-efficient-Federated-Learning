'''
Reference:
    FedML: https://github.com/FedML-AI/FedML
'''
import os
import sys

import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as T

# 这段代码定义了一个名为 FederatedCIFAR100Dataset 的类，用于加载和处理联邦CIFAR100数据集。
# 这个数据集通常用于联邦学习的研究。以下是对代码的详细解释：


class FederatedCIFAR100Dataset:
    def __init__(self, data_dir, args):
        # FederatedCIFAR100Dataset 类的构造函数接受数据目录和参数对象 args。
        # 它初始化了一些属性，包括类别数、训练客户端数、测试客户端数和批量大小。然后调用 _init_data 方法来加载数据。
        self.num_classes = 100
        self.train_num_clients = 500
        self.test_num_clients = 100
        self.batch_size = args.batch_size # local batch size for local training # 20

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')

    def _init_data(self, data_dir):
        # _init_data 方法尝试从文件中加载预处理后的数据集。如果文件不存在，则调用 preprocess 函数来预处理数据，并将结果保存到文件中。
        file_name = os.path.join(data_dir, 'FedCIFAR100_preprocessed.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = preprocess(data_dir, self.train_num_clients)
        self.dataset = dataset


def preprocess(data_dir, num_clients=None):
    # preprocess 函数读取 CIFAR100 数据集的 HDF5 文件，加载训练和测试数据，并将图像进行预处理。
    # 然后，它创建 TensorDataset 对象来存储图像数据和标签，并将它们存储在字典中。
    train_data = h5py.File(os.path.join(data_dir, 'fed_cifar100_train.h5'), 'r')
    test_data = h5py.File(os.path.join(data_dir, 'fed_cifar100_test.h5'), 'r')

    train_ids = list(train_data['examples'].keys())
    test_ids = list(test_data['examples'].keys())
    num_clients_train = len(train_ids)
    num_clients_test = len(test_ids)
    print(f'num_clients_train {num_clients_train} num_clients_test {num_clients_test}')

    # local data
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}

    # train
    for client_idx in range(num_clients_train):
        client_id = train_ids[client_idx]

        train_x = np.expand_dims(train_data['examples'][client_id]['image'][()], axis=1)
        train_y = train_data['examples'][client_id]['label'][()]

        # preprocess
        train_x = preprocess_cifar_img(torch.tensor(train_x), train=True)

        local_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
        train_data_local_dict[client_idx] = local_data
        train_data_local_num_dict[client_idx] = len(train_x)

    # test
    for client_idx in range(num_clients_test):
        client_id = test_ids[client_idx]

        test_x = np.expand_dims(test_data['examples'][client_id]['image'][()], axis=1)
        test_y = test_data['examples'][client_id]['label'][()]

        # preprocess
        test_x = preprocess_cifar_img(torch.tensor(test_x), train=False)

        local_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        test_data_local_dict[client_idx] = local_data
        test_data_local_num_dict[client_idx] = len(test_x)
        if len(test_x) == 0:
            print(client_idx)

    train_data.close()
    test_data.close()

    dataset = {}
    dataset['train'] = {
        'data_sizes': train_data_local_num_dict,
        'data': train_data_local_dict,
    }
    dataset['test'] = {
        'data_sizes': test_data_local_num_dict,
        'data': test_data_local_dict,
    }

    with open(os.path.join(data_dir, 'FedCIFAR100_preprocessed.pickle'), 'wb') as f:
        pickle.dump(dataset, f)

    return dataset


def cifar100_transform(img_mean, img_std, train = True, crop_size = (24,24)):
    """cropping, flipping, and normalizing."""
    # cifar100_transform 函数定义了 CIFAR100 数据集的转换操作，包括随机裁剪、翻转和归一化。
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ])


def preprocess_cifar_img(img, train):
    # preprocess_cifar_img 函数对 CIFAR100 图像进行预处理，包括缩放和应用定义的转换操作。
    # scale img to range [0,1] to fit ToTensor api
    img = torch.div(img, 255.0)
    transoformed_img = torch.stack(
        [cifar100_transform(i.type(torch.DoubleTensor).mean(),
                            i.type(torch.DoubleTensor).std(),
                            train)
         (i[0].permute(2,0,1)) ##
         for i in img])
    return transoformed_img
