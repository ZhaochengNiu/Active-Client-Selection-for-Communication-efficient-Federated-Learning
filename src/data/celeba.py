from PIL import Image
import os
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch import Tensor
import torch
import numpy as np
import json
from collections import defaultdict

# 这段代码定义了一个名为 CelebADataset 的类，用于加载和处理 CelebA 数据集，该数据集常用于联邦学习的研究。
# 此外，还提供了一些辅助函数和类来支持数据的预处理和加载。以下是对代码的详细解释：

# 这些代码提供了一种方法来加载和处理 CelebA 数据集，使其适用于联邦学习的研究。
# 通过预处理数据并将其存储为 TensorDataset 对象，研究人员可以轻松地在 PyTorch 中使用这些数据进行模型训练和测试。


class CelebADataset(object):
    def __init__(self, data_dir, args):
        # CelebADataset 类的构造函数接受数据目录和参数对象 args。它初始化了一些属性，包括类别数、最小样本数、最大客户端数和图像尺寸。
        # 然后调用 _init_data 方法来加载数据，并打印训练和测试用户的总数。
        self.num_classes = 2
        self.min_num_samples = args.min_num_samples
        self.max_num_clients = args.total_num_clients
        self.img_size = 84

        self._init_data(data_dir)
        print(f'Total number of users: train {self.train_num_clients} test {self.test_num_clients}')

    def _init_data(self, data_dir):
        # _init_data 方法尝试从文件中加载预处理后的数据集。
        # 如果文件不存在，则调用 preprocess_online_read 函数来预处理数据，并将结果保存到文件中。
        # file_name = os.path.join(data_dir, f'CelebA_preprocessed.pickle')
        file_name = os.path.join(data_dir, f'CelebA.pickle')
        if os.path.isfile(file_name):
            print('> read data ...')
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            # dataset = preprocess(data_dir, self.img_size)
            dataset = preprocess_online_read(data_dir, self.img_size)
            with open(file_name, 'wb') as f:
                pickle.dump(dataset, f)
        
        self.dataset = dataset

        self.train_num_clients = len(dataset["train"]["data_sizes"])
        self.test_num_clients = len(dataset["test"]["data_sizes"]) 


def preprocess(data_dir, img_size=84):
    # preprocess 函数读取 CelebA 数据集的目录，加载训练和测试数据，并将图像调整为指定的尺寸。
    # 然后，它创建 TensorDataset 对象来存储图像数据和标签，并将它们存储在字典中。
    img_dir = os.path.join(data_dir, 'raw/img_align_celeba')

    train_clients, train_groups, train_data = read_dir(os.path.join(data_dir, 'train'))
    test_clients, test_groups, test_data = read_dir(os.path.join(data_dir, 'test'))

    assert train_clients == test_clients
    assert train_groups == test_groups

    clients = sorted(map(int, train_clients))

    trainset_data, trainset_datasize = {}, {}
    testset_data, testset_datasize = {}, {}

    for idx in tqdm(range(len(clients)), desc='create dataset'):
        client_id = str(clients[idx])
        # train data
        train_x = [load_image(i, img_dir, img_size) for i in train_data[client_id]['x']]
        train_y = list(map(int, train_data[client_id]['y']))
        trainset_data[idx] = TensorDataset(Tensor(train_x), Tensor(train_y))
        trainset_datasize[idx] = len(train_y)

        # test data
        test_x = [load_image(i, img_dir, img_size) for i in test_data[client_id]['x']]
        test_y = list(map(int, test_data[client_id]['y']))
        testset_data[idx] = TensorDataset(Tensor(test_x), Tensor(test_y))
        testset_datasize[idx] = len(test_y)

    dataset = {
        'train': {'data': trainset_data, 'data_sizes': trainset_datasize}, 
        'test': {'data': testset_data, 'data_sizes': testset_datasize}
    }
    return dataset
        

def read_dir(data_dir):
    # read_dir 函数读取目录中的 JSON 文件，提取用户信息和数据，并将它们存储在字典中。
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def load_image(img_name, img_dir, img_size):
    # load_image 函数加载图像文件，调整其大小，并将其转换为 PyTorch 张量。
    img = Image.open(os.path.join(img_dir, img_name))
    img = img.resize((img_size, img_size)).convert('RGB')
    return np.array(img).transpose(2,0,1)


class CelebA_ClientData(object):
    # CelebA_ClientData 类表示单个客户端的数据集。
    # 它提供了 __getitem__ 方法来访问数据集中的图像和标签，以及 __len__ 方法来获取数据集的大小。
    def __init__(self, img_dir, img_size, dataset):
        self.img_dir = img_dir
        self.img_size = img_size
        self.dataset = dataset
        self.num_data = len(self.dataset['y'])

    def __getitem__(self, index):
        img_name = self.dataset['x'][index]
        data = self.load_image(img_name)
        target = torch.tensor(self.dataset['y'][index], dtype=torch.long)
        return data, target

    def __len__(self):
        return self.num_data
    
    def load_image(self, img_name):
        img = Image.open(os.path.join(self.img_dir, img_name))
        img = img.resize((self.img_size, self.img_size)).convert('RGB')
        img = torch.tensor(np.array(img).transpose(2,0,1)).float()
        return img


def preprocess_online_read(data_dir, img_size=84):
    # preprocess_online_read 函数类似于 preprocess 函数，但它使用 CelebA_ClientData 类来加载和处理每个客户端的数据。
    img_dir = os.path.join(data_dir, 'raw/img_align_celeba')

    train_clients, train_groups, train_data = read_dir(os.path.join(data_dir, 'train'))
    test_clients, test_groups, test_data = read_dir(os.path.join(data_dir, 'test'))

    assert train_clients == test_clients
    assert train_groups == test_groups

    clients = sorted(map(int, train_clients))

    trainset_data, trainset_datasize = {}, {}
    testset_data, testset_datasize = {}, {}

    for idx in range(len(clients)):
        client_id = str(clients[idx])
        # train data
        client_data = CelebA_ClientData(img_dir, img_size, train_data[client_id])
        trainset_data[idx] = client_data
        trainset_datasize[idx] = client_data.num_data

        # test data
        client_data = CelebA_ClientData(img_dir, img_size, test_data[client_id])
        testset_data[idx] = client_data
        testset_datasize[idx] = client_data.num_data

    dataset = {
        'train': {'data': trainset_data, 'data_sizes': trainset_datasize}, 
        'test': {'data': testset_data, 'data_sizes': testset_datasize}
    }
    return dataset