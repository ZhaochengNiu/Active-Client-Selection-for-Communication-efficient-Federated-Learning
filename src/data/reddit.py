import os
import pickle

import bz2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
# os: 用于处理文件路径和文件操作。
# pickle: 用于序列化和反序列化 Python 对象。
# bz2: 用于解压 .bz2 压缩文件（如 Reddit 数据集文件）。
# numpy: 提供数组处理和数值计算功能。
# torch: PyTorch 的核心模块，支持张量操作。
# torch.nn.functional: 包含 PyTorch 中的一些函数操作（如填充操作 F.pad）。
# tqdm: 用于显示循环进度条，方便观察处理进度。

class RedditDataset:
    def __init__(self, data_dir, args):
        # num_classes = 2: Reddit 数据集有两个类别（可能是 "controversial" 和 "non-controversial" 的帖子）。
        # train_num_clients 和 test_num_clients: 分别定义训练和测试客户端数量。
        # batch_size 和 maxlen: 分别定义每个批次的数据量和输入文本的最大长度，这些参数通过 args 传递。
        # self._init_data(data_dir): 调用 _init_data 方法来初始化数据集。
        self.num_classes = 2
        #self.train_size = 124638 # messages
        #self.test_size = 15568 # messages
        self.train_num_clients = 7668 # 7527 (paper)
        self.test_num_clients = 2099
        self.batch_size = args.batch_size #128
        self.maxlen = args.maxlen #400

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')

    def _init_data(self, data_dir):
        # file_name: 指定预处理后数据集的文件路径。
        # os.path.isfile(file_name): 检查预处理的文件是否存在。
        # 如果存在，并且 batch_size 和 maxlen 都符合要求（分别是 128 和 400），则直接用 pickle 加载数据集。
        # 否则，调用 preprocess 函数对原始 Reddit 数据进行预处理。
        # self.data = data: 保存处理好的数据集。
        file_name = os.path.join(data_dir, 'Reddit_preprocessed_7668.pickle')
        if os.path.isfile(file_name) and self.batch_size == 128 and self.maxlen == 400:
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f) # user_id, num_data, text, label
        else:
            dataset = preprocess(data_dir)
        self.dataset = dataset


def preprocess(data_dir):
    # 定义一个空字典来存储用户信息和数据集，初始化用户索引。
    users, dataset = {}, {}
    user_idx = 0
    # 读取.bz2压缩文件，对每一行进行解析，获取作者信息。
    with bz2.BZ2File(data_dir+'/RC_2017-11.bz2', 'r') as f:
        for line in tqdm(f):
            line = json.loads(line.rstrip())
            user = line['author']
            # 如果用户不在字典中，添加用户并初始化数据；如果用户已存在，更新其数据。
            if user not in users.keys():
                users[user] = user_idx
                dataset[user_idx] = {
                    'num_data': 1,
                    'user_id': user,
                    'subreddit': [line['subreddit']],
                    'text': [line['body']],
                    'label': [int(line['controversiality'])]
                }
                user_idx += 1
            else:
                dataset[users[user]]['num_data'] += 1
                dataset[users[user]]['subreddit'].append(line['subreddit'])
                dataset[users[user]]['text'].append(line['body'])
                dataset[users[user]]['label'].append(line['controversiality'])
    # 打印用户数量和数据集大小，以及每个客户端的数据数量的统计信息。
    print(len(users.keys()), len(dataset.keys()))

    num_data_per_clients = [dataset[x]['num_data'] for x in dataset.keys()]
    print(min(num_data_per_clients), max(num_data_per_clients), np.mean(num_data_per_clients),
          np.median(num_data_per_clients))
    # 设置随机种子，随机选择8000个用户。
    np.random.seed(0)
    select_users_indices = np.random.randint(len(users.keys()), size=8000).tolist()
    # 对选定的用户进行进一步筛选，确保每个用户至少有100条数据。
    final_dataset = {}
    new_idx = 0
    for user_id, user_idx in tqdm(users.items()):
        # data[user_idx]['num_data'] = len(data[user_idx]['label'])
        # preprocess 1
        if user_idx in select_users_indices:
            _data = dataset[user_idx]
            # print(user_idx, _data['num_data'], [x[:5] for x in _data['text']], user_id, _data['subreddit'])
            # preprocess 2
            if _data['num_data'] <= 100:
                select_idx = []
                for idx in range(_data['num_data']):
                    # preprocess 3-4
                    if user_id != _data['subreddit'][idx] and _data['text'] != '':
                        select_idx.append(idx)
                if len(select_idx) > 0:
                    final_dataset[new_idx] = {
                        'user_id': user_id,
                        'num_data': len(select_idx),
                        'text': np.array(_data['text'])[select_idx].tolist(),
                        'label': np.array(_data['label'])[select_idx].tolist()
                    }
                    new_idx += 1
    # 打印客户端数量，初始化训练和测试数据集。
    num_clients = len(final_dataset.keys())
    print(num_clients)

    train_dataset, test_dataset = {}, {}
    # 对每个客户端的数据进行训练和测试集的划分。
    for client_idx in tqdm(range(num_clients), desc='>> Split data to clients'):
        local_data = final_dataset[client_idx]
        user_train_data_num = local_data['num_data']

        # split train, test in local data
        num_train = int(0.9 * user_train_data_num) if user_train_data_num >= 10 else user_train_data_num
        num_test = user_train_data_num - num_train if user_train_data_num >= 10 else 0

        if user_train_data_num >= 10:
            np.random.seed(client_idx)
            train_indices = np.random.choice(user_train_data_num, num_train, replace=False).tolist()
            test_indices = list(set(np.arange(user_train_data_num)) - set(train_indices))

            train_dataset[client_idx] = {'datasize': num_train,
                                         'text': np.array(local_data['text'])[train_indices].tolist(),
                                         'label': np.array(local_data['label'])[train_indices].tolist()}
            test_dataset[client_idx] = {'datasize': num_test,
                                        'text': np.array(local_data['text'])[test_indices].tolist(),
                                        'label': np.array(local_data['label'])[test_indices].tolist()}
        else:
            train_dataset[client_idx] = {'datasize': num_train, 'text': local_data['text'],
                                         'label': local_data['label']}
    # 对训练和测试数据进行批处理。
    train_data_num, test_data_num = 0, 0
    train_data_local_dict, test_data_local_dict = {}, {}
    train_data_local_num_dict, test_data_local_num_dict = {}, {}
    test_clients = test_dataset.keys()

    for client_idx in tqdm(range(len(train_dataset.keys())), desc='>> Split data to clients'):
        train_data = train_dataset[client_idx]
        training_data = _batch_data(train_data)

        train_data_local_dict[client_idx] = training_data
        train_data_local_num_dict[client_idx] = train_data['datasize']

        if client_idx in test_clients:
            test_data = test_dataset[client_idx]
            testing_data = _batch_data(test_data)
            test_data_local_dict[client_idx] = testing_data
            test_data_local_num_dict[client_idx] = test_data['datasize']
    # 保存最终的数据集，并返回。
    final_final_dataset = {}
    final_final_dataset['train'] = {
        'data_sizes': train_data_local_num_dict,
        'data': train_data_local_dict,
    }
    final_final_dataset['test'] = {
        'data_sizes': test_data_local_num_dict,
        'data': test_data_local_dict,
    }

    with open(data_dir+'/Reddit_preprocessed_7668.pickle', 'wb') as f:
        pickle.dump(final_final_dataset, f)

    return final_final_dataset


def _batch_data(data, batch_size=128, maxlen=400):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    # 这个函数用于将数据批处理。它首先随机打乱数据，然后按照指定的批次大小
    data_x = np.array(data['text'])
    data_y = np.array(data['label'])

    # randomly shuffle data
    np.random.seed(0)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = _process_x(batched_x, maxlen)
        batched_y = torch.tensor(batched_y, dtype=torch.long)
        batch_data.append((batched_x, batched_y))
    return batch_data#, maxlen_lst


def _process_x(raw_x_batch, maxlen=400):
    CHAR_VOCAB = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
    ALL_LETTERS = "".join(CHAR_VOCAB)

    x_batch = []
    for word in raw_x_batch:
        indices = torch.empty((0,), dtype=torch.long)
        for c in word:
            tmp = ALL_LETTERS.find(c)
            tmp = len(ALL_LETTERS) if tmp == -1 else tmp
            tmp = torch.tensor([tmp], dtype=torch.long)
            indices = torch.cat((indices, tmp), dim=0)
        x_batch.append(indices)

    x_batch2 = torch.empty((0, maxlen), dtype=torch.long)
    for x in x_batch:
        x = torch.unsqueeze(F.pad(x, (0, maxlen-x.size(0)), value=maxlen-1), 0)
        x_batch2 = torch.cat((x_batch2, x), dim=0)
    return x_batch2
