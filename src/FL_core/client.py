from .trainer import Trainer
# 这行代码从当前包中导入了 Trainer 类，这个类可能包含了训练模型所需的方法。
import numpy as np
import torch.nn.functional as F
import torch
from copy import deepcopy
# 这些行导入了所需的Python库：numpy 用于数学运算，torch.nn.functional 和 torch 用于PyTorch深度学习框架，deepcopy 用于创建对象的深拷贝。

# 这段代码定义了一个名为 Client 的类，它代表联邦学习（Federated Learning, FL）中的一个客户端。下面是对代码的逐行解释：

# 这个 Client 类是联邦学习框架中的一个基本组件，它封装了客户端的数据、模型训练和测试过程。通过这种方式，联邦学习算法可以在多个客户端之间协调，以训练一个全局模型，同时保持数据的隐私性。


class Client(object):
    def __init__(self, client_idx, nTrain, local_train_data, local_test_data, model, args):
        """
        A client
        ---
        Args
            client_idx: index of the client
            nTrain: number of train data of the client
            local_train_data: train data of the client
            local_test_data: test data of the client
            model: given model for the client
            args: arguments for overall FL training
        """
        # 定义了 Client 类的构造函数，它接受以下参数：
        #
        # client_idx：客户端的索引。
        # nTrain：训练数据集的大小。
        # local_train_data：客户端的训练数据。
        # local_test_data：客户端的测试数据。
        # model：给定的模型。
        # args：整体FL训练的参数。
        self.client_idx = client_idx
        self.test_data = local_test_data
        self.device = args.device
        self.trainer = Trainer(model, args)
        # 在构造函数中，初始化客户端的属性，包括客户端索引、测试数据、设备（CPU或GPU）、以及一个 Trainer 实例。
        self.num_epoch = args.num_epoch  # E: number of local epoch
        self.nTrain = nTrain
        self.loss_div_sqrt = args.loss_div_sqrt
        self.loss_sum = args.loss_sum
        # 初始化客户端的本地训练周期数、训练数据集大小、损失函数的选项（是否除以数据量的平方根或总和）。
        self.labeled_indices = [*range(nTrain)]
        self.labeled_data = local_train_data  # train_data
        # 初始化已标记数据的索引和数据。

    def train(self, global_model):
        """
        train each client
        ---
        Args
            global_model: given current global model
        Return
            result = model, loss, acc
        """
        # 定义了 train 方法，用于训练客户端的模型。
        # SET MODEL
        self.trainer.set_model(global_model)
        # 在训练之前，将全局模型设置为本地训练器的模型。
        # TRAIN
        if self.num_epoch == 0:  # no SGD updates
            result = self.trainer.train_E0(self.labeled_data)
        else:
            result = self.trainer.train(self.labeled_data)
        # 根据本地训练周期数进行训练。如果周期数为0，则不进行SGD更新，否则执行正常的训练过程。
        #result['model'] = self.trainer.get_model()

        # total loss / sqrt (# of local data)
        if self.loss_div_sqrt:  # total loss / sqrt (# of local data)
            result['metric'] *= np.sqrt(len(self.labeled_data))  # loss * n_k / np.sqrt(n_k)
        elif self.loss_sum:
            result['metric'] *= len(self.labeled_data)  # total loss
        # 根据损失函数的选项调整结果中的度量（例如，损失值）。
        return result
        # 返回训练的结果，通常包括模型、损失和准确率。

    def test(self, model, test_on_training_data=False):
        # TEST
        # 定义了 test 方法，用于测试模型。
        if test_on_training_data:
            # test on training data
            result = self.trainer.test(model, self.labeled_data)
        else:
            # test on test data
            result = self.trainer.test(model, self.test_data)
        # 根据参数决定是在训练数据还是测试数据上进行测试。
        return result
        # 返回测试的结果。

    def get_client_idx(self):
        # 定义了 get_client_idx 方法，用于获取客户端的索引。
        return self.client_idx