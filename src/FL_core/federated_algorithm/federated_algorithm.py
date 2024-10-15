from copy import deepcopy
from collections import OrderedDict
import torch
import numpy as np

# 这段代码定义了两个联邦学习算法类：FedAvg 和 FedAdam，它们都继承自一个基类 FederatedAlgorithm。
# 这些算法用于在联邦学习环境中聚合来自多个客户端的模型更新。以下是对代码的详细解释：

# 这些类提供了在联邦学习环境中聚合客户端模型更新的两种不同方法，FedAvg 和 FedAdam，分别对应于简单的平均和更复杂的 Adam 优化器。
# 这些算法可以帮助在保护用户隐私的同时训练共享模型。


class FederatedAlgorithm:
    # 这是一个基类，用于存储每个客户端的训练数据大小和初始化模型的参数键。
    # 它检查初始化模型是否是一个有序字典（OrderedDict），如果是，则直接获取其键；否则，它将模型转换为 CPU 并获取其状态字典的键。
    def __init__(self, train_sizes, init_model):
        self.train_sizes = train_sizes
        if type(init_model) == OrderedDict:
            self.param_keys = init_model.keys()
        else:
            self.param_keys = init_model.cpu().state_dict().keys()

    def update(self, local_models, client_indices, global_model=None):
        pass


class FedAvg(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model):
        super().__init__(train_sizes, init_model)
        # FedAvg 类继承自 FederatedAlgorithm 并初始化基类的构造函数。

    def update(self, local_models, client_indices, global_model=None):
        # update 方法实现了联邦平均（FedAvg）算法。它计算所有客户端的总训练数据量，然后对每个客户端的模型参数进行加权求和，
        # 权重是客户端的训练数据量与总训练数据量的比值。最后返回聚合后的模型参数。
        num_training_data = sum([self.train_sizes[idx] for idx in client_indices])
        update_model = OrderedDict()
        for idx in range(len(client_indices)):
            local_model = local_models[idx].cpu().state_dict()
            num_local_data = self.train_sizes[client_indices[idx]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if idx == 0:
                    update_model[k] = weight * local_model[k]
                else:
                    update_model[k] += weight * local_model[k]
        return update_model


class FedAdam(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model, args):
        # FedAdam 类继承自 FederatedAlgorithm 并初始化基类的构造函数。
        # 它还初始化了 Adam 优化器的超参数（beta1、beta2、epsilon）和全局学习率（lr_global），
        # 以及用于存储一阶和二阶矩估计的字典（m 和 v）。
        super().__init__(train_sizes, init_model)
        self.beta1 = args.beta1  # 0.9
        self.beta2 = args.beta2  # 0.999
        self.epsilon = args.epsilon  # 1e-8
        self.lr_global = args.lr_global
        self.m, self.v = OrderedDict(), OrderedDict()
        for k in self.param_keys:
            self.m[k], self.v[k] = 0., 0.

    def update(self, local_models, client_indices, global_model):
        # update 方法实现了联邦 Adam（FedAdam）算法。它首先计算所有客户端的总训练数据量，然后对每个客户端的模型参数进行加权求和。
        # 接着，它使用 Adam 优化器的更新规则来更新全局模型参数。最后返回聚合后的模型参数。
        num_training_data = sum([self.train_sizes[idx] for idx in client_indices])
        gradient_update = OrderedDict()
        for idx in range(len(local_models)):
            local_model = local_models[idx].cpu().state_dict()
            num_local_data = self.train_sizes[client_indices[idx]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if idx == 0:
                    gradient_update[k] = weight * local_model[k]
                else:
                    gradient_update[k] += weight * local_model[k]
                torch.cuda.empty_cache()

        global_model = global_model.cpu().state_dict()
        update_model = OrderedDict()
        for k in self.param_keys:
            g = gradient_update[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * torch.mul(g, g)
            m_hat = self.m[k] / (1 - self.beta1)
            v_hat = self.v[k] / (1 - self.beta2)
            update_model[k] = global_model[k] - self.lr_global * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update_model
