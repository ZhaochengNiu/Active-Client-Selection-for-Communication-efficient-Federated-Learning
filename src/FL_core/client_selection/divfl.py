'''
Diverse Client Selection For Federated Learning via Submodular Maximization

Reference:
    https://openreview.net/pdf?id=nwKXyFvaUm
'''
from .client_selection import ClientSelection
import numpy as np
import torch
from tqdm import tqdm
from itertools import product
import sys

# 这段代码定义了一个名为 DivFL 的类，它实现了一种基于子模函数最大化的多样化客户端选择策略，用于联邦学习环境。以下是对代码的详细解释：

# 这个 DivFL 类提供了一种基于梯度信息的多样化客户端选择策略，旨在提高联邦学习中的学习效率和公平性。
# 通过选择具有代表性梯度信息的客户端，该方法有助于减少通信成本，同时提高模型的泛化能力。


'''Diverse Client Selection'''
class DivFL(ClientSelection):
    def __init__(self, total, device, subset_ratio=0.1):
        super().__init__(total, device)
        '''
        Args:
            subset ratio: 0.1
        '''
        if subset_ratio is None:
            sys.exit('Please set the hyperparameter: subset ratio! =)')
        self.subset_ratio = subset_ratio
        # DivFL 类继承自 ClientSelection 基类，并初始化构造函数。它接受客户端总数、设备和子集比例作为参数。

    def init(self, global_m, l=None):
        # init 方法用于初始化全局模型。
        self.prev_global_m = global_m

    def select(self, n, client_idxs, metric, round=0, results=None):
        # select 方法实现了客户端选择逻辑。它首先获取客户端的梯度，然后计算梯度之间的相似性矩阵，最后使用随机贪婪算法选择客户端。
        # pre-select
        '''
        ---
        Args
            metric: local_gradients
        '''
        # get clients' gradients
        local_grads = self.get_gradients(self.prev_global_m, metric)
        # get clients' dissimilarity matrix
        self.norm_diff = self.get_matrix_similarity_from_grads(local_grads)
        # stochastic greedy
        selected_clients = self.stochastic_greedy(len(client_idxs), n)
        return list(selected_clients)

    def get_gradients(self, global_m, local_models):
        """
        return the `representative gradient` formed by the difference
        between the local work and the sent global model
        """
        # get_gradients 方法计算本地模型与全局模型之间的梯度差异。
        local_model_params = []
        for model in local_models:
            local_model_params += [[tens.detach().to(self.device) for tens in list(model.parameters())]] #.numpy()

        global_model_params = [tens.detach().to(self.device) for tens in list(global_m.parameters())]

        local_model_grads = []
        for local_params in local_model_params:
            local_model_grads += [[local_weights - global_weights
                                   for local_weights, global_weights in
                                   zip(local_params, global_model_params)]]

        return local_model_grads

    def get_matrix_similarity_from_grads(self, local_model_grads):
        """
        return the similarity matrix where the distance chosen to
        compare two clients is set with `distance_type`
        """
        # get_matrix_similarity_from_grads 方法计算梯度之间的相似性矩阵。
        n_clients = len(local_model_grads)
        metric_matrix = torch.zeros((n_clients, n_clients), device=self.device)
        for i, j in tqdm(product(range(n_clients), range(n_clients)), desc='>> similarity', total=n_clients**2, ncols=80):
            grad_1, grad_2 = local_model_grads[i], local_model_grads[j]
            for g_1, g_2 in zip(grad_1, grad_2):
                metric_matrix[i, j] += torch.sum(torch.square(g_1 - g_2))

        return metric_matrix

    def stochastic_greedy(self, num_total_clients, num_select_clients):
        # stochastic_greedy 方法实现了随机贪婪算法，用于选择客户端。
        # num_clients is the target number of selected clients each round,
        # subsample is a parameter for the stochastic greedy alg
        # initialize the ground set and the selected set
        V_set = set(range(num_total_clients))
        SUi = set()

        m = max(num_select_clients, int(self.subset_ratio * num_total_clients))
        for ni in range(num_select_clients):
            if m < len(V_set):
                R_set = np.random.choice(list(V_set), m, replace=False)
            else:
                R_set = list(V_set)
            if ni == 0:
                marg_util = self.norm_diff[:, R_set].sum(0)
                i = marg_util.argmin()
                client_min = self.norm_diff[:, R_set[i]]
            else:
                client_min_R = torch.minimum(client_min[:, None], self.norm_diff[:, R_set])
                marg_util = client_min_R.sum(0)
                i = marg_util.argmin()
                client_min = client_min_R[:, i]
            SUi.add(R_set[i])
            V_set.remove(R_set[i])
        return SUi
