from copy import deepcopy
from collections import Counter
import numpy as np

from .client_selection import ClientSelection

# 这段代码定义了两个联邦学习中的客户端选择策略类：ActiveFederatedLearning 和 PowerOfChoice。这两个类都继承自 ClientSelection 基类。
# 以下是对每个类的详细解释：

# 这两个类提供了不同的客户端选择策略，ActiveFederatedLearning 基于概率和损失值选择客户端，
# 而 PowerOfChoice 则基于客户端的样本数量和损失值进行选择。
# 这些策略可以帮助联邦学习系统更有效地选择参与训练的客户端，从而提高学习效率和模型性能。

'''Active Federated Learning'''


class ActiveFederatedLearning(ClientSelection):
    def __init__(self, total, device, args):
        # ActiveFederatedLearning 类的构造函数接受客户端总数、设备和参数对象 args。
        # 它初始化了三个alpha参数，这些参数用于调整选择策略，以及一个标志 save_probs 用于决定是否保存概率信息。
        super().__init__(total, device)
        self.alpha1 = args.alpha1  # 0.75
        self.alpha2 = args.alpha2  # 0.01
        self.alpha3 = args.alpha3  # 0.1
        self.save_probs = args.save_probs

    def select(self, n, client_idxs, metric, round=0, results=None):
        # select 方法实现了一个基于概率的客户端选择策略。它首先根据 metric（例如损失值）计算每个客户端的选择概率。
        # 然后，它选择一部分客户端基于这些概率，其余的客户端则随机选择。这个方法还提供了保存选择概率的功能。
        # set sampling distribution
        values = np.exp(np.array(metric) * self.alpha2)
        # 1) select 75% of K(total) users
        num_drop = len(metric) - int(self.alpha1 * len(metric))
        drop_client_idxs = np.argsort(metric)[:num_drop]
        probs = deepcopy(values)
        probs[drop_client_idxs] = 0
        probs /= sum(probs)
        #probs = np.nan_to_num(probs, nan=max(probs))
        # 2) select 99% of m users using prob.
        num_select = int((1 - self.alpha3) * n)
        #np.random.seed(round)
        selected = np.random.choice(len(metric), num_select, p=probs, replace=False)
        # 3) select 1% of m users randomly
        not_selected = np.array(list(set(np.arange(len(metric))) - set(selected)))
        selected2 = np.random.choice(not_selected, n - num_select, replace=False)
        selected_client_idxs = np.append(selected, selected2, axis=0)
        print(f'{len(selected_client_idxs)} selected users: {selected_client_idxs}')

        if self.save_probs:
            self.save_results(metric, results, f'{round},loss,')
            self.save_results(values, results, f'{round},value,')
            self.save_results(probs, results, f'{round},prob,')
        return selected_client_idxs.astype(int)


'''Power-of-Choice'''


class PowerOfChoice(ClientSelection):
    def __init__(self, total, device, d):
        # PowerOfChoice 类的构造函数接受客户端总数、设备和一个参数 d，这个参数可能用于后续的选择策略。
        super().__init__(total, device)
        #self.d = d

    def setup(self, n_samples):
        # setup 方法用于初始化客户端的选择权重，这些权重基于每个客户端的样本数量。
        client_ids = sorted(n_samples.keys())
        n_samples = np.array([n_samples[i] for i in client_ids])
        self.weights = n_samples / np.sum(n_samples)

    def select_candidates(self, client_idxs, d):
        # select_candidates 方法从所有客户端中选择一部分作为候选集，选择的概率基于客户端的权重。
        # 1) sample the candidate client set
        weights = np.take(self.weights, client_idxs)
        candidate_clients = np.random.choice(client_idxs, d, p=weights/sum(weights), replace=False)
        return candidate_clients

    def select(self, n, client_idxs, metric, round=0, results=None):
        # select 方法选择了损失值最高的 n 个客户端。
        # 3) select highest loss clients
        selected_client_idxs = np.argsort(metric)[-n:]
        return selected_client_idxs
