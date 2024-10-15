import numpy as np

# 这段代码定义了一个名为 ClientSelection 的基类和它的一个子类 RandomSelection，这些类用于在联邦学习环境中选择参与训练的客户端。
# 以下是对代码的详细解释：

# RandomSelection 类提供了一种简单的随机选择策略，适用于没有特定选择标准的情况。
# 这种策略可以确保所有客户端都有平等的机会被选中，有助于模型训练的多样性。


class ClientSelection:
    def __init__(self, total, device):
        # ClientSelection 类的构造函数接受两个参数：total 表示客户端的总数，device 表示执行计算的设备。
        self.total = total
        self.device = device

    def select(self, n, client_idxs, metric):
        # select 方法是一个占位符，它应该在子类中被实现。这个方法应该根据 metric 选择 n 个客户端，从 client_idxs 列表中选择。
        pass

    def save_selected_clients(self, client_idxs, results):
        # save_selected_clients 方法将选中的客户端索引保存到一个文件中。
        # 它创建一个长度为 self.total 的数组，将选中的客户端索引位置设置为 1，然后将该数组以逗号分隔的格式写入 results 文件。
        tmp = np.zeros(self.total)
        tmp[client_idxs] = 1
        tmp.tofile(results, sep=',')
        results.write("\n")

    def save_results(self, arr, results, prefix=''):
        # save_results 方法将一个数值数组 arr 保存到 results 文件中，前面可以加上一个字符串前缀 prefix。
        results.write(prefix)
        np.array(arr).astype(np.float32).tofile(results, sep=',')
        results.write("\n")


'''Random Selection'''


class RandomSelection(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)
        # RandomSelection 类继承自 ClientSelection 类，并调用基类的构造函数来初始化 total 和 device 属性。

    def select(self, n, client_idxs, metric=None):
        # select 方法实现了随机选择客户端的逻辑。
        # 它使用 NumPy 的 random.choice 函数从 client_idxs 列表中随机选择 n 个不重复的客户端索引，并返回这些索引。
        # 这个方法不需要 metric 参数，因为选择是完全随机的。
        selected_client_idxs = np.random.choice(client_idxs, size=n, replace=False)
        return selected_client_idxs
