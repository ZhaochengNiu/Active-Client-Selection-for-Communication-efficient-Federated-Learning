from .client_selection import ClientSelection
import torch
import numpy as np
from itertools import product
import sys

# 这段代码定义了一个名为 GradNorm 的客户端选择类，它基于客户端模型的梯度范数来选择客户端。以下是对代码的详细解释：

# GradNorm 类提供了一种基于模型梯度范数的客户端选择策略，这种策略可以帮助选择那些对当前模型最不确定的客户端，
# 从而可能提高模型在这些客户端上的性能。这种策略特别适用于那些希望在模型训练过程中特别关注某些特定客户端的场景。

'''GradNorm Client Selection'''
class GradNorm(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)
        # GradNorm 类继承自 ClientSelection 基类，并初始化构造函数。它接受客户端总数 total 和设备 device 作为参数。

    def select(self, n, client_idxs, metric, round=0, results=None):
        # select 方法实现了基于梯度范数的选择策略。它首先从 metric 参数中获取本地模型，然后计算每个模型梯度的范数，
        # 这可以被视为模型对该客户端数据的不确定性。然后，它根据这些不确定性分数（ood_scores）选择 n 个具有最高不确定性的客户端。
        # 最后，它返回这些客户端的索引。
        local_models = metric
        confs = []
        for local_model in local_models:
            local_grad = local_model.linear_2.weight.grad.data #head.conv.weight.grad.data
            local_grad_norm = torch.sum(torch.abs(local_grad)).cpu().numpy()
            confs.append(local_grad_norm)

        ood_scores = np.array(confs).reshape(-1)
        # high uncertainty (high ood score)
        selected_client_idxs = np.argsort(ood_scores)[-n:]
        return selected_client_idxs.astype(int)


def progressBar(idx, total, bar_length=20):
    # progressBar 函数是一个用于显示进度条的辅助函数。它接受当前索引 idx、总数量 total 和进度条长度 bar_length 作为参数。
    # 函数计算完成的百分比，并在控制台上打印出一个进度条，显示当前进度。
    percent = float(idx) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r> compute similarity: [{}] {}% ({}/{})".format(arrow + spaces, int(round(percent * 100)),
                                                                       idx, total))
    sys.stdout.flush()
