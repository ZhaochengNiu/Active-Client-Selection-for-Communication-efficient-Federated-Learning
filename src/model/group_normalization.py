import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
# 导入 PyTorch 的函数式接口和批量归一化模块。
"""Pytorch implementation of group normalization in https://arxiv.org/abs/1803.08494 (Following the PyTorch Style)"""

# 这段代码是 Group Normalization (GroupNorm) 的 PyTorch 实现。
# Group Normalization 是一种归一化技术，用于替代 Batch Normalization (BatchNorm)。
# 与 BatchNorm 不同，GroupNorm 不依赖于批量大小，因此适用于小批量或单样本的情况。
# GroupNorm 将通道分成多个组，并在每个组内进行归一化处理。

# 这段代码提供了 Group Normalization 的完整实现，包括函数式接口和类接口。
# GroupNorm 通过将通道分组并在组内进行归一化，解决了 BatchNorm 在小批量大小下的不稳定性问题。
# 这使得 GroupNorm 适用于各种批量大小，包括单样本的情况，如在某些在线学习或实时应用中。


def group_norm(input, group, running_mean, running_var, weight=None, bias=None,
               use_input_stats=True, momentum=0.1, eps=1e-5):
    """Applies Group Normalization for channels in the same group in each data sample in a
    batch.
    See :class:`~torch.nn.GroupNorm1d`, :class:`~torch.nn.GroupNorm2d`,
    :class:`~torch.nn.GroupNorm3d` for details.
    """
    # 定义 GroupNorm 函数，接受以下参数：
    #
    # input：输入数据。
    # group：通道分组的数量。
    # running_mean 和 running_var：运行均值和方差。
    # weight 和 bias：可学习的权重和偏置。
    # use_input_stats：是否使用输入统计量。
    # momentum：动量参数。
    # eps：数值稳定性的小值。

    # 函数内部，首先检查是否使用输入统计量，如果不使用，则需要提供运行均值和方差。
    # 然后，根据是否使用权重和偏置，对它们进行重复以匹配输入的批量大小。
    # 接着，使用一个内部函数 _instance_norm 来实现 GroupNorm 的核心逻辑。
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(input, group, running_mean=None, running_var=None, weight=None,
                       bias=None, use_input_stats=None, momentum=None, eps=None):
        # Repeat stored stats and affine transform params if necessary
        # 这个内部函数实现了实例归一化（Instance Normalization），但在这里用于实现 GroupNorm。它接受与 group_norm 相同的参数，
        # 并进行以下操作：
        #
        # 调整输入数据的形状以适应 GroupNorm 的计算。
        # 调用 PyTorch 的 F.batch_norm 函数进行归一化。
        # 将归一化后的数据恢复到原始形状。
        # 更新运行均值和方差。
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)

        # norm_shape = [1, b * c / group, group]
        # print(norm_shape)
        # Apply instance norm
        input_reshaped = input.contiguous().view(1, int(b * c / group), group, *input.size()[2:])

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight=weight, bias=bias,
            training=use_input_stats, momentum=momentum, eps=eps)

        # Reshape back
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(b, int(c / group)).mean(0, keepdim=False))
        if running_var is not None:
            running_var_orig.copy_(running_var.view(b, int(c / group)).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])

    return _instance_norm(input, group, running_mean=running_mean,
                          running_var=running_var, weight=weight, bias=bias,
                          use_input_stats=use_input_stats, momentum=momentum,
                          eps=eps)


class _GroupNorm(_BatchNorm):
    # 定义一个基类 _GroupNorm，继承自 PyTorch 的 _BatchNorm 类。这个类封装了 GroupNorm 的参数和行为，包括：
    #
    # num_groups：通道分组的数量。
    # track_running_stats：是否跟踪运行统计量。
    # running_mean 和 running_var：运行均值和方差。
    # weight 和 bias：可学习的权重和偏置。
    def __init__(self, num_features, num_groups=1, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_GroupNorm, self).__init__(int(num_features / num_groups), eps,
                                         momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input, self.num_groups, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)


class GroupNorm2d(_GroupNorm):
    r"""Applies Group Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    https://arxiv.org/pdf/1803.08494.pdf
    `Group Normalization`_ .
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        num_groups:
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # Without Learnable Parameters
        >>> m = GroupNorm2d(100, 4)
        >>> # With Learnable Parameters
        >>> m = GroupNorm2d(100, 4, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """
    # 定义 GroupNorm2d 类，用于 4D 输入数据（例如，图像数据）。这个类继承自 _GroupNorm 并添加了对 4D 输入数据的检查。
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class GroupNorm3d(_GroupNorm):
    """
        Assume the data format is (B, C, D, H, W)
    """
    # 定义 GroupNorm3d 类，用于 5D 输入数据（例如，视频数据）。这个类继承自 _GroupNorm 并添加了对 5D 输入数据的检查。
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
