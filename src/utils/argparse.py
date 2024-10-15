import argparse

# 这段代码是一个 Python 脚本，使用 argparse 库来解析命令行参数。
# 这个脚本定义了一系列参数，这些参数通常用于配置联邦学习（Federated Learning, FL）实验。以下是对每个参数的解释：


# 这个脚本允许用户通过命令行参数灵活配置联邦学习实验的各个方面，包括硬件设置、模型选择、优化器配置、训练参数和实验设置。
# 通过这种方式，用户可以轻松地调整实验的各个方面，以适应不同的研究需求。

ALL_METHODS = [
    'Random', 'Cluster1', 'Cluster2', 'Pow-d', 'AFL', 'DivFL', 'GradNorm'
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')
    # 指定使用的 GPU 的索引，默认为 '0'。
    parser.add_argument('--dataset', type=str, default='FederatedEMNIST', help='dataset',
                        choices=['Reddit','FederatedEMNIST','FedCIFAR100','CelebA', 'PartitionedCIFAR10', 'FederatedEMNIST_IID', 'FederatedEMNIST_nonIID'])
    # 指定使用的数据集，默认为 'FederatedEMNIST'。
    parser.add_argument('--data_dir', type=str, default='../dataset/FederatedEMNIST/', help='dataset directory')
    # 指定数据集的目录，默认为 '../dataset/FederatedEMNIST/'。
    parser.add_argument('--model', type=str, default='CNN', help='model', choices=['BLSTM','CNN','ResNet'])
    # 指定使用的模型，默认为 'CNN'。
    parser.add_argument('--method', type=str, default='Random', help='client selection',
                        choices=ALL_METHODS)
    # 指定客户端选择方法，默认为 'Random'。
    parser.add_argument('--fed_algo', type=str, default='FedAvg', help='Federated algorithm for aggregation',
                        choices=['FedAvg', 'FedAdam'])
    # 指定联邦学习中的聚合算法，默认为 'FedAvg'。
    # optimizer
    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='client optim')
    # 指定客户端优化器，默认为 'sgd'。
    parser.add_argument('--lr_local', type=float, default=0.1, help='learning rate for client optim')
    # 指定客户端优化器的学习率，默认为 0.1。
    parser.add_argument('--lr_global', type=float, default=0.001, help='learning rate for server optim')
    # 指定服务器优化器的学习率，默认为 0.001。
    parser.add_argument('--wdecay', type=float, default=0, help='weight decay for optim')
    # 指定权重衰减参数，默认为 0。
    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')
    # 指定 SGD 优化器的动量参数，默认为 0。
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for Adam')

    parser.add_argument('--alpha1', type=float, default=0.75, help='alpha1 for AFL')
    parser.add_argument('--alpha2', type=float, default=1, help='alpha2 for AFL')
    parser.add_argument('--alpha3', type=float, default=0.1, help='alpha3 for AFL')
    
    # training setting
    parser.add_argument('-E', '--num_epoch', type=int, default=1, help='number of epochs')
    # 指定每个客户端训练的轮数，默认为 1。
    parser.add_argument('-B', '--batch_size', type=int, default=64, help='batch size of each client data')
    # 指定每个客户端的批量大小，默认为 64。
    parser.add_argument('-R', '--num_round', type=int, default=2000, help='total number of rounds')
    # 指定联邦学习轮数，默认为 2000。
    parser.add_argument('-A', '--num_clients_per_round', type=int, default=10, help='number of participated clients')
    # 指定每轮参与训练的客户端数量，默认为 10。
    parser.add_argument('-K', '--total_num_clients', type=int, default=None, help='total number of clients')

    parser.add_argument('-u', '--num_updates', type=int, default=None, help='number of updates')
    parser.add_argument('-n', '--num_available', type=int, default=None, help='number of available clients at each round')
    parser.add_argument('-d', '--num_candidates', type=int, default=None, help='buffer size; d of power-of-choice')

    parser.add_argument('--loss_div_sqrt', action='store_true', default=False, help='loss_div_sqrt')
    parser.add_argument('--loss_sum', action='store_true', default=False, help='sum of losses')
    parser.add_argument('--num_gn', type=int, default=0, help='number of group normalization')

    parser.add_argument('--distance_type', type=str, default='L1', help='distance type for clustered sampling 2')
    parser.add_argument('--subset_ratio', type=float, default=None, help='subset size for DivFL')

    parser.add_argument('--dirichlet_alpha', type=float, default=0.1, help='ratio of data partition from dirichlet distribution')
    
    parser.add_argument('--min_num_samples', type=int, default=None, help='mininum number of samples')
    parser.add_argument('--schedule', type=int, nargs='+', default=[0, 5, 10, 15, 20, 30, 40, 60, 90, 140, 210, 300],
                        help='splitting points (epoch number) for multiple episodes of training')
    # parser.add_argument('--maxlen', type=int, default=400, help='maxlen for NLP dataset')

    # experiment setting
    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix random seed')
    # 是否固定随机种子以确保实验可重复。
    parser.add_argument('--seed', type=int, default=0, help='seed')
    # 随机种子的值。
    parser.add_argument('--parallel', action='store_true', default=False, help='use multi GPU')
    # 是否使用多个 GPU 进行训练。
    parser.add_argument('--use_mp', action='store_true', default=False, help='use multiprocessing')
    # 是否使用多进程进行训练。
    parser.add_argument('--nCPU', type=int, default=None, help='number of CPU cores for multiprocessing')
    # 用于多进程的 CPU 核心数。
    parser.add_argument('--save_probs', action='store_true', default=False, help='save probs')
    # 是否保存概率信息。
    parser.add_argument('--no_save_results', action='store_true', default=False, help='save results')
    # 是否保存结果。
    parser.add_argument('--test_freq', type=int, default=1, help='test all frequency')
    # 测试的频率。
    parser.add_argument('--comment', type=str, default='', help='comment')
    # 对实验的注释。
    args = parser.parse_args()
    return args