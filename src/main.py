'''
Client Selection for Federated Learning
这是一段Python代码的注释，说明了该代码是用于联邦学习中的客户端选择。
'''

# 导入Python标准库中的模块，用于操作操作系统功能、系统参数和时间。
import os
os.environ["WANDB_API_KEY"] = "f08348e3235c9ce05b0eb194f5d885c74d48c0e7"
import sys
import time
import numpy as np

# 设置一个标志变量，假设wandb模块是可用的
AVAILABLE_WANDB = True
try:
    # 尝试导入wandb模块。
    import wandb
except ModuleNotFoundError:
    # 如果模块未找到，则设置标志变量为False。
    AVAILABLE_WANDB = False

# 导入PyTorch库，这是一个流行的深度学习框架，以及 random 模块，用于生成随机数。
import torch
import random

# 从当前包中导入数据集、模型、服务器、客户端选择和联邦算法相关的模块。
from src.data import *
from src.model import *
from src.FL_core.server import Server
from src.FL_core.client_selection import *
from src.FL_core.federated_algorithm import *
from src.utils import utils
from src.utils.argparse import get_args


# 定义一个函数load_data，用于根据参数args加载不同的数据集
def load_data(args):
    # 根据args.dataset的值，返回不同的数据集对象。
    if args.dataset == 'Reddit':
        return RedditDataset(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST':
        return FederatedEMNISTDataset(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST_IID':
        return FederatedEMNISTDatasetIID(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST_nonIID':
        return FederatedEMNISTDataset_nonIID(args.data_dir, args)
    elif args.dataset == 'FedCIFAR100':
        return FederatedCIFAR100Dataset(args.data_dir, args)
    elif args.dataset == 'CelebA':
        return CelebADataset(args.data_dir, args)
    elif args.dataset == 'PartitionedCIFAR10':
        return PartitionedCIFAR10Dataset(args.data_dir, args)


# 定义一个函数create_model，用于根据参数args创建不同的模型。
def create_model(args):
    # 根据args.dataset和args.model的值，创建并返回不同的模型对象。
    if args.dataset == 'Reddit' and args.model == 'BLSTM':
        model = BLSTM(vocab_size=args.maxlen, num_classes=args.num_classes)
    elif args.dataset == 'FederatedEMNIST_nonIID' and args.model == 'CNN':
        model = CNN_DropOut(True)
    elif args.dataset == 'FederatedEMNIST_nonIID' and args.model == 'CNN':
        model = CNN_DropOut(True)
    elif 'FederatedEMNIST' in args.dataset and args.model == 'CNN':
        model = CNN_DropOut(False)
    elif args.dataset == 'FedCIFAR100' and args.model == 'ResNet':
        model = resnet18(num_classes=args.num_classes, group_norm=args.num_gn)  # ResNet18+GN
    elif args.dataset == 'CelebA' and args.model == 'CNN':
        model = ModelCNNCeleba()
    elif args.dataset == 'PartitionedCIFAR10':
        model = CNN_CIFAR_dropout()

    model = model.to(args.device)
    # 如果启用了模型并行，则使用DataParallel包装模型。
    if args.parallel:
        model = torch.nn.DataParallel(model, output_device=0)
    return model


# 定义一个函数federated_algorithm，用于根据参数args选择不同的联邦学习算法。
def federated_algorithm(dataset, model, args):
    train_sizes = dataset['train']['data_sizes']
    if args.fed_algo == 'FedAdam':
        return FedAdam(train_sizes, model, args=args)
    else:
        return FedAvg(train_sizes, model)


# 定义一个函数client_selection_method，用于根据参数args选择不同的客户端选择方法。
def client_selection_method(args):
    #total = args.total_num_client if args.num_available is None else args.num_available
    # 根据args.method的值，返回不同的客户端选择对象。
    kwargs = {'total': args.total_num_client, 'device': args.device}
    if args.method == 'Random':
        return RandomSelection(**kwargs)
    elif args.method == 'AFL':
        return ActiveFederatedLearning(**kwargs, args=args)
    elif args.method == 'Cluster1':
        return ClusteredSampling1(**kwargs, n_cluster=args.num_clients_per_round)
    elif args.method == 'Cluster2':
        return ClusteredSampling2(**kwargs, dist=args.distance_type)
    elif args.method == 'Pow-d':
        assert args.num_candidates is not None
        return PowerOfChoice(**kwargs, d=args.num_candidates)
    elif args.method == 'DivFL':
        assert args.subset_ratio is not None
        return DivFL(**kwargs, subset_ratio=args.subset_ratio)
    elif args.method == 'GradNorm':
        return GradNorm(**kwargs)
    else:
        raise('CHECK THE NAME OF YOUR SELECTION METHOD')



if __name__ == '__main__':
    # set up
    # 获取命令行参数。
    args = get_args()
    # 如果有评论参数，则添加到 args.comment 中。
    if args.comment != '': args.comment = '-'+args.comment
    # 如果使用的不是默认的联邦学习算法，则添加到args.comment中。
    #if args.labeled_ratio < 1: args.comment = f'-L{args.labeled_ratio}{args.comment}'
    if args.fed_algo != 'FedAvg': args.comment = f'-{args.fed_algo}{args.comment}'
    # 如果wandb模块可用，初始化wandb项目并记录代码。
    # save to wandb
    args.wandb = AVAILABLE_WANDB
    if args.wandb:
        wandb.init(
            project=f'AFL-{args.dataset}-{args.num_clients_per_round}-{args.num_available}-{args.total_num_clients}',
            name=f"{args.method}{args.comment}",
            config=args,
            dir='.',
            save_code=True
        )
        wandb.run.log_code(".", include_fn=lambda x: 'src/' in x or 'main.py' in x)

    # fix seed
    # 如果需要固定随机种子以确保实验可重复，则设置随机种子。
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # device setting
    # 设置设备（CPU或GPU）。
    if args.gpu_id == 'cpu' or not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        if ',' in args.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
        args.device = torch.device(f"cuda:{args.gpu_id[0]}")
        torch.cuda.set_device(args.device)
        print('Current cuda device ', torch.cuda.current_device())

    # set data
    # 加载数据集。
    data = load_data(args)
    args.num_classes = data.num_classes
    args.total_num_client, args.test_num_clients = data.train_num_clients, data.test_num_clients
    dataset = data.dataset

    # set model
    # 创建模型。
    model = create_model(args)
    client_selection = client_selection_method(args)
    fed_algo = federated_algorithm(dataset, model, args)

    # save results
    # 保存结果。
    files = utils.save_files(args)

    ## train
    # set federated optim algorithm
    # 训练模型。
    ServerExecute = Server(dataset, model, args, client_selection, fed_algo, files)
    ServerExecute.train()
