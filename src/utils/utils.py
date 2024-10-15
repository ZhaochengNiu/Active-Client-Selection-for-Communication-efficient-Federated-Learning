import platform
import time
import os
# 导入了必要的 Python 模块，用于获取机器名称、时间戳和文件操作。

# 这段代码定义了一个名为 save_files 的函数，其目的是为联邦学习实验创建一个目录结构，并在其中保存实验的配置和结果。以下是对代码的逐行解释：

# 这个函数为联邦学习实验创建了一个完整的文件结构，用于保存实验的配置、结果和客户端信息。通过这种方式，可以方便地追踪和记录实验的进展和结果。


def save_files(args):
    # 定义了一个函数 save_files，它接受一个包含实验参数的 args 对象。
    args.machine = platform.uname().node
    args.start = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    # 记录实验的机器名称和开始时间。
    #args.save_path = f'./results/{args.dataset}/{args.method}-{args.fed_algo}-{args.num_clients_per_round}-{args.total_num_clients}/{args.start}'
    alpha = f'_a{args.alpha2}' if 'MaxEntropySampling' in args.method else ''
    dirichlet_alpha = f'_Da{args.dirichlet_alpha}' if args.dataset == 'PartitionedCIFAR10' else ''
    # 根据实验方法和数据集，生成额外的标识符，用于区分不同的实验设置。
    add = ''
    if args.loss_div_sqrt:
        add += '_sqrt'
    elif args.loss_sum:
        add += '_total'
        # 根据损失函数的设置，添加相应的后缀。
    path = f'./results/{args.dataset}/{args.method}{alpha}{dirichlet_alpha}{add}-{args.start}'
    os.makedirs(path, exist_ok=True)
    # 创建一个包含实验结果的目录路径，并确保该目录存在。
    if args.loss_sum:
        args.comment += '_total'
    elif args.loss_div_sqrt:
        args.comment += '_sqrt'
    # 更新实验注释，以反映损失函数的设置。
    args.file_name_opt = f'{args.method}-{args.fed_algo}-{args.num_clients_per_round}-{args.total_num_clients}{args.comment}'
    # 生成一个用于保存实验结果的文件名。
    opts_file = open(f'{path}/options_{args.file_name_opt}_{args.start}.txt', 'w')
    # 创建一个文件用于保存实验的配置参数。
    opts_file.write('=' * 30 + '\n')
    for arg in vars(args):
        opts_file.write(f' {arg} = {getattr(args, arg)}\n')
    # 将实验的所有参数写入配置文件。
    opts_file.write('=' * 30 + '\n')

    result_files = {}
    result_files['result'] = open(f'{path}/results_{args.file_name_opt}_{args.start}.txt', 'w')
    result_files['result'].write('Round,TrainLoss,TrainAcc,TestLoss,TestAcc\n')
    # 创建一个文件用于保存实验结果，并写入标题行。
    result_files['client'] = open(f'{path}/client_{args.file_name_opt}_{args.start}.txt', 'w')
    # 创建一个文件用于保存客户端信息。
    if args.save_probs:
        result_files['prob'] = open(f'{path}/probs_{args.file_name_opt}_{args.start}.txt', 'w')
        result_files['num_samples'] = open(f'{path}/num_samples_{args.file_name_opt}_{args.start}.txt', 'w')
    # 如果需要保存概率信息，创建相应的文件。
    return result_files
    # 返回一个包含所有结果文件句柄的字典。
