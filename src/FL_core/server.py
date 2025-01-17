import torch
import wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import sys
import multiprocessing as mp
import random
# 导入了 Client 类和客户端选择配置。
from .client import Client
from .client_selection.config import *

# 这段代码定义了一个名为 Server 的类，它在联邦学习（Federated Learning, FL）框架中扮演中心服务器的角色。
# 服务器负责初始化全局模型，管理客户端选择，协调客户端训练，以及聚合客户端更新。下面是对代码的逐行解释：


class Server(object):
    def __init__(self, data, init_model, args, selection, fed_algo, files):
        """
        Server to execute
        ---
        Args
            data: data for FL
            init_model: initial global model
            args: arguments for overall FL training
            selection: client selection method
            fed_algo: FL algorithm for aggregation at server
            results: results for recording
        """
        # 定义了 Server 类的构造函数，它接受以下参数：
        #
        # data：FL的Dataset。
        # init_model：初始全局模型。
        # args：整体FL训练的参数。
        # selection：客户端选择方法。
        # fed_algo：FL算法用于服务器端的聚合。
        # files：用于记录结果的文件。
        self.train_data = data['train']['data']
        # 初始化训练数据。
        self.train_sizes = data['train']['data_sizes']
        self.test_data = data['test']['data']
        self.test_sizes = data['test']['data_sizes']
        self.test_clients = data['test']['data_sizes'].keys()

        self.device = args.device
        self.args = args
        self.global_model = init_model
        # 设置全局模型。
        self.selection_method = selection
        # 设置客户端选择方法。
        self.federated_method = fed_algo
        # 设置联邦学习方法用于模型聚合。
        self.files = files

        self.nCPU = mp.cpu_count() // 2 if args.nCPU is None else args.nCPU

        self.total_num_client = args.total_num_client
        self.num_clients_per_round = args.num_clients_per_round
        self.num_available = args.num_available
        if self.num_available is not None:
            random.seed(args.seed)

        self.total_round = args.num_round
        self.save_results = not args.no_save_results
        self.save_probs = args.save_probs

        if self.save_probs:
            num_local_data = np.array([self.train_sizes[idx] for idx in range(args.total_num_client)])
            num_local_data.tofile(files['num_samples'], sep=',')
            files['num_samples'].close()
            del files['num_samples']

        self.test_on_training_data = False

        ## INITIALIZE
        # initialize the training status of each client
        self._init_clients(init_model)

        # initialize the client selection method
        if self.args.method in NEED_SETUP_METHOD:
            self.selection_method.setup(self.train_sizes)

        if self.args.method in LOSS_THRESHOLD:
            self.ltr = 0.0


    def _init_clients(self, init_model):
        """
        initialize clients' model
        ---
        Args
            init_model: initial given global model
        """
        self.client_list = []
        # 初始化客户端列表。
        for client_idx in range(self.total_num_client):
            # 为每个客户端创建一个 Client 实例并添加到客户端列表中。
            local_train_data = self.train_data[client_idx]
            local_test_data = self.test_data[client_idx] if client_idx in self.test_clients else np.array([])
            c = Client(client_idx, self.train_sizes[client_idx], local_train_data, local_test_data,
                       deepcopy(init_model), self.args)
            self.client_list.append(c)

    def train(self):
        """
        FL training
        """
        # 定义了 train 方法，用于执行FL训练。
        ## ITER COMMUNICATION ROUND
        for round_idx in range(self.total_round):
            # 进行多轮训练，每轮训练可能包括客户端选择、本地更新、模型聚合和测试。
            print(f'\n>> ROUND {round_idx}')

            ## GET GLOBAL MODEL
            #self.global_model = self.trainer.get_model()
            self.global_model = self.global_model.to(self.device)

            # set clients
            client_indices = [*range(self.total_num_client)]
            if self.num_available is not None:
                print(f'> available clients {self.num_available}/{len(client_indices)}')
                np.random.seed(self.args.seed + round_idx)
                client_indices = np.random.choice(client_indices, self.num_available, replace=False)
                self.save_selected_clients(round_idx, client_indices)

            # set client selection methods
            # initialize selection methods by setting given global model
            if self.args.method in NEED_INIT_METHOD:
                local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
                self.selection_method.init(self.global_model, local_models)
                del local_models
            # candidate client selection before local training
            if self.args.method in CANDIDATE_SELECTION_METHOD:
                # np.random.seed((self.args.seed+1)*10000000 + round_idx)
                print(f'> candidate client selection {self.args.num_candidates}/{len(client_indices)}')
                client_indices = self.selection_method.select_candidates(client_indices, self.args.num_candidates)


            ## PRE-CLIENT SELECTION
            # client selection before local training (for efficiency)
            if self.args.method in PRE_SELECTION_METHOD:
                # np.random.seed((self.args.seed+1)*10000 + round_idx)
                print(f'> pre-client selection {self.num_clients_per_round}/{len(client_indices)}')
                client_indices = self.selection_method.select(self.num_clients_per_round, client_indices, None)
                # 根据客户端选择方法选择参与本轮训练的客户端。
                print(f'selected clients: {sorted(client_indices)[:10]}')


            ## CLIENT UPDATE (TRAINING)
            local_losses, accuracy, local_metrics = self.train_clients(client_indices)
            # 在选定的客户端上进行本地训练，并返回损失、准确率和本地指标。


            ## CLIENT SELECTION
            if self.args.method not in PRE_SELECTION_METHOD:
                print(f'> post-client selection {self.num_clients_per_round}/{len(client_indices)}')
                kwargs = {'n': self.num_clients_per_round, 'client_idxs': client_indices, 'round': round_idx}
                kwargs['results'] = self.files['prob'] if self.save_probs else None
                # select by local models(gradients)
                if self.args.method in NEED_LOCAL_MODELS_METHOD:
                    local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
                    selected_client_indices = self.selection_method.select(**kwargs, metric=local_models)
                    del local_models
                # select by local losses
                else:
                    selected_client_indices = self.selection_method.select(**kwargs, metric=local_metrics)
                if self.args.method in CLIENT_UPDATE_METHOD:
                    for idx in client_indices:
                        self.client_list[idx].update_ema_variables(round_idx)
                # update local metrics
                client_indices = np.take(client_indices, selected_client_indices).tolist()
                local_losses = np.take(local_losses, selected_client_indices)
                accuracy = np.take(accuracy, selected_client_indices)


            ## CHECK and SAVE current updates
            # self.weight_variance(local_models) # check variance of client weights
            self.save_current_updates(local_losses, accuracy, len(client_indices), phase='Train', round=round_idx)
            self.save_selected_clients(round_idx, client_indices)


            ## SERVER AGGREGATION
            # DEBUGGING
            assert len(client_indices) == self.num_clients_per_round

            # aggregate local models
            local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
            if self.args.fed_algo == 'FedAvg':
                global_model_params = self.federated_method.update(local_models, client_indices)
                # 使用联邦学习算法聚合客户端的模型更新。
            else:
                global_model_params = self.federated_method.update(local_models, client_indices, self.global_model, self.client_list)

            # update aggregated model to global model
            self.global_model.load_state_dict(global_model_params)


            ## TEST
            if round_idx % self.args.test_freq == 0:
                self.global_model.eval()
                # test on train data
                if self.test_on_training_data:
                    self.test(self.total_num_client, phase='TrainALL')
                    self.test_on_training_data = False

                # test on test data
                self.test(len(self.test_clients), phase='Test')

            del local_models, local_losses, accuracy

        for k in self.files:
            # 关闭所有打开的文件。
            if self.files[k] is not None:
                self.files[k].close()


    def local_training(self, client_idx):
        """
        train one client
        ---
        Args
            client_idx: client index for training
        Return
            result: trained model, (total) loss value, accuracy
        """
        # 定义了 local_training 方法，用于在单个客户端上进行训练。
        client = self.client_list[client_idx]
        if self.args.method in LOSS_THRESHOLD:
            client.trainer.update_ltr(self.ltr)
        result = client.train(deepcopy(self.global_model))
        return result

    def local_testing(self, client_idx):
        """
        test one client
        ---
        Args
            client_idx: client index for test
            results: loss, acc, auc
        """
        # 定义了 local_testing 方法，用于在单个客户端上进行测试。
        client = self.client_list[client_idx]
        result = client.test(self.global_model, self.test_on_training_data)
        return result

    def train_clients(self, client_indices):
        """
        train multiple clients (w. or w.o. multi processing)
        ---
        Args
            client_indices: client indices for training
        Return
            trained models, loss values, accuracies
        """
        # 定义了 train_clients 方法，用于在多个客户端上并行进行训练。
        local_losses, accuracy, local_metrics = [], [], []
        ll, lh = np.inf, 0.
        # local training with multi processing
        if self.args.use_mp:
            iter = 0
            with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                iter += 1
                result = list(pool.imap(self.local_training, client_indices))

                result = {k: [result[idx][k] for idx in range(len(result))] for k in result[0].keys()}
                local_losses.extend(result['loss'])
                accuracy.extend(result['acc'])
                local_metrics.extend(result['metric'])

                progressBar(len(local_losses), len(client_indices),
                            {'loss': sum(result['loss'])/len(result), 'acc': sum(result['acc'])/len(result)})

                if self.args.method in LOSS_THRESHOLD:
                    if min(result['llow']) < ll: ll = min(result['llow'])
                    lh += sum(result['lhigh'])
        # local training without multi processing
        else:
            for client_idx in client_indices:
                result = self.local_training(client_idx)

                local_losses.append(result['loss'])
                accuracy.append(result['acc'])
                local_metrics.append(result['metric'])

                if self.args.method in LOSS_THRESHOLD:
                    if result['llow'] < ll: ll = result['llow'].item()
                    lh += result['lhigh']

                progressBar(len(local_losses), len(client_indices), result)

        if self.args.method in LOSS_THRESHOLD:
            lh /= len(client_indices)
            self.ltr = self.selection_method.update(lh, ll, self.ltr)

        print()
        return local_losses, accuracy, local_metrics


    def test(self, num_clients_for_test, phase='Test'):
        """
        test multiple clients
        ---
        Args
            num_clients_for_test: number of clients for test
            TrainALL: test on train data
            Test: test on test data
        """
        # 定义了 test 方法，用于在多个客户端上并行进行测试。
        metrics = {'loss': [], 'acc': []}
        if self.args.use_mp:
            iter = 0
            with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                iter += 1
                result = list(tqdm(pool.imap(self.local_testing, [*range(num_clients_for_test)]),
                                   desc=f'>> local testing on {phase} set'))

                result = {k: [result[idx][k] for idx in range(len(result))] for k in result[0].keys()}
                metrics['loss'].extend(result['loss'])
                metrics['acc'].extend(result['acc'])

                progressBar(len(metrics['acc']) * iter, num_clients_for_test, phase='Test',
                            result={'loss': sum(result['loss']) / len(result), 'acc': sum(result['acc']) / len(result)})
        else:
            for client_idx in range(num_clients_for_test):
                result = self.local_testing(client_idx)

                metrics['loss'].append(result['loss'])
                metrics['acc'].append(result['acc'])

                progressBar(len(metrics['acc']), num_clients_for_test, result, phase='Test')

        print()
        self.save_current_updates(metrics['loss'], metrics['acc'], num_clients_for_test, phase=phase)


    def save_current_updates(self, losses, accs, num_clients, phase='Train', round=None):
        """
        update current updated results for recording
        ---
        Args
            losses: losses
            accs: accuracies
            num_clients: number of clients
            phase: current phase (Train or TrainALL or Test)
            round: current round
        Return
            record "Round,TrainLoss,TrainAcc,TestLoss,TestAcc"
        """
        # 定义了 save_current_updates 方法，用于记录当前的训练或测试结果。
        loss, acc = sum(losses) / num_clients, sum(accs) / num_clients

        if phase == 'Train':
            self.record = {}
            self.round = round
        self.record[f'{phase}/Loss'] = loss
        self.record[f'{phase}/Acc'] = acc
        status = num_clients if phase == 'Train' else 'ALL'

        print('> {} Clients {}ing: Loss {:.6f} Acc {:.4f}'.format(status, phase, loss, acc))

        if phase == 'Test':
            wandb.log(self.record)
            if self.save_results:
                if self.test_on_training_data:
                    tmp = '{:.8f},{:.4f},'.format(self.record['TrainALL/Loss'], self.record['TrainALL/Acc'])
                else:
                    tmp = ''
                rec = '{},{:.8f},{:.4f},{}{:.8f},{:.4f}\n'.format(self.round,
                                                                  self.record['Train/Loss'], self.record['Train/Acc'], tmp,
                                                                  self.record['Test/Loss'], self.record['Test/Acc'])
                self.files['result'].write(rec)

    def save_selected_clients(self, round_idx, client_indices):
        """
        save selected clients' indices
        ---
        Args
            round_idx: current round
            client_indices: clients' indices to save
        """
        # 定义了 save_selected_clients 方法，用于记录每轮选中的客户端。
        self.files['client'].write(f'{round_idx+1},')
        np.array(client_indices).astype(int).tofile(self.files['client'], sep=',')
        self.files['client'].write('\n')

    def weight_variance(self, local_models):
        """
        calculate the variances of model weights
        ---
        Args
            local_models: local clients' models
        """
        # 定义了 weight_variance 方法，用于计算模型权重的方差。
        variance = 0
        for k in tqdm(local_models[0].state_dict().keys(), desc='>> compute weight variance'):
            tmp = []
            for local_model_param in local_models:
                tmp.extend(torch.flatten(local_model_param.cpu().state_dict()[k]).tolist())
            variance += torch.var(torch.tensor(tmp), dim=0)
        variance /= len(local_models)
        print('variance of model weights {:.8f}'.format(variance))



def progressBar(idx, total, result, phase='Train', bar_length=20):
    """
    progress bar
    ---
    Args
        idx: current client index or number of trained clients till now
        total: total number of clients
        phase: Train or Test
        bar_length: length of progress bar
    """
    # 定义了 progressBar 函数，用于显示训练或测试的进度条。
    percent = float(idx) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r> Client {}ing: [{}] {}% ({}/{}) Loss {:.6f} Acc {:.4f}".format(
        phase, arrow + spaces, int(round(percent * 100)), idx, total, result['loss'], result['acc'])
    )
    sys.stdout.flush()