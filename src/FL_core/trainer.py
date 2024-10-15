from copy import deepcopy
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
# 这些行导入了所需的Python库，包括PyTorch深度学习框架、NumPy数学库、scikit-learn机器学习库等。

# 这段代码定义了一个名为 Trainer 的类，它负责在联邦学习环境中对模型进行训练和测试。下面是对代码的逐行解释：
# 这个 Trainer 类封装了模型的训练和测试过程，使得在联邦学习环境中可以方便地对模型进行本地更新和评估。
# 通过这种方式，可以在保护数据隐私的同时提高模型的性能。

class Trainer:
    def __init__(self, model, args):
        """
        trainer
        ---
        Args
            model: given model  for training (or test)
            args: arguments for FL training
        """
        # 定义了 Trainer 类的构造函数，它接受一个模型和一个包含训练参数的 args 对象。
        self.device = args.device
        self.num_classes = args.num_classes
        # 初始化设备（CPU或GPU）和类别数量。
        # hyperparameter
        self.lr = args.lr_local
        self.wdecay = args.wdecay
        self.momentum = args.momentum
        self.num_epoch = args.num_epoch    # num of local epoch E
        self.num_updates = args.num_updates  # num of local updates u
        self.batch_size = args.batch_size  # local batch size B
        self.loader_kwargs = {'batch_size': self.batch_size, 'pin_memory': True, 'shuffle': True}
        # 初始化超参数，包括学习率、权重衰减、动量、本地训练周期数、本地更新次数和批量大小。
        # model
        self.model = model
        self.client_optimizer = args.client_optimizer
        # 设置模型和客户端优化器。

    def get_model(self):
        """
        get current model
        """
        self.model.eval()
        return self.model
        # 将模型设置为评估模式并返回。

    def set_model(self, model):
        """
        set current model for training
        """
        self.model.load_state_dict(model.cpu().state_dict())
        # 将传入的模型参数加载到当前模型中。

    def train(self, data):
        """
        train
        ---
        Args
            data: dataset for training
        Returns
            accuracy, loss
        """
        dataloader = DataLoader(data, **self.loader_kwargs)
        # 创建数据加载器。
        self.model = self.model.to(self.device)
        # 将模型移动到指定的设备（CPU或GPU）。
        
        self.model.train()
        # 将模型设置为训练模式。
        # optimizer
        if self.client_optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wdecay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        # 根据客户端优化器的选择初始化优化器。
        criterion = nn.CrossEntropyLoss()
        # 初始化交叉熵损失函数。
        for epoch in range(self.num_epoch):
            loss_lst = []
            output_lst, res_lst = torch.empty((0, self.num_classes)).to(self.device), torch.empty((0, self.num_classes)).to(self.device)
            min_loss, num_ot = np.inf, 0
            train_loss, correct, total = 0., 0, 0
            probs = 0
            for num_update, (input, labels) in enumerate(dataloader):
                input, labels = input.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(input)
                _, preds = torch.max(output.detach().data, 1)

                loss = criterion(output, labels.long())

                loss.backward()
                optimizer.step()

                train_loss += loss.detach().item() * input.size(0)
                
                correct += preds.eq(labels).sum().cpu().data.numpy()
                total += input.size(0)

                
                if self.num_updates is not None and num_update + 1 == self.num_updates:
                    if total < self.batch_size:
                        print(f'break! {total}', end=' ')
                    break

                del input, labels, output
        # 在每个批次上执行前向传播、计算损失、反向传播和优化器步骤。

        self.model = self.model.cpu()
        # 将模型移回CPU。
        assert total > 0
            
        result = {'loss': train_loss / total, 'acc': correct / total, 'metric': train_loss / total}
        # 计算并返回训练损失、准确率和其他指标。
        
        # if you track each client's loss
        # sys.stdout.write(r'\nLoss {:.6f} Acc {:.4f}'.format(result['loss'], result['acc']))
        # sys.stdout.flush()

        return result

    def train_E0(self, data):
        """
        train with no local SGD updates
        ---
        Args
            data: dataset for training
        Returns
            accuracy, loss
        """
        dataloader = DataLoader(data, **self.loader_kwargs)
        # 创建数据加载器。
        self.model = self.model.to(self.device)
        self.model.train()
        # 将模型移动到指定的设备并设置为训练模式。
        # optimizer
        if self.client_optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                       weight_decay=self.wdecay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        # 初始化优化器。
        criterion = nn.CrossEntropyLoss()
        # 初始化交叉熵损失函数。
        correct, total = 0, 0
        batch_loss = []
        for input, labels in dataloader:
            input, labels = input.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            output = self.model(input)

            loss = criterion(output, labels.long())
            _, preds = torch.max(output.data, 1)

            batch_loss.append(loss * input.size(0))  ##### loss sum
            total += input.size(0).detach().cpu().data.numpy()
            correct += preds.eq(labels).sum().detach().cpu().data.numpy()
        # 执行前向传播和损失计算。
        train_acc = correct / total
        avg_loss = sum(batch_loss) / total

        avg_loss.backward()
        optimizer.step()
        # 执行反向传播和优化器步骤。
        sys.stdout.write('\rTrainLoss {:.6f} TrainAcc {:.4f}'.format(avg_loss, train_acc))

        result = {'loss': avg_loss.detach().cpu(), 'acc': train_acc}
        # 返回平均训练损失和准确率。
        return result

    #@torch.no_grad()
    def test(self, model, data, ema=False):
        """
        test
        ---
        Args
            model: model for test
            data: dataset for test
        Returns
            accuracy, loss, AUC (optional)
        """
        dataloader = DataLoader(data, **self.loader_kwargs)
        # 创建数据加载器。
        model = model.to(self.device)
        model.eval()
        # 将模型移动到指定的设备并设置为评估模式。
        criterion = nn.CrossEntropyLoss()
        # 初始化交叉熵损失函数。
        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            y_true, y_score = np.empty((0)), np.empty((0))
            output_lst, res_lst = torch.empty((0, self.num_classes)), torch.empty((0, self.num_classes))
            # 初始化测试损失和准确率变量，不需要计算梯度。
            for input, labels in dataloader:
                input, labels = input.to(self.device), labels.to(self.device)
                output = model(input)

                loss = criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                test_loss += loss.detach().cpu().item() * input.size(0)
                correct += preds.eq(labels).sum().detach().cpu().data.numpy()
                total += input.size(0)

                if self.num_classes == 2:
                    y_true = np.append(y_true, labels.detach().cpu().numpy(), axis=0)
                    y_score = np.append(y_score, preds.detach().cpu().numpy(), axis=0)
                
                del input, labels, output, preds
            # 执行前向传播和损失计算。
        assert total > 0

        result = {'loss': test_loss / total, 'acc': correct / total}
        # 计算并返回测试损失和准确率。
        #if self.num_classes == 2:
        #    auc = roc_auc_score(y_true, y_score)
        #    result['auc'] = auc

        return result