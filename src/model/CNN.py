'''
Reference:
    FedML: https://github.com/FedML-AI/FedML
'''

# 这段代码定义了四个不同的卷积神经网络（CNN）模型，分别用于不同的数据集和任务。下面是对每个模型的逐行解释：

# 这些模型都是用于图像分类任务的CNN，每个模型都有其特定的结构和参数，以适应不同的数据集和任务需求。如果你需要我解析特定的GitHub链接，我可以尝试再次解析，但请注意，如果链接无效或网络问题，解析可能会失败。如果你有其他问题或需要进一步的帮助，请告诉我。

import torch
import torch.nn as nn
import torch.nn.functional as F


'''Federated EMNIST'''
class CNN_DropOut(nn.Module):
    def __init__(self, only_digits=False, num_channel=1):
        super(CNN_DropOut, self).__init__()
        # 定义了一个名为CNN_DropOut的类，继承自nn.Module，用于构建模型。
        # 构造函数接受两个参数：only_digits用于指示是否只处理数字，num_channel表示输入通道的数量。
        self.conv2d_1 = nn.Conv2d(num_channel, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # 初始化模型的层，包括两个卷积层、最大池化层、两个Dropout层、一个Flatten层和两个全连接层。ReLU激活函数也被初始化。
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 定义了模型的前向传播函数。
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        # 在前向传播中，输入数据x通过卷积层、ReLU激活函数、最大池化层、Dropout层、Flatten层、全连接层和ReLU激活函数，
        # 最后通过第二个全连接层得到输出。
        return x


'''CelebA'''
class CNN(nn.Module):
    # 定义了一个名为CNN的类，用于CelebA数据集。构造函数接受输入通道数和类别数作为参数。
    def __init__(self, num_channel=3, num_class=2):
        super(CNN, self).__init__()
        # 初始化四个卷积层和一个全连接层。每个卷积层都是通过_make_layer函数创建的，
        # 该函数创建一个包含卷积、批量归一化、最大池化和ReLU激活函数的序列。
        self.layer1 = self._make_layer(num_channel, 32, 15)
        self.layer2 = self._make_layer(32, 32, 15)
        self.layer3 = self._make_layer(32, 32, 16)
        self.layer4 = self._make_layer(32, 32, 16)
        self.fc = nn.Linear(1152, num_class)

        # 使用Xavier均匀初始化方法初始化权重。
        self.layer1.apply(xavier_uniform)
        self.layer2.apply(xavier_uniform)
        self.layer3.apply(xavier_uniform)
        self.layer4.apply(xavier_uniform)
        self.fc.apply(xavier_uniform)


    def _make_layer(self, inp, outp=32, pad=0):
        # 定义了一个私有函数_make_layer，用于创建卷积层。
        layers = [
            nn.Conv2d(inp, outp, kernel_size=3, padding=(outp - inp)//2),
            nn.BatchNorm2d(outp),
            nn.MaxPool2d(outp, stride=2, padding=pad),
            nn.ReLU()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # 定义了模型的前向传播函数。
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # 在前向传播中，输入数据x通过四个卷积层，然后被展平并通过全连接层得到输出。
        return x



def xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


'''CelebA

Reference:
    https://github.com/PengchaoHan/EasyFL
'''
# 这个模型与上面的CNN模型类似，但使用了不同的层结构和参数。
class ModelCNNCeleba(nn.Module):
    def __init__(self):
        super(ModelCNNCeleba, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(1152, 2)

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output = output4.view(-1, 1152)
        output = self.fc(output)
        return output



'''Partitioned CIFAR100'''
class CNN_CIFAR_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""
    # 定义了一个名为CNN_CIFAR_dropout的类，用于处理Partitioned CIFAR100数据集。
    def __init__(self, num_class=10):
        super(CNN_CIFAR_dropout, self).__init__()
        # 构造函数接受类别数作为参数。
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )
        # 初始化三个卷积层。
        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, num_class)

        self.dropout = nn.Dropout(p=0.2)
        # 初始化两个全连接层和一个Dropout层。

    def forward(self, x):
        # 定义了模型的前向传播函数。
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        # 在前向传播中，输入数据x通过卷积层、ReLU激活函数、最大池化层和Dropout层。

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        # 重复上述过程。

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 64)
        # 通过第三个卷积层，然后应用Dropout和展平操作。

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x
        # 通过第一个全连接层和ReLU激活函数，然后通过第二个全连接层得到输出。