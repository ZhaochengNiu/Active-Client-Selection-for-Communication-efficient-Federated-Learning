# 这段代码定义了一个使用PyTorch框架的双向长短时记忆网络（BLSTM）模型，通常用于处理序列数据，如文本或时间序列。下面是对代码的逐行解释：

import torch.nn as nn
# 这行代码导入了PyTorch的神经网络模块，它提供了构建神经网络所需的类和函数。

# 这个模型可以用于分类任务，其中输入是序列数据，输出是类别标签的预测。例如，它可以用于情感分析，其中输入是电影评论，输出是正面或负面情感的预测。


class BLSTM(nn.Module):
    # 定义了一个名为BLSTM的类，它继承自nn.Module。nn.Module是PyTorch中所有神经网络模块的基类。
    def __init__(self, embedding_dim=64, vocab_size=500, blstm_hidden_size=32, mlp_hidden_size=64, blstm_num_layers=1, num_classes=2):
        # 这是BLSTM类的构造函数，它初始化模型的各个组件。函数参数定义了模型的不同配置：
        # embedding_dim：嵌入层的维度。
        # vocab_size：词汇表的大小，即嵌入层中嵌入向量的数量。
        # blstm_hidden_size：BLSTM层的隐藏单元数。
        # mlp_hidden_size：多层感知机（MLP）隐藏层的大小，但在这个模型中并未使用。
        # blstm_num_layers：BLSTM层的数量。
        # num_classes：输出类别的数量。
        super(BLSTM, self).__init__()
        # 这行代码调用了父类nn.Module的构造函数，是初始化模型组件前的必要步骤。
        # AFL: 64-dim embedding, 32-dim BLSTM, MLP with one layer(64-dim)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        # 初始化一个嵌入层，将词汇表中的每个词映射到一个固定维度的向量。
        self.blstm = nn.LSTM(input_size=embedding_dim, hidden_size=blstm_hidden_size, num_layers=blstm_num_layers,
                             batch_first=True, bidirectional=True)
        # 初始化一个LSTM层，设置为双向（bidirectional=True），这意味着每个时间步的输入将被两个LSTM方向处理，向前和向后。
        #self.fc1 = nn.Linear(blstm_hidden_size*2, mlp_hidden_size)
        #self.fc2 = nn.Linear(mlp_hidden_size, 2)
        # 初始化一个全连接层，将BLSTM的输出映射到最终的类别数。由于是双向LSTM，所以隐藏状态的大小是blstm_hidden_size*2。
        self.fc = nn.Linear(blstm_hidden_size*2, num_classes)

    def forward(self, input_seq):
        # 定义了模型的前向传播函数，这是模型如何处理输入数据并产生输出的地方。
        embeds = self.embeddings(input_seq)
        # 将输入序列通过嵌入层，得到每个词的向量表示。
        lstm_out, _ = self.blstm(embeds)
        # 将嵌入后的序列输入到BLSTM层，得到序列的输出和最终的隐藏状态。这里只关心输出，所以使用_忽略隐藏状态。
        final_hidden_state = lstm_out[:, -1]
        # 从BLSTM层的输出中提取每个序列的最后一个时间步的隐藏状态，这通常代表了整个序列的信息。
        #output = self.fc1(final_hidden_state)
        #output = self.fc2(output)
        output = self.fc(final_hidden_state)
        # 将最终的隐藏状态通过全连接层，得到最终的输出，即每个类别的预测分数。
        return output
        # 返回模型的输出，这通常是用于计算损失函数和进行预测的。
