# config for client selection

# 这段代码定义了几个列表，这些列表包含了不同的客户端选择方法的名称，用于联邦学习环境中的客户端选择策略。以下是对每个列表的解释：

# 这些列表为联邦学习框架提供了灵活的客户端选择策略，允许研究人员和开发者根据具体的应用场景和需求选择合适的方法。

PRE_SELECTION_METHOD = ['Random', 'Cluster1', 'Cluster2', 'NumDataSampling', 'NumDataSampling_rep',
                        'Random_d', 'Random_d_smp']
# 这个列表包含了在本地训练之前执行客户端选择的方法名称。这些方法可能用于减少参与每轮训练的客户端数量，以提高效率。

# POST_SELECTION: 'Pow-d','AFL','MaxEntropy','MaxEntropySampling','MaxEntropySampling_1_p','MinEntropy',
#                 'GradNorm','GradSim','GradCosSim','OCS','DivFL','LossCurr','MisClfCurr'

NEED_SETUP_METHOD = ['Cluster1', 'Cluster2', 'Pow-d', 'NumDataSampling', 'NumDataSampling_rep',
                     'Random_d_smp', 'GradSim', 'GradCosSim',
                     'Powd_baseline0', 'Powd_baseline1', 'Powd_baseline2']
# 这个列表包含了需要在训练开始之前进行设置的方法名称。这些方法可能需要一些初始化步骤，例如计算聚类中心或设置特定的参数。

NEED_INIT_METHOD = ['Cluster2', 'OCS', 'DivFL']
# 这个列表包含了需要在每轮训练开始时进行初始化的方法名称。这些方法可能需要在每轮训练开始时重置某些参数或状态。

CANDIDATE_SELECTION_METHOD = ['Pow-d', 'Powd_baseline0', 'Powd_baseline1', 'Powd_baseline2']
# 这个列表包含了用于候选客户端选择的方法名称。这些方法可能用于从所有可用客户端中选择一部分作为候选客户端。

NEED_LOCAL_MODELS_METHOD = ['GradNorm', 'GradSim', 'GradCosSim', 'OCS', 'DivFL']
# 这个列表包含了需要使用客户端的本地模型来进行选择的方法名称。这些方法可能基于客户端的本地模型的性能或特征来选择客户端。

LOSS_THRESHOLD = ['LossCurr']
# 这个列表包含了需要使用损失函数阈值来选择客户端的方法名称。这些方法可能基于客户端的损失值来选择客户端。

CLIENT_UPDATE_METHOD = ['DoCL']
# 这个列表包含了客户端更新方法的名称。这些方法可能涉及到客户端如何根据全局模型更新其本地模型。
