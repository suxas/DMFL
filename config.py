import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 环境设置 ---
    num_users = 30  # 用户数
    num_edge_servers = 3  # 边缘服务器
    straggler_prob = 0.5  #掉队率
    num_global_rounds = 30  #全局训练次数
    num_local_epochs = 3    #本地训练次数
    batch_size = 32
    lr = 0.01 #
    momentum = 0.5 #SGD动量设置

    # --- 扩散模型设置 ---
    diff_timesteps = 20  # 时间步
    diff_hidden_dim = 64
    diff_lr = 0.005  # 学习率
    warmup_rounds = 3  # 预热轮数

    data_path = './data'


args = Config()