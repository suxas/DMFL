import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 环境与联邦学习设置 ---
    num_users = 30
    num_edge_servers = 3
    num_global_rounds = 30
    num_local_epochs = 3
    batch_size = 16
    lr = 0.01
    momentum = 0.5

    # --- 物理通信与计算参数 ---
    bandwidth = 1e6
    p_ue_max = 0.2  # UE功率参数
    kappa = 1e-28  # 功率系数
    f_cpu_min = 1e9
    f_cpu_max = 2e9
    cycles_per_sample = 2e4

    # 掉队率
    target_straggler_rate = 0.5
    t_deadline = 1.0  # 物理死线时间
    noise_scale = 0.02  # 6G信道噪声尺度

    # --- 扩散模型设置 ---
    diff_timesteps = 20
    diff_hidden_dim = 64
    diff_lr = 0.005
    warmup_rounds = 5

    data_path = './data'


args = Config()