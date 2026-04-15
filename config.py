import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = 'cifar10'

    # --- 环境与联邦学习设置 ---
    num_users = 30
    num_edge_servers = 3
    num_global_rounds = 1
    num_local_epochs = 1
    batch_size = 64
    lr = 0.01
    momentum = 0.5

    # --- 物理通信与计算参数 ---
    bandwidth = 1e6
    p_ue_max = 0.2
    kappa = 1e-28
    f_cpu_min = 1e9
    f_cpu_max = 2e9
    cycles_per_sample = 2e4

    # 掉队率与物理死线
    target_straggler_rate = 0.5
    t_deadline = 99999.0
    noise_scale = 0.001

    # --- 扩散模型设置 ---
    diff_timesteps = 50
    diff_hidden_dim = 1024
    diff_lr = 0.001
    warmup_rounds = num_global_rounds * 0.1

    data_path = './data'


args = Config()