import torch

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 环境设置 ---
    num_users = 30
    num_edge_servers = 3
    num_global_rounds = 40
    num_local_epochs = 3
    batch_size = 32
    lr = 0.01
    momentum = 0.5

    # --- 通信参数 ---
    bandwidth = 1e6
    noise_psd = -174
    p_ue_max = 0.2
    f_cpu_min = 1e9
    f_cpu_max = 2e9
    cycles_per_sample = 2e5
    kappa = 1e-28
    t_deadline = 1.0
    noise_scale = 0.02

    # --- 扩散模型设置 ---
    diff_timesteps = 20
    diff_hidden_dim = 64
    diff_lr = 0.005
    warmup_rounds = 5

    data_path = './data'

args = Config()