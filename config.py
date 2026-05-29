import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = 'mnist'  # mnist / cifar10

    # --- 联邦学习 ---
    num_users = 30
    num_edge_servers = 3
    num_global_rounds = 300
    num_local_epochs = 3          # mnist: 3, cifar10: 1
    batch_size = 64
    lr = 0.001
    momentum = 0.5
    inner_client_iid = 0

    # --- 物理层 ---
    bandwidth = 1e6
    p_ue_max = 0.2
    kappa = 1e-28
    f_cpu_min = 1e9
    f_cpu_max = 2e9
    cycles_per_sample = 2e4
    target_straggler_rate = 0.7
    t_deadline = 99999.0
    noise_scale = 0.001

    # --- 扩散模型 ---
    diff_timesteps = 50
    diff_hidden_dim = 1024
    diff_lr = 0.001
    diff_dim_mnist = 510
    diff_dim_cifar = 5130
    warmup_rounds = int(num_global_rounds * 0.1)

    data_path = './data'


args = Config()
