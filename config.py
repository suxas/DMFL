"""
4.17 0.9掉队，0.1簇内iid，0.15比例热身，lr分别为0.005和0.001，噪声程度0.001
"""
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = 'mnist'      # mnist/cifar10

    # --- 环境与联邦学习设置 ---
    num_users = 30
    num_edge_servers = 3
    num_global_rounds = 300
    num_local_epochs = 3    # mnist 3，cifar10 1
    batch_size = 64         # 客户端本地训练batchsize
    lr = 0.001              # 客户端本地训练学习率
    momentum = 0.5          # 客户端本地训练动量
    inner_client_iid = 0.5  # 控制簇内IID程度

    # --- 物理通信与计算参数 ---
    bandwidth = 1e6  # 保持 1MHz
    p_ue_max = 0.2  # 保持最大 0.2W
    kappa = 1e-28
    f_cpu_min = 1e9
    f_cpu_max = 2e9
    # 增加每样本计算周期数，让 CNN 的运算符合实际并拉长耗时
    cycles_per_sample = 1e5  # 设为 100,000 (计算耗时拉长到约 0.6 秒)
    # 放大底噪，让信噪比掉到合理的 10~20 dB 范围
    noise_scale = 0.05  # (底噪变大，通信差的设备会被迫变成“掉队者”)
    #  严格限制最大容忍时间
    target_straggler_rate = 0.5
    t_deadline = 1.0  # (限制为 1.0 秒。由于计算需要 0.6秒，留给传输的只有 0.4秒，信道差的设备必超时)

    # --- 扩散模型设置 ---
    diff_timesteps = 50
    diff_hidden_dim = 1024
    diff_lr = 0.001
    warmup_rounds = num_global_rounds * 0.1

    # --- 资源优化器专用参数 ---
    eta = 0.5  # 优化目标权重：(1-eta)*误差 + eta*时延
    t_bh = 0.1  # 宏基站到小基站的回传时延 (s)
    delta_diff_sq = 0.05  # 预估的扩散模型预测误差上界

    data_path = './data'


args = Config()