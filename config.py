import torch

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 环境设置 ---
    num_users = 30  # 用户数
    num_edge_servers = 3  # 边缘服务器
    num_global_rounds = 40  # 全局训练次数
    num_local_epochs = 3    # 本地训练次数
    batch_size = 128
    lr = 0.01
    momentum = 0.5

    # --- 通信参数 ---
    bandwidth = 1e6           # 分配给每个用户的带宽 B (Hz), 例如 1MHz
    noise_psd = -174          # 噪声功率谱密度 (dBm/Hz)
    p_ue_max = 0.2            # 用户最大上行发射功率 P^UE (W), 即200mW
    f_cpu_min = 1e9           # 用户最小 CPU 频率 (1 GHz)
    f_cpu_max = 2e9           # 用户最大 CPU 频率 (2 GHz)
    cycles_per_sample = 2e5   # 单样本训练周期数 C_{n,k}
    kappa = 1e-28             # CPU 计算能耗系数 \kappa
    t_deadline = 15.0          # 系统最大允许时间窗口 T_deadline (秒)
    noise_scale = 0.02        # 噪声基础缩放系数

    # --- 扩散模型设置 ---
    diff_timesteps = 20
    diff_hidden_dim = 64
    diff_lr = 0.005
    warmup_rounds = 10   # 3.4 扩散模型热身轮次3次→5次

    data_path = './data'

args = Config()