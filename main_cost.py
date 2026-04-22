import numpy as np
import matplotlib.pyplot as plt
from config import args
from models.network import SimpleCNN, VGG11CIFAR


def calculate_costs():
    print(">>> 正在分析联邦学习仿真开销...")

    # 1. 初始化模型并获取物理参数量
    if args.dataset_name == 'cifar10':
        global_model = VGG11CIFAR()
    else:
        global_model = SimpleCNN()

    param_sizes = [p.numel() for p in global_model.parameters()]
    param_dim = sum(param_sizes)
    model_size_mb = param_dim * 4 / (1024 * 1024)  # Float32类型占用4字节

    # 2. 存活与掉队统计
    survival_rate = 1.0 - args.target_straggler_rate
    num_survivors = int(args.num_users * survival_rate)
    num_dropouts = args.num_users - num_survivors

    # 3. 通信开销计算 (每轮上传的总数据量)
    # FedAvg: 仅存活者上传完整模型
    fedavg_comm = num_survivors * model_size_mb

    # DMFL: 存活者上传完整模型，掉队者零通信 (边缘端预测)
    dmfl_comm = num_survivors * model_size_mb

    # SALF: 存活者上传完整模型，掉队者上传随机层 (Partial Model)
    salf_partial_avg = 0
    for up_to_layer in range(1, len(param_sizes) + 1):
        salf_partial_avg += sum(param_sizes[-up_to_layer:])
    salf_partial_avg /= len(param_sizes)  # 计算平均需要上传的残缺层大小
    salf_partial_mb = salf_partial_avg * 4 / (1024 * 1024)
    salf_comm = (num_survivors * model_size_mb) + (num_dropouts * salf_partial_mb)

    # 打印分析报告
    print("\n" + "=" * 50)
    print("联邦学习算法开销分析报告 (Cost Analysis)")
    print("=" * 50)
    print(f"模型结构: {args.dataset_name.upper()} Model")
    print(f"总参数量: {param_dim:,} ({model_size_mb:.2f} MB)")
    print(f"全网设备数: {args.num_users} | 目标掉队率: {args.target_straggler_rate * 100}%")
    print(f"每轮存活设备: {num_survivors} | 每轮掉队设备: {num_dropouts}")
    print("-" * 50)
    print("【每轮上行通信开销 (Upload Communication Cost / Round)】")
    print(f"  - FedAvg : {fedavg_comm:.2f} MB  (仅存活者上传完整模型)")
    print(f"  - SALF   : {salf_comm:.2f} MB  (掉队者上传部分特征层，带宽高)")
    print(f"  - DMFL   : {dmfl_comm:.2f} MB  (掉队者零通信，极省带宽)")
    print("=" * 50)

    # 可视化柱状图
    labels = ['FedAvg', 'SALF', 'DMFL']
    comm_costs = [fedavg_comm, salf_comm, dmfl_comm]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, comm_costs, color=['red', 'green', 'blue'], alpha=0.7, width=0.5)
    plt.title(f'Communication Cost per Round ({args.dataset_name.upper()}, {args.target_straggler_rate * 100}% Drop)')
    plt.ylabel('Upload Communication Cost (MB)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 顶部加标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f} MB', ha='center', va='bottom',
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"Results_Cost_Communication_{args.dataset_name}.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    calculate_costs()