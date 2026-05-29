import matplotlib.pyplot as plt
from config import args
from models.network import SimpleCNN, VGG11CIFAR


def calc_costs():
    print(">>> 联邦学习开销分析")

    if args.dataset_name == 'cifar10':
        model = VGG11CIFAR()
        d_dim = args.diff_dim_cifar
    else:
        model = SimpleCNN()
        d_dim = args.diff_dim_mnist

    sizes = [p.numel() for p in model.parameters()]
    p_dim = sum(sizes)
    model_mb = p_dim * 4 / (1024 * 1024)

    surv_rate = 1.0 - args.target_straggler_rate
    n_surv = int(args.num_users * surv_rate)
    n_drop = args.num_users - n_surv

    # 上行通信 (MB/轮)
    comm_fedavg = n_surv * model_mb
    comm_dmfl = n_surv * model_mb

    # SALF 平均残缺大小
    avg_partial = sum(sum(sizes[-i:]) for i in range(1, len(sizes) + 1)) / len(sizes)
    partial_mb = avg_partial * 4 / (1024 * 1024)
    comm_salf = n_surv * model_mb + n_drop * partial_mb

    # 服务器计算 (MFLOPs/轮)
    comp_fedavg = (n_surv * p_dim) / 1e6
    comp_salf = (n_surv * p_dim + n_drop * avg_partial) / 1e6

    mlp_flops = 4 * d_dim * args.diff_hidden_dim
    diff_train = 3 * mlp_flops * 32 * 10 * args.num_edge_servers
    diff_gen = mlp_flops * n_drop
    comp_dmfl = (n_surv * p_dim) / 1e6 + (diff_train + diff_gen) / 1e6

    # ---- 打印 ----
    print("=" * 55)
    print(f" 数据集: {args.dataset_name.upper()} | 参数量: {p_dim:,}")
    print(f" 设备: {args.num_users} | 掉队率: {args.target_straggler_rate * 100:.0f}%")
    print(f" 存活: {n_surv} | 掉队: {n_drop}")
    print("-" * 55)
    print(" 上行通信 (MB/轮):")
    print(f"   FedAvg : {comm_fedavg:8.2f}")
    print(f"   SALF   : {comm_salf:8.2f}")
    print(f"   DMFL   : {comm_dmfl:8.2f}  (掉队者零通信)")
    print("-" * 55)
    print(" 服务器计算 (MFLOPs/轮):")
    print(f"   FedAvg : {comp_fedavg:8.2f}")
    print(f"   SALF   : {comp_salf:8.2f}")
    print(f"   DMFL   : {comp_dmfl:8.2f}  (含扩散模型训练)")
    print("=" * 55)

    # ---- 绘图 ----
    labels = ['FedAvg', 'SALF', 'DMFL']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    comm_vals = [comm_fedavg, comm_salf, comm_dmfl]
    b1 = axes[0].bar(labels, comm_vals,
                     color=colors, alpha=0.85, width=0.5, edgecolor='black')
    axes[0].set_title(f'Upload Comm Cost ({args.target_straggler_rate * 100:.0f}% Straggler)')
    axes[0].set_ylabel('MB')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for bar in b1:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(comm_vals) * 0.02,
                     f'{bar.get_height():.2f}', ha='center', va='bottom')

    b2 = axes[1].bar(labels, [comp_fedavg, comp_salf, comp_dmfl],
                     color=colors, alpha=0.85, width=0.5, edgecolor='black')
    axes[1].set_title('Server Computation (Log Scale)')
    axes[1].set_ylabel('MFLOPs')
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    for bar in b2:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                     f'{bar.get_height():.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'Results_Costs_Tradeoff_{args.dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    calc_costs()
