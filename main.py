import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import args
from dataset import get_mnist_data, split_data
from models.network import SimpleCNN
from utils import flatten_params, unflatten_params
from nodes.client import LocalClient
from nodes.edge import EdgeServer


def test_model(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=args.batch_size)
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(dataset)


def run_simulation(method='baseline'):
    print(f"\n>>> 开始仿真: Method = {method}")
    print(f"    预热轮数: {args.warmup_rounds}, 掉队概率: {args.straggler_prob}")

    # 1. 准备数据
    train_data, test_data = get_mnist_data()
    user_groups = split_data(train_data, args.num_users)

    # 2. 初始化模型
    global_model = SimpleCNN().to(args.device)
    param_dim = flatten_params(global_model).numel()
    print(f"    模型参数量: {param_dim}")

    # 3. 初始化各层节点
    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]

    edge_servers = []
    users_per_edge = args.num_users // args.num_edge_servers
    for i in range(args.num_edge_servers):
        assigned_clients = clients[i * users_per_edge: (i + 1) * users_per_edge]
        edge_servers.append(EdgeServer(i, assigned_clients, param_dim))

    acc_history = []

    # 4. 全局迭代
    for epoch in range(args.num_global_rounds):
        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []

        # Tier-2 聚合 (支持 baseline, dmfl, salf)
        for edge_server in edge_servers:
            # 传入 method 和 global_model (SALF 需要 global_model 获取网络层结构进行梯度截断)
            agg_grad = edge_server.aggregate(global_weights, method=method, global_model=global_model)
            if agg_grad is not None:
                edge_grads.append(agg_grad)

        # Tier-3 聚合
        if len(edge_grads) > 0:
            global_grad = torch.stack(edge_grads).mean(dim=0)

            # W_new = W_old - Gradient
            curr_params = flatten_params(global_model)
            new_params = curr_params - global_grad
            unflatten_params(global_model, new_params)

        # 测试
        acc = test_model(global_model, test_data)
        acc_history.append(acc)
        status = "Warmup" if epoch < args.warmup_rounds and method == 'dmfl' else "Active"
        print(f"Round {epoch + 1:02d}/{args.num_global_rounds} [{status}] | Accuracy: {acc:.2f}%")

    return acc_history


if __name__ == '__main__':
    # 运行对比实验
    acc_baseline = run_simulation(method='baseline')
    acc_salf = run_simulation(method='salf')
    acc_dmfl = run_simulation(method='dmfl')

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_global_rounds + 1), acc_baseline, 'r--o', label=f'Baseline (Drop Stragglers)')
    plt.plot(range(1, args.num_global_rounds + 1), acc_salf, 'g-^', label=f'SALF (Layer-wise Update)')
    plt.plot(range(1, args.num_global_rounds + 1), acc_dmfl, 'b-s', label=f'DMFL (Diffusion Compensation)')

    # 绘制预热分界线
    if args.warmup_rounds > 0:
        plt.axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End (DMFL)')

    plt.xlabel('Global Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'6G 3-Tier FL: Baseline vs SALF vs DMFL\n(Non-IID, Straggler Prob: {args.straggler_prob})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()