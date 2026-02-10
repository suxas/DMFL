import torch
import matplotlib.pyplot as plt
import numpy as np

# 导入您现有的配置和模块
from config import args
from dataset import get_mnist_data, split_data
from models.network import SimpleCNN
from utils import flatten_params, unflatten_params
from nodes.client import LocalClient
from nodes.edge import EdgeServer  # 您原有的 DMFL Edge
# 导入新增的 LOGIT Edge
from nodes.edge_logit import LogitEdgeServer


def test_model(model, dataset):
    """测试模型精度的辅助函数"""
    model.eval()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(dataset)


def run_experiment(mode='baseline'):
    """
    运行单次实验
    mode: 'baseline' | 'logit' | 'dmfl'
    """
    print(f"\n{'=' * 20}")
    print(f"Running Experiment: {mode.upper()}")
    print(f"{'=' * 20}")

    # 1. 数据准备
    train_data, test_data = get_mnist_data()
    # 固定随机种子以保证不同算法的数据划分一致，确保对比公平
    torch.manual_seed(42)
    np.random.seed(42)
    user_groups = split_data(train_data, args.num_users)

    # 2. 初始化全局模型
    global_model = SimpleCNN().to(args.device)
    param_dim = flatten_params(global_model).numel()
    print(f"Total Params: {param_dim}, Diffusion/Logit Target Dim: 510")

    # 3. 初始化客户端
    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]

    # 4. 初始化边缘服务器 (根据模式选择不同类型)
    edge_servers = []
    users_per_edge = args.num_users // args.num_edge_servers

    for i in range(args.num_edge_servers):
        assigned_clients = clients[i * users_per_edge: (i + 1) * users_per_edge]

        if mode == 'dmfl':
            # 使用您原本的 Diffusion Edge Server
            edge = EdgeServer(i, assigned_clients, param_dim)
            print(f"Initialized DMFL Edge Server {i}")
        elif mode == 'logit':
            # 使用新增的 LOGIT Edge Server
            edge = LogitEdgeServer(i, assigned_clients, param_dim)
            print(f"Initialized LOGIT Edge Server {i}")
        else:
            # Baseline: 使用普通的 EdgeServer (关闭扩散功能)
            edge = EdgeServer(i, assigned_clients, param_dim)
            print(f"Initialized Baseline Edge Server {i}")

        edge_servers.append(edge)

    acc_history = []

    # 5. 全局训练循环
    for epoch in range(args.num_global_rounds):
        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []

        # 遍历所有边缘服务器进行聚合
        for edge in edge_servers:
            # 根据模式调用不同的聚合接口
            if mode == 'dmfl':
                # DMFL 模式：开启 enable_diffusion
                agg_grad = edge.aggregate(global_weights, enable_diffusion=True)
            elif mode == 'logit':
                # LOGIT 模式：开启 enable_logit
                agg_grad = edge.aggregate(global_weights, enable_logit=True)
            else:
                # Baseline 模式：关闭所有增强功能
                agg_grad = edge.aggregate(global_weights, enable_diffusion=False)

            if agg_grad is not None:
                edge_grads.append(agg_grad)

        # Tier-3 全局聚合 (简单平均)
        if len(edge_grads) > 0:
            global_grad = torch.stack(edge_grads).mean(dim=0)

            # 更新全局模型 W_new = W_old - Gradient
            curr_params = flatten_params(global_model)
            new_params = curr_params - global_grad
            unflatten_params(global_model, new_params)

        # 测试并记录精度
        acc = test_model(global_model, test_data)
        acc_history.append(acc)

        # 打印进度 (标记预热状态)
        status = "Warmup" if epoch < args.warmup_rounds and mode != 'baseline' else "Active"
        print(f"Round {epoch + 1:02d}/{args.num_global_rounds} [{status}] | Acc: {acc:.2f}%")

    return acc_history


if __name__ == '__main__':
    # 按顺序运行三种方案
    acc_baseline = run_experiment('baseline')
    acc_logit = run_experiment('logit')
    acc_dmfl = run_experiment('dmfl')

    # === 绘图部分 ===
    epochs = range(1, args.num_global_rounds + 1)
    plt.figure(figsize=(12, 7))

    # 绘制曲线
    plt.plot(epochs, acc_baseline, 'k--', linewidth=2, label=f'Baseline (Drop, p={args.straggler_prob})')
    plt.plot(epochs, acc_logit, 'b-o', linewidth=2, markersize=6, label='LOGIT (LSTM Trajectory)')
    plt.plot(epochs, acc_dmfl, 'r-s', linewidth=2, markersize=6, label='DMFL (Diffusion Generative)')

    # 标记预热结束线
    if args.warmup_rounds > 0:
        plt.axvline(x=args.warmup_rounds, color='g', linestyle=':', alpha=0.8, label='Warmup End')

    plt.title(f"Performance Comparison: 6G FL with Stragglers\n(Non-IID, Straggler Prob: {args.straggler_prob})")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)

    # 保存图片
    plt.tight_layout()
    plt.savefig('comparison_result.png')
    print("\nSimulation Finished! Plot saved as 'comparison_result.png'")
    plt.show()