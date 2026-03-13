import torch
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    print(f"\n>>> 正在进行仿真: Method = {method}")
    train_data, test_data = get_mnist_data()
    user_groups = split_data(train_data, args.num_users)

    global_model = SimpleCNN().to(args.device)
    param_dim = flatten_params(global_model).numel()

    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]

    edge_servers = []
    users_per_edge = args.num_users // args.num_edge_servers
    for i in range(args.num_edge_servers):
        assigned_clients = clients[i * users_per_edge: (i + 1) * users_per_edge]
        edge_servers.append(EdgeServer(i, assigned_clients, param_dim))

    acc_history = []
    time_history = []
    energy_history = []

    cumulative_time = 0.0
    cumulative_energy = 0.0

    # 使用 tqdm 加入带有 ETA 和进度的加载条
    pbar = tqdm(range(args.num_global_rounds), desc=f"Training [{method}]", ncols=100, file=sys.stdout, colour='green')

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []

        round_max_time = 0.0
        round_total_energy = 0.0

        for edge_server in edge_servers:
            agg_grad, t_time, t_energy = edge_server.aggregate(global_weights, method=method, global_model=global_model)
            if agg_grad is not None:
                edge_grads.append(agg_grad)

            # 层级时延由最慢的小基站决定 (max)
            round_max_time = max(round_max_time, t_time)
            # 系统总能耗为各个基站之和
            round_total_energy += t_energy

        if len(edge_grads) > 0:
            global_grad = torch.stack(edge_grads).mean(dim=0)
            curr_params = flatten_params(global_model)
            new_params = curr_params - global_grad
            unflatten_params(global_model, new_params)

        acc = test_model(global_model, test_data)
        acc_history.append(acc)

        cumulative_time += round_max_time
        cumulative_energy += round_total_energy
        time_history.append(cumulative_time)
        energy_history.append(cumulative_energy)

        # 更新进度条显示的附加信息
        pbar.set_postfix({'Acc': f"{acc:.2f}%", 'Time(s)': f"{cumulative_time:.1f}"})

    return acc_history, time_history, energy_history


if __name__ == '__main__':

    acc_baseline, time_base, energy_base = run_simulation(method='fedavg')
    acc_salf, time_salf, energy_salf = run_simulation(method='salf')
    acc_dmfl, time_dmfl, energy_dmfl = run_simulation(method='dmfl')

    # 绘制三大指标对照图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, args.num_global_rounds + 1)

    # 子图1: 精度对比
    axes[0].plot(epochs, acc_baseline, 'r--o', label='fedavg')
    axes[0].plot(epochs, acc_salf, 'g-^', label='SALF')
    axes[0].plot(epochs, acc_dmfl, 'b-s', label='DMFL')
    if args.warmup_rounds > 0:
        axes[0].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[0].set_xlabel('Global Communication Rounds')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy vs. Rounds')
    axes[0].legend()
    axes[0].grid(True)


    # 子图2: 累计时延对比
    axes[1].plot(epochs, time_base, 'r--o', label='Baseline')
    axes[1].plot(epochs, time_salf, 'g-^', label='SALF')
    axes[1].plot(epochs, time_dmfl, 'b-s', label='DMFL (Ours)')
    axes[1].set_xlabel('Global Communication Rounds')
    axes[1].set_ylabel('Cumulative Time Delay (s)')
    axes[1].set_title('Time Cost Comparison')
    axes[1].legend()
    axes[1].grid(True)



    # 子图3: 累计系统能耗对比
    axes[2].plot(epochs, energy_base, 'r--o', label='Baseline')
    axes[2].plot(epochs, energy_salf, 'g-^', label='SALF')
    axes[2].plot(epochs, energy_dmfl, 'b-s', label='DMFL (Ours)')
    axes[2].set_xlabel('Global Communication Rounds')
    axes[2].set_ylabel('Cumulative Energy (Joules)')
    axes[2].set_title('System Power Consumption')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle('6G 3-Tier FL Communication and Computing Analysis')
    plt.tight_layout()
    plt.show()