import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import args
from dataset import get_dataset, split_data
from models.network import SimpleCNN, CNNCifar
from utils import flatten_params, unflatten_params
from nodes.client import LocalClient
from nodes.edge import EdgeServer

def get_salf_partial_grad(model, full_grad_vec):
    param_sizes = [p.numel() for p in model.parameters()]
    num_of_layers = len(param_sizes)
    up_to_layer = np.random.randint(1, num_of_layers + 1)
    layers_to_zero = num_of_layers - up_to_layer
    num_zeros = sum(param_sizes[:layers_to_zero])

    salf_grad_vec = full_grad_vec.clone()
    if num_zeros > 0:
        salf_grad_vec[:num_zeros] = 0.0
    return salf_grad_vec

def evaluate_model(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size)
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(dataset), total_loss / len(dataset)

def run_salf(skip_train_eval=False):
    print("\n>>> 正在进行仿真: Method = SALF (CIFAR-10)")
    train_data, test_data = get_dataset()
    user_groups = split_data(train_data, args.num_users)

    if args.dataset_name == 'cifar10':
        global_model = CNNCifar().to(args.device)
    else:
        global_model = SimpleCNN().to(args.device)

    param_dim = flatten_params(global_model).numel()

    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]
    edge_servers = []
    users_per_edge = args.num_users // args.num_edge_servers
    for i in range(args.num_edge_servers):
        edge_servers.append(EdgeServer(i, clients[i * users_per_edge: (i + 1) * users_per_edge], param_dim))

    t_acc_hist, t_loss_hist, v_acc_hist, v_loss_hist = [], [], [], []
    pbar = tqdm(range(1, args.num_global_rounds + 1), desc="Training [SALF]", ncols=100, file=sys.stdout)

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []

        for edge_server in edge_servers:
            valid_grads = []

            client_conditions = [client.simulate_physical_conditions(param_dim) for client in edge_server.clients]
            total_times = [c[0] + c[1] for c in client_conditions]

            K_min = max(2, int(len(edge_server.clients) * 0.5))
            sorted_times = sorted(total_times)
            dynamic_t_win = min(args.t_deadline, sorted_times[K_min - 1])

            for local_idx, client in enumerate(edge_server.clients):
                t_train, t_up = client_conditions[local_idx][:2]
                if (t_train + t_up) <= dynamic_t_win:
                    valid_grads.append(client.train(global_weights))
                else:
                    full_grad = client.train(global_weights)
                    valid_grads.append(get_salf_partial_grad(global_model, full_grad))

            if valid_grads:
                edge_grads.append(torch.stack(valid_grads).mean(dim=0))

        if edge_grads:
            global_grad = torch.stack(edge_grads).mean(dim=0)
            unflatten_params(global_model, flatten_params(global_model) - global_grad)

        val_acc, val_loss = evaluate_model(global_model, test_data)
        v_acc_hist.append(val_acc)
        v_loss_hist.append(val_loss)

        if not skip_train_eval:
            train_acc, train_loss = evaluate_model(global_model, train_data)
            t_acc_hist.append(train_acc)
            t_loss_hist.append(train_loss)
        else:
            t_acc_hist.append(0.0)
            t_loss_hist.append(0.0)

        pbar.set_postfix({'Val Acc': f"{val_acc:.2f}%", 'Val Loss': f"{val_loss:.4f}"})

    return t_acc_hist, t_loss_hist, v_acc_hist, v_loss_hist


if __name__ == '__main__':
    # 独立运行时，默认计算训练集 (skip_train_eval=False)
    t_acc, t_loss, v_acc, v_loss = run_salf(skip_train_eval=False)
    epochs = range(1, args.num_global_rounds + 1)
    ms = 3  # 缩小描点体积

    # 绘制独立运行时的 Train vs Validation 对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 精度图
    axes[0].plot(epochs, t_acc, 'b--o', markersize=ms, label='Train Accuracy')
    axes[0].plot(epochs, v_acc, 'r-^', markersize=ms, label='Validation Accuracy')
    if args.warmup_rounds > 0:
        axes[0].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[0].set_title(f'SALF ({args.dataset_name.upper()}): Accuracy')
    axes[0].set_xlabel('Global Communication Rounds')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True)

    # Loss图
    axes[1].plot(epochs, t_loss, 'b--o', markersize=ms, label='Train Loss')
    axes[1].plot(epochs, v_loss, 'r-^', markersize=ms, label='Validation Loss')
    if args.warmup_rounds > 0:
        axes[1].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[1].set_title(f'SALF ({args.dataset_name.upper()}): Loss')
    axes[1].set_xlabel('Global Communication Rounds')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()