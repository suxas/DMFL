import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import args
from dataset import get_mnist_data, split_data
from models.network import SimpleCNN
from utils import flatten_params, unflatten_params, evaluate_model
from nodes.client import LocalClient


def get_salf_partial_grad(model, full_grad_vec):
    """
    SALF 截断逻辑：模拟 straggler 仅上传部分深层网络梯度。
    """
    param_sizes = [p.numel() for p in model.parameters()]
    num_of_layers = len(param_sizes)

    # 随机选择从后往前上传的层数
    up_to_layer = np.random.randint(1, num_of_layers + 1)
    layers_to_zero = num_of_layers - up_to_layer
    num_zeros = sum(param_sizes[:layers_to_zero])

    salf_grad_vec = full_grad_vec.clone()
    if num_zeros > 0:
        # 将前端未计算完/未传完的层对应的含噪梯度直接置0
        salf_grad_vec[:num_zeros] = 0.0

    return salf_grad_vec


def run_salf():
    print(f"\n>>> [2/3] 正在运行 SALF (掉队率: {args.target_straggler_rate * 100}%)")
    train_data, test_data = get_mnist_data()
    user_groups = split_data(train_data, args.num_users)

    global_model = SimpleCNN().to(args.device)
    param_dim = flatten_params(global_model).numel()

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]

    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    pbar = tqdm(range(args.num_global_rounds), desc="Training [SALF]", file=sys.stdout, colour='green')

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []

        users_per_edge = args.num_users // args.num_edge_servers
        for i in range(args.num_edge_servers):
            assigned_clients = clients[i * users_per_edge: (i + 1) * users_per_edge]

            # --- 根据目标掉队率设置动态时间窗口 ---
            client_times = [c.simulate_physical_time(param_dim) for c in assigned_clients]
            total_times = [c[0] + c[1] for c in client_times]

            success_rate = 1.0 - args.target_straggler_rate
            K_min = max(1, int(len(assigned_clients) * success_rate))
            dynamic_t_win = min(args.t_deadline, sorted(total_times)[K_min - 1])

            valid_grads = []
            for idx, client in enumerate(assigned_clients):
                t_total = total_times[idx]

                # 附加通信噪声
                full_grad = client.train(global_weights, add_noise=True)

                # 若耗时超过时间窗，触发 SALF 截断
                if t_total > dynamic_t_win:
                    partial_grad = get_salf_partial_grad(global_model, full_grad)
                    valid_grads.append(partial_grad)
                else:
                    valid_grads.append(full_grad)

            if valid_grads:
                edge_grads.append(torch.stack(valid_grads).mean(dim=0))

        if edge_grads:
            global_grad = torch.stack(edge_grads).mean(dim=0)
            curr_params = flatten_params(global_model)
            unflatten_params(global_model, curr_params - global_grad)

        train_acc, train_loss = evaluate_model(global_model, train_loader, args.device)
        val_acc, val_loss = evaluate_model(global_model, test_loader, args.device)
        history['train_acc'].append(train_acc);
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc);
        history['val_loss'].append(val_loss)
        pbar.set_postfix({'Val Acc': f"{val_acc:.2f}%"})

    return history


if __name__ == '__main__':
    history = run_salf()
    epochs = range(1, args.num_global_rounds + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0].plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    axes[0].set_title('SALF Accuracy (Train vs Val)')
    axes[0].legend();
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[1].plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    axes[1].set_title('SALF Loss (Train vs Val)')
    axes[1].legend();
    axes[1].grid(True)
    plt.show()