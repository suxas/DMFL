import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import args
from dataset import get_mnist_data, split_data
from models.network import SimpleCNN
from utils import flatten_params, unflatten_params, evaluate_model
from nodes.client import LocalClient


def run_fedavg():
    print(f"\n>>> [1/3] 正在运行FedAvg (掉队率: {args.target_straggler_rate * 100}%)")
    train_data, test_data = get_mnist_data()
    user_groups = split_data(train_data, args.num_users)
    global_model = SimpleCNN().to(args.device)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]
    param_dim = flatten_params(global_model).numel()

    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    pbar = tqdm(range(args.num_global_rounds), desc="Training [FedAvg]", file=sys.stdout, colour='green')

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []

        users_per_edge = args.num_users // args.num_edge_servers
        for i in range(args.num_edge_servers):
            assigned_clients = clients[i * users_per_edge: (i + 1) * users_per_edge]

            # 掉队率
            client_times = [c.simulate_physical_time(param_dim) for c in assigned_clients]
            total_times = [c[0] + c[1] for c in client_times]

            # 计算需要成功接收的保底设备数
            success_rate = 1.0 - args.target_straggler_rate
            K_min = max(1, int(len(assigned_clients) * success_rate))
            # 动态时间窗口受限于排序后的时间和绝对死线
            dynamic_t_win = min(args.t_deadline, sorted(total_times)[K_min - 1])

            valid_grads = []
            for idx, client in enumerate(assigned_clients):
                t_total = total_times[idx]
                # 只有未掉队的设备才会被记录 (含空口噪声)
                if t_total <= dynamic_t_win:
                    grad = client.train(global_weights, add_noise=True)
                    valid_grads.append(grad)

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
    history = run_fedavg()
    epochs = range(1, args.num_global_rounds + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0].plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    axes[0].set_title('FedAvg Accuracy (Train vs Val)')
    axes[0].legend();
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[1].plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    axes[1].set_title('FedAvg Loss (Train vs Val)')
    axes[1].legend();
    axes[1].grid(True)
    plt.show()