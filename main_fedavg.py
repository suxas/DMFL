import torch
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import args
from dataset import get_mnist_data, split_data
from models.network import SimpleCNN
from utils import flatten_params, unflatten_params
from nodes.client import LocalClient
from nodes.edge import EdgeServer


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


def run_fedavg():
    print("\n>>> 正在进行仿真: Method = fedavg")
    train_data, test_data = get_mnist_data()
    user_groups = split_data(train_data, args.num_users)

    global_model = SimpleCNN().to(args.device)
    param_dim = flatten_params(global_model).numel()

    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]
    edge_servers = []
    users_per_edge = args.num_users // args.num_edge_servers
    for i in range(args.num_edge_servers):
        edge_servers.append(EdgeServer(i, clients[i * users_per_edge: (i + 1) * users_per_edge], param_dim))

    t_acc_hist, t_loss_hist, v_acc_hist, v_loss_hist = [], [], [], []
    pbar = tqdm(range(args.num_global_rounds), desc="Training [fedavg]", ncols=100, file=sys.stdout)

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []

        for edge_server in edge_servers:
            valid_grads = []

            # --- 原聚合函数中的掉队判断逻辑 ---
            client_conditions = [client.simulate_physical_conditions(param_dim) for client in edge_server.clients]
            total_times = [c[0] + c[1] for c in client_conditions]

            K_min = max(2, int(len(edge_server.clients) * 0.5))
            sorted_times = sorted(total_times)
            dynamic_t_win = min(args.t_deadline, sorted_times[K_min - 1])

            for local_idx, client in enumerate(edge_server.clients):
                t_train, t_up, e_comp, e_comm = client_conditions[local_idx]
                t_total = t_train + t_up
                is_straggler = (t_total > dynamic_t_win)

                if not is_straggler:  # FedAvg 只接收非掉队设备
                    grad = client.train(global_weights)
                    valid_grads.append(grad)

            agg_grad = torch.stack(valid_grads).mean(dim=0) if len(valid_grads) > 0 else None
            if agg_grad is not None:
                edge_grads.append(agg_grad)

        if edge_grads:
            global_grad = torch.stack(edge_grads).mean(dim=0)
            new_params = flatten_params(global_model) - global_grad
            unflatten_params(global_model, new_params)

        train_acc, train_loss = evaluate_model(global_model, train_data)
        val_acc, val_loss = evaluate_model(global_model, test_data)

        t_acc_hist.append(train_acc)
        t_loss_hist.append(train_loss)
        v_acc_hist.append(val_acc)
        v_loss_hist.append(val_loss)
        pbar.set_postfix({'Val Acc': f"{val_acc:.2f}%", 'Val Loss': f"{val_loss:.4f}"})

    return t_acc_hist, t_loss_hist, v_acc_hist, v_loss_hist


if __name__ == '__main__':
    t_acc, t_loss, v_acc, v_loss = run_fedavg()
    epochs = range(1, args.num_global_rounds + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, t_acc, 'b--o', label='Train Accuracy')
    axes[0].plot(epochs, v_acc, 'r-^', label='Validation Accuracy')
    axes[0].set_title('FedAvg: Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, t_loss, 'b--o', label='Train Loss')
    axes[1].plot(epochs, v_loss, 'r-^', label='Validation Loss')
    axes[1].set_title('FedAvg: Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()