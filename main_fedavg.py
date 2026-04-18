import torch
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import args
from dataset import get_dataset, split_data
from models.network import SimpleCNN, CNNCifar,VGG11CIFAR,VGG13CIFAR,VGG16CIFAR,VGG19CIFAR
from utils import flatten_params, unflatten_params
from nodes.client import LocalClient
from nodes.edge import EdgeServer

# 🌟 【修复】：定义固定随机种子的函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def run_fedavg(skip_train_eval=False):
    # 🌟 【核心修复】：在每次运行算法的最开始，强制同步随机数宇宙！
    set_seed(42)

    print("\n>>> 正在进行仿真: Method = FedAvg")
    train_data, test_data = get_dataset()
    user_groups = split_data(train_data, args.num_users)

    if args.dataset_name == 'cifar10':
        global_model = VGG11CIFAR().to(args.device)
    else:
        global_model = SimpleCNN().to(args.device)

    param_dim = flatten_params(global_model).numel()

    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]
    edge_servers = []
    users_per_edge = args.num_users // args.num_edge_servers
    for i in range(args.num_edge_servers):
        edge_servers.append(EdgeServer(i, clients[i * users_per_edge: (i + 1) * users_per_edge], param_dim))

    t_acc_hist, t_loss_hist, v_acc_hist, v_loss_hist = [], [], [], []
    pbar = tqdm(range(1, args.num_global_rounds + 1), desc="Training [FedAvg]", ncols=100, file=sys.stdout)

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()

        edge_grads = []
        edge_weights = []  # 记录有效客户端的数量进行加权

        for edge_server in edge_servers:
            valid_grads = []

            client_conditions = [client.simulate_physical_conditions(param_dim) for client in edge_server.clients]
            times = [c[0] + c[1] for c in client_conditions]

            survival_rate = 1.0 - args.target_straggler_rate
            k_min_idx = max(1, int(len(edge_server.clients) * survival_rate)) - 1
            t_win = min(args.t_deadline, sorted(times)[k_min_idx])

            for local_idx, client in enumerate(edge_server.clients):
                t_train, t_up = client_conditions[local_idx][:2]
                if (t_train + t_up) <= t_win:
                    valid_grads.append(client.train(global_weights))

            if valid_grads:
                edge_grads.append(torch.stack(valid_grads).sum(dim=0))  # 边缘改为求和
                edge_weights.append(len(valid_grads))

        if edge_grads:
            #根据全网的有效客户端数量进行加权平均
            total_grad_sum = sum(edge_grads)
            total_weights = sum(edge_weights)
            global_grad = total_grad_sum / total_weights
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

        # 🌟 【修复】：解开学习率衰减注释，防止后期震荡
        if epoch == int(args.num_global_rounds * 0.5) or epoch == int(args.num_global_rounds * 0.75):
           args.lr *= 0.1

    return t_acc_hist, t_loss_hist, v_acc_hist, v_loss_hist

if __name__ == '__main__':
    # ... 单独运行画图逻辑保留 ...
    pass