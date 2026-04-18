import torch
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import args
from dataset import get_dataset, split_data
from models.network import SimpleCNN, CNNCifar, VGG11CIFAR, VGG13CIFAR, VGG16CIFAR, VGG19CIFAR
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


def get_salf_partial_grad(model, full_grad_vec):
    param_sizes = [p.numel() for p in model.parameters()]
    num_of_layers = len(param_sizes)
    # SALF 原版逻辑：随机选择保留最后 up_to_layer 层
    up_to_layer = np.random.randint(1, num_of_layers + 1)
    layers_to_zero = num_of_layers - up_to_layer
    num_zeros = sum(param_sizes[:layers_to_zero])

    salf_grad_vec = full_grad_vec.clone()
    # 丢弃的前半部分层，梯度置为 0
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
    set_seed(42)

    print(f"\n>>> 正在进行仿真: Method = SALF ({args.dataset_name.upper()})")
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
    pbar = tqdm(range(1, args.num_global_rounds + 1), desc="Training [SALF]", ncols=100, file=sys.stdout)

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()

        edge_grads = []
        edge_weights = []

        for edge_server in edge_servers:
            valid_grads = []

            client_conditions = [client.simulate_physical_conditions(param_dim) for client in edge_server.clients]
            times = [c[0] + c[1] for c in client_conditions]

            survival_rate = 1.0 - args.target_straggler_rate
            k_min_idx = max(1, int(len(edge_server.clients) * survival_rate)) - 1
            t_win = min(args.t_deadline, sorted(times)[k_min_idx])

            for local_idx, client in enumerate(edge_server.clients):
                t_train, t_up = client_conditions[local_idx][:2]
                full_grad = client.train(global_weights)

                if (t_train + t_up) <= t_win:
                    valid_grads.append(full_grad)
                else:
                    partial_grad = get_salf_partial_grad(global_model, full_grad)
                    valid_grads.append(partial_grad)

            if valid_grads:
                edge_grads.append(torch.stack(valid_grads).sum(dim=0))
                edge_weights.append(len(valid_grads))

        if edge_grads:
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

        # 学习率衰减
        if epoch == int(args.num_global_rounds * 0.5) or epoch == int(args.num_global_rounds * 0.75):
            args.lr *= 0.1

    return t_acc_hist, t_loss_hist, v_acc_hist, v_loss_hist

if __name__ == '__main__':
    pass