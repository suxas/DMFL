import sys
import torch
from tqdm import tqdm

from config import args
from dataset import get_dataset, split_data
from models.network import SimpleCNN, VGG11CIFAR
from utils import set_seed, flatten_params, unflatten_params, evaluate
from nodes.client import LocalClient
from nodes.edge import EdgeServer


def run_fedavg(skip_train_eval=False):
    set_seed(42)
    print(f"\n>>> FedAvg 仿真 ({args.dataset_name.upper()})")
    train_ds, test_ds = get_dataset()
    groups = split_data(train_ds, args.num_users)

    model = VGG11CIFAR().to(args.device) if args.dataset_name == 'cifar10' else \
        SimpleCNN().to(args.device)
    p_dim = flatten_params(model).numel()

    clients = [LocalClient(train_ds, groups[i], model) for i in range(args.num_users)]
    u_per = args.num_users // args.num_edge_servers
    servers = [EdgeServer(i, clients[i * u_per:(i + 1) * u_per], p_dim)
               for i in range(args.num_edge_servers)]

    t_acc, t_loss, v_acc, v_loss = [], [], [], []
    pbar = tqdm(range(1, args.num_global_rounds + 1), desc="FedAvg", ncols=100, file=sys.stdout)

    for rnd in pbar:
        model.train()
        w_global = model.state_dict()
        edge_grads = []

        for srv in servers:
            times = [sum(c.sim_time(p_dim)) for c in srv.clients]
            k = max(1, int(len(srv.clients) * (1.0 - args.target_straggler_rate))) - 1
            t_win = min(args.t_deadline, sorted(times)[k])

            valid = [c.train(w_global) for c, t in zip(srv.clients, times) if t <= t_win]
            if valid:
                edge_grads.append(torch.stack(valid).sum(dim=0))

        if edge_grads:
            g_global = sum(edge_grads) / args.num_users
            unflatten_params(model, flatten_params(model) - g_global)

        acc_v, loss_v = evaluate(model, test_ds, args.batch_size)
        v_acc.append(acc_v)
        v_loss.append(loss_v)

        if not skip_train_eval:
            acc_t, loss_t = evaluate(model, train_ds, args.batch_size)
            t_acc.append(acc_t)
            t_loss.append(loss_t)
        else:
            t_acc.append(0.0)
            t_loss.append(0.0)

        pbar.set_postfix({'Val Acc': f'{acc_v:.2f}%', 'Val Loss': f'{loss_v:.4f}'})

    return t_acc, t_loss, v_acc, v_loss
