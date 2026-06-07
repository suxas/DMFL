import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import args
from dataset import get_dataset, split_data
from models.network import SimpleCNN, VGG11CIFAR
from utils import set_seed, flatten_params, unflatten_params, evaluate
from nodes.client import LocalClient
from nodes.edge import EdgeServer


class FedProxClient(LocalClient):

    def train(self, global_weights, mu=0.01):
        self.model.load_state_dict(global_weights)
        self.model.train()
        opt = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)

        # 缓存全局模型参数
        global_params = [p.detach().clone() for p in self.model.parameters()]
        # 记录用于计算上传更新量
        p0 = flatten_params(self.model).detach().clone()

        for _ in range(args.num_local_epochs):
            for imgs, labels in self.loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                opt.zero_grad()

                loss = nn.CrossEntropyLoss()(self.model(imgs), labels)
                # 向 loss 中加入 Proximal Term
                if mu > 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_params):
                        proximal_term += torch.sum((w - w_t) ** 2)
                    loss += (mu / 2.0) * proximal_term

                loss.backward()
                opt.step()

        # 返回参数的更新量
        grad = p0 - flatten_params(self.model).detach().clone()
        if args.noise_scale > 0:
            grad = grad + torch.randn_like(grad) * args.noise_scale
        return grad


def run_fedprox(skip_train_eval=False, mu=0.01):
    set_seed(42)
    print(f"\n>>> 运行 FedProx 算法 ({args.dataset_name.upper()}, mu={mu})")
    train_ds, test_ds = get_dataset()
    groups = split_data(train_ds, args.num_users)

    model = VGG11CIFAR().to(args.device) if args.dataset_name == 'cifar10' else \
        SimpleCNN().to(args.device)
    p_dim = flatten_params(model).numel()

    clients = [FedProxClient(train_ds, groups[i], model) for i in range(args.num_users)]
    u_per = args.num_users // args.num_edge_servers
    servers = [EdgeServer(i, clients[i * u_per:(i + 1) * u_per], p_dim)
               for i in range(args.num_edge_servers)]

    t_acc, t_loss, v_acc, v_loss = [], [], [], []
    pbar = tqdm(range(1, args.num_global_rounds + 1), desc="FedProx", ncols=100, file=sys.stdout)

    for rnd in pbar:
        model.train()
        w_global = model.state_dict()
        edge_grads = []

        for srv in servers:
            times = [sum(c.sim_time(p_dim)) for c in srv.clients]
            k = max(1, int(len(srv.clients) * (1.0 - args.target_straggler_rate))) - 1
            t_win = min(args.t_deadline, sorted(times)[k])

            # 使用传入的 mu 值进行本地训练
            valid = [c.train(w_global, mu=mu) for c, t in zip(srv.clients, times) if t <= t_win]
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