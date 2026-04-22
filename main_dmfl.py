import torch
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from config import args
from dataset import get_dataset, split_data
from models.network import SimpleCNN, CNNCifar, VGG11CIFAR, VGG13CIFAR, VGG16CIFAR, VGG19CIFAR
from models.diffusion import GradientDiffusion
from utils import flatten_params, unflatten_params
from nodes.client import LocalClient
from nodes.edge import EdgeServer


class EdgeState:
    def __init__(self, num_clients, param_dim, diff_dim):
        self.diff_dim = diff_dim
        self.diffusion = GradientDiffusion(self.diff_dim, args.diff_hidden_dim, args.diff_timesteps).to(args.device)
        self.optimizer = optim.SGD(self.diffusion.parameters(), lr=args.diff_lr, momentum=args.momentum,
                                   weight_decay=5e-4)

        self.hist_grads = {i: torch.zeros(param_dim).to(args.device) for i in range(num_clients)}
        self.buf_hist, self.buf_targ = [], []
        self.max_buf = num_clients * 5


def evaluate(model, dataset):
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


def run_dmfl(skip_train_eval=False):
    set_seed(42)

    print("\n>>> 正在进行仿真: Method = DMFL")
    train_data, test_data = get_dataset()
    user_groups = split_data(train_data, args.num_users)

    if args.dataset_name == 'cifar10':
        global_model = VGG11CIFAR().to(args.device)
        diff_dim = 5130
    else:
        global_model = SimpleCNN().to(args.device)
        diff_dim = 510

    param_dim = flatten_params(global_model).numel()

    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]
    edge_servers = []
    u_per_edge = args.num_users // args.num_edge_servers
    for i in range(args.num_edge_servers):
        edge_servers.append(EdgeServer(i, clients[i * u_per_edge: (i + 1) * u_per_edge], param_dim))

    states = {s.id: EdgeState(len(s.clients), param_dim, diff_dim) for s in edge_servers}
    t_acc, t_loss, v_acc, v_loss = [], [], [], []

    pbar = tqdm(range(1, args.num_global_rounds + 1), desc="Training [DMFL]", ncols=100, file=sys.stdout)

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()

        edge_grads = []
        edge_weights = []

        for edge_server in edge_servers:
            state = states[edge_server.id]
            valid_grads = []

            times = [sum(c.simulate_physical_conditions(param_dim)[:2]) for c in edge_server.clients]

            # 使用您提供的完全一致的计算公式
            survival_rate = 1.0 - args.target_straggler_rate
            k_min_idx = max(1, int(len(edge_server.clients) * survival_rate)) - 1
            t_win = min(args.t_deadline, sorted(times)[k_min_idx])

            for i, client in enumerate(edge_server.clients):
                if times[i] <= t_win:
                    grad = client.train(global_weights)
                    valid_grads.append(grad)

                    if torch.norm(state.hist_grads[i]) > 0:
                        head_curr, head_hist = grad[-state.diff_dim:], state.hist_grads[i][-state.diff_dim:]
                        state.buf_hist.append(head_hist.detach().clone())
                        state.buf_targ.append((head_curr - head_hist).detach().clone())
                    state.hist_grads[i] = grad.detach().clone()

                elif epoch > args.warmup_rounds:
                    base_grad = state.hist_grads[i].clone()
                    if torch.norm(base_grad) > 0:
                        head_hist = base_grad[-state.diff_dim:].unsqueeze(0)
                        delta = state.diffusion.generate(head_hist).squeeze(0)

                        norm_d, norm_b = torch.norm(delta), torch.norm(head_hist)
                        if norm_d > norm_b:
                            delta = delta * (norm_b / (norm_d + 1e-6)) * 0.8

                        base_grad[-state.diff_dim:] += delta
                        valid_grads.append(base_grad)

            if state.buf_hist:
                state.buf_hist = state.buf_hist[-state.max_buf:]
                state.buf_targ = state.buf_targ[-state.max_buf:]

                dataset = TensorDataset(torch.stack(state.buf_targ), torch.stack(state.buf_hist))
                loader = DataLoader(dataset, batch_size=32, shuffle=True)

                state.diffusion.train()
                for _ in range(10):
                    for targ, hist in loader:
                        loss = state.diffusion.train_step(targ, hist)
                        state.optimizer.zero_grad()
                        loss.backward()
                        state.optimizer.step()

            if valid_grads:
                edge_grads.append(torch.stack(valid_grads).sum(dim=0))
                edge_weights.append(len(valid_grads))

        if edge_grads:
            total_grad_sum = sum(edge_grads)
            total_weights = sum(edge_weights)

            global_grad = total_grad_sum / total_weights
            unflatten_params(global_model, flatten_params(global_model) - global_grad)

        acc_v, loss_v = evaluate(global_model, test_data)
        v_acc.append(acc_v)
        v_loss.append(loss_v)

        if not skip_train_eval:
            acc_t, loss_t = evaluate(global_model, train_data)
            t_acc.append(acc_t)
            t_loss.append(loss_t)
        else:
            t_acc.append(0.0)
            t_loss.append(0.0)

        pbar.set_postfix({'Val Acc': f"{acc_v:.2f}%", 'Val Loss': f"{loss_v:.4f}"})

        # 学习率衰减
        # if epoch == int(args.num_global_rounds * 0.5) or epoch == int(args.num_global_rounds * 0.75):
        #     args.lr *= 0.1

    return t_acc, t_loss, v_acc, v_loss


# 🌟 添加绘图辅助函数（放在 if __name__ == '__main__': 上方即可）
def plot_with_shadow(ax, x, y, color, label, window=20):
    import numpy as np
    y_arr = np.array(y)
    y_mean = np.zeros_like(y_arr)
    y_std = np.zeros_like(y_arr)
    for i in range(len(y_arr)):
        start = max(0, i - window // 2)
        end = min(len(y_arr), i + window // 2 + 1)
        y_mean[i] = np.mean(y_arr[start:end])
        y_std[i] = np.std(y_arr[start:end])

    ax.plot(x, y_mean, color=color, linestyle='-', label=label, linewidth=2.0)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)


if __name__ == '__main__':
    # 注意：在不同文件里把 run_xxx 改成本文件的函数名（run_dmfl / run_fedavg / run_salf）
    # 以下以 run_dmfl 为例，若是其他文件请自行修改为对应的运行函数
    t_acc, t_loss, v_acc, v_loss = run_dmfl(skip_train_eval=False)

    epochs = range(1, args.num_global_rounds + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 精度图
    plot_with_shadow(axes[0], epochs, t_acc, 'blue', 'Train Accuracy')
    plot_with_shadow(axes[0], epochs, v_acc, 'red', 'Validation Accuracy')
    if args.warmup_rounds > 0:
        axes[0].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[0].set_title(f'Performance ({args.dataset_name.upper()}): Accuracy')
    axes[0].set_xlabel('Global Communication Rounds')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True)

    # Loss 图
    plot_with_shadow(axes[1], epochs, t_loss, 'blue', 'Train Loss')
    plot_with_shadow(axes[1], epochs, v_loss, 'red', 'Validation Loss')
    if args.warmup_rounds > 0:
        axes[1].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[1].set_title(f'Performance ({args.dataset_name.upper()}): Loss')
    axes[1].set_xlabel('Global Communication Rounds')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()