import torch
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random      # 新增
import numpy as np # 新增

# 固定随机种子的函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 强制 cuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


from config import args
from dataset import get_dataset, split_data
from models.network import SimpleCNN, CNNCifar
from models.diffusion import GradientDiffusion
from utils import flatten_params, unflatten_params
from nodes.client import LocalClient
from nodes.edge import EdgeServer


class EdgeState:
    def __init__(self, num_clients, param_dim, diff_dim):
        # CIFAR-10 网络 CNNCifar 最后一层 (fc3) 参数量: 84 * 10 + 10 = 850
        self.diff_dim = diff_dim
        self.diffusion = GradientDiffusion(self.diff_dim, args.diff_hidden_dim, args.diff_timesteps).to(args.device)
        self.optimizer = optim.SGD(self.diffusion.parameters(), lr=args.diff_lr, momentum=args.momentum,
                                   weight_decay=5e-4)

        self.hist_grads = {i: torch.zeros(param_dim).to(args.device) for i in range(num_clients)}
        self.buf_hist, self.buf_targ = [], []
        self.max_buf = num_clients * 5      #经验池数量


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
    print("\n>>> 正在进行仿真: Method = DMFL")
    train_data, test_data = get_dataset()
    user_groups = split_data(train_data, args.num_users)

    if args.dataset_name == 'cifar10':
        global_model = CNNCifar().to(args.device)
        diff_dim = 510
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

        for server in edge_servers:
            state = states[server.id]
            valid_grads = []

            times = [sum(c.simulate_physical_conditions(param_dim)[:2]) for c in server.clients]
            t_win = min(args.t_deadline, sorted(times)[max(2, int(len(server.clients) * 0.5)) - 1])

            for i, client in enumerate(server.clients):
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
                            delta = delta * (norm_b / (norm_d + 1e-6)) * 0.8   #MNIST情况下要*0.5，CIFAR*0.8

                        base_grad[-state.diff_dim:] += delta
                        valid_grads.append(base_grad)

            if state.buf_hist:
                state.buf_hist = state.buf_hist[-state.max_buf:]
                state.buf_targ = state.buf_targ[-state.max_buf:]

                dataset = TensorDataset(torch.stack(state.buf_targ), torch.stack(state.buf_hist))
                loader = DataLoader(dataset, batch_size=32, shuffle=True)

                state.diffusion.train()
                for _ in range(10):  # 扩撒模型训练次数
                    for targ, hist in loader:
                        loss = state.diffusion.train_step(targ, hist)
                        state.optimizer.zero_grad()
                        loss.backward()
                        state.optimizer.step()

            if valid_grads:
                edge_grads.append(torch.stack(valid_grads).mean(dim=0))

        if edge_grads:
            global_grad = torch.stack(edge_grads).mean(dim=0)
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

    return t_acc, t_loss, v_acc, v_loss

if __name__ == '__main__':
    # 独立运行时，默认计算训练集 (skip_train_eval=False)
    t_acc, t_loss, v_acc, v_loss = run_dmfl(skip_train_eval=False)
    epochs = range(1, args.num_global_rounds + 1)
    ms = 2  # 缩小描点体积

    # 绘制独立运行时的 Train vs Validation 对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 精度图
    axes[0].plot(epochs, t_acc, 'b--o', markersize=ms, label='Train Accuracy')
    axes[0].plot(epochs, v_acc, 'r-^', markersize=ms, label='Validation Accuracy')
    if args.warmup_rounds > 0:
        axes[0].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[0].set_title(f'DMFL ({args.dataset_name.upper()}): Accuracy')
    axes[0].set_xlabel('Global Communication Rounds')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True)

    # Loss图
    axes[1].plot(epochs, t_loss, 'b--o', markersize=ms, label='Train Loss')
    axes[1].plot(epochs, v_loss, 'r-^', markersize=ms, label='Validation Loss')
    if args.warmup_rounds > 0:
        axes[1].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[1].set_title(f'DMFL ({args.dataset_name.upper()}): Loss')
    axes[1].set_xlabel('Global Communication Rounds')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()