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

    print("\n>>> 正在进行仿真: Method = DMFL (基于先验 IID 比例的层级异构重建机制)")
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

        for edge_server in edge_servers:
            state = states[edge_server.id]

            # 分离存活者和掉队者的索引
            active_grads = []
            straggler_indices = []

            times = [sum(c.simulate_physical_conditions(param_dim)[:2]) for c in edge_server.clients]
            survival_rate = 1.0 - args.target_straggler_rate
            k_min_idx = max(1, int(len(edge_server.clients) * survival_rate)) - 1
            t_win = min(args.t_deadline, sorted(times)[k_min_idx])

            # 1. 优先收集本轮成功上传的客户端真实梯度
            for i, client in enumerate(edge_server.clients):
                if times[i] <= t_win:
                    grad = client.train(global_weights)
                    active_grads.append(grad)

                    # 更新历史经验池用于训练扩散模型
                    if torch.norm(state.hist_grads[i]) > 0:
                        head_curr, head_hist = grad[-state.diff_dim:], state.hist_grads[i][-state.diff_dim:]
                        state.buf_hist.append(head_hist.detach().clone())
                        state.buf_targ.append((head_curr - head_hist).detach().clone())
                    state.hist_grads[i] = grad.detach().clone()
                else:
                    straggler_indices.append(i)

            # 2. 计算本轮簇内其他客户端的识别层（特征提取器）平均梯度
            if len(active_grads) > 0:
                avg_active_feat = torch.stack([g[:-diff_dim] for g in active_grads]).mean(dim=0)
            else:
                avg_active_feat = torch.zeros(param_dim - diff_dim).to(args.device)

            # 3. 开始重建掉队者，并组装全簇的梯度集合
            edge_all_grads = list(active_grads)  # 首先把真实梯度放进大池子

            for i in straggler_indices:
                base_grad = state.hist_grads[i].clone()

                # 如果度过了预热期，且该掉队者有过历史记录，则进行高级伪造
                if epoch > args.warmup_rounds and torch.norm(base_grad) > 0:

                    # --- A. 分类头：使用扩散模型预测 ---
                    head_hist = base_grad[-diff_dim:].unsqueeze(0)
                    delta = state.diffusion.generate(head_hist).squeeze(0)
                    norm_d, norm_b = torch.norm(delta), torch.norm(head_hist)
                    if norm_d > norm_b:
                        delta = delta * (norm_b / (norm_d + 1e-6)) * 0.8
                    fake_head = base_grad[-diff_dim:] + delta

                    # --- B. 识别层：通过 IID 程度进行插值混合 ---
                    hist_feat = base_grad[:-diff_dim]
                    iid_deg = args.inner_client_iid

                    # 只有当簇内有存活者时，才能混合；否则 100% 相信历史
                    if len(active_grads) > 0:
                        fake_feat = (iid_deg * avg_active_feat) + ((1.0 - iid_deg) * hist_feat)
                    else:
                        fake_feat = hist_feat

                    # --- C. 拼接并加入全簇大池子 ---
                    fake_grad = torch.cat([fake_feat, fake_head])
                    edge_all_grads.append(fake_grad)

                else:
                    # 如果在预热期，直接用历史梯度垫底 (如果没有历史记录，base_grad 本身就是全 0)
                    edge_all_grads.append(base_grad)

            # 扩散模型训练 (逻辑不变)
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

            # 4. 小基站内聚合：直接将全簇 (存活+重建) 梯度求和
            if len(edge_all_grads) > 0:
                server_grad_sum = torch.stack(edge_all_grads).sum(dim=0)
                edge_grads.append(server_grad_sum)

        if edge_grads:
            # 5. 大基站聚合：严格使用“全网所有客户端数量”作为分母
            total_grad_sum = sum(edge_grads)
            global_grad = total_grad_sum / float(args.num_users)
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

        # if epoch == int(args.num_global_rounds * 0.5) or epoch == int(args.num_global_rounds * 0.75):
        # args.lr *= 0.1

    return t_acc, t_loss, v_acc, v_loss


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
    t_acc, t_loss, v_acc, v_loss = (run_dmfl(skip_train_eval=False))

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