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
from resource_optimizer import run_resource_optimization


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


def run_dmfl(use_optimizer=True, skip_train_eval=False):
    set_seed(42)  # 严格对齐随机环境

    mode_name = "CR-SQP(精度优先)" if use_optimizer else "Baseline(满功率截断)"
    print(f"\n>>> 启动仿真: DMFL - {mode_name} (Dataset: {args.dataset_name.upper()})")

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

    history_avg_P_UE = []
    history_avg_T_win = []

    pbar = tqdm(range(1, args.num_global_rounds + 1), desc=f"Training [{mode_name}]", ncols=110, file=sys.stdout)

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []
        edge_weights = []

        N_sbs = args.num_edge_servers
        K_users = args.num_users // args.num_edge_servers

        T_train_mat = np.zeros((N_sbs, K_users))
        h_sq_mat = np.zeros((N_sbs, K_users))
        D_nk_mat = np.zeros((N_sbs, K_users))
        T_diff_mat = np.ones(N_sbs) * 0.5

        for n, server in enumerate(edge_servers):
            for k, client in enumerate(server.clients):
                f_cpu = random.uniform(args.f_cpu_min, args.f_cpu_max)
                t_train = (args.cycles_per_sample * client.dataset_len * args.num_local_epochs) / f_cpu
                T_train_mat[n, k] = t_train
                h_sq_mat[n, k] = np.random.exponential(scale=1.0)
                D_nk_mat[n, k] = client.dataset_len

        current_state = {
            'T_train': T_train_mat, 'h_sq': h_sq_mat,
            'D_nk': D_nk_mat, 'T_diff': T_diff_mat
        }

        # --- 资源调度与物理判定 ---
        if use_optimizer:
            opt_P_UE, opt_T_win, final_state = run_resource_optimization(
                N=N_sbs, K=K_users, param_dim=param_dim, current_state=current_state, device=args.device
            )
        else:
            opt_P_UE = np.ones((N_sbs, K_users)) * args.p_ue_max
            opt_T_win = np.zeros(N_sbs)
            final_state = np.zeros((N_sbs, K_users))
            K_min = max(1, int(K_users * (1.0 - args.target_straggler_rate)))

            for n in range(N_sbs):
                times = []
                for k in range(K_users):
                    SNR = (args.p_ue_max * h_sq_mat[n, k]) / (args.noise_scale ** 2)
                    R_UL = (args.bandwidth / K_users) * np.log2(1.0 + SNR + 1e-9)
                    T_up = (param_dim * 32) / (R_UL + 1e-9)
                    times.append(T_train_mat[n, k] + T_up)

                t_win = min(args.t_deadline, sorted(times)[K_min - 1])
                opt_T_win[n] = t_win
                for k in range(K_users):
                    if times[k] <= t_win:
                        final_state[n, k] = 1

        # 打印日志与记录轨迹
        tqdm.write(f"\n--- [Round {epoch}] 物理资源分配日志 ({mode_name}) ---")
        for n in range(N_sbs):
            tqdm.write(f"  > SBS {n} 截断时间 T_win = {opt_T_win[n]:.4f} s")
            p_str = ", ".join([f"{p:.4f}" for p in opt_P_UE[n]])
            tqdm.write(f"    客户端发射功率 (W): [{p_str}]")

        history_avg_P_UE.append(np.mean(opt_P_UE))
        history_avg_T_win.append(np.mean(opt_T_win))

        # --- 执行本地训练与扩散代偿 (原生逻辑) ---
        for n, edge_server in enumerate(edge_servers):
            state = states[edge_server.id]
            valid_grads = []

            for k, client in enumerate(edge_server.clients):
                if final_state[n, k] == 1:
                    grad = client.train(global_weights)
                    valid_grads.append(grad)

                    if torch.norm(state.hist_grads[k]) > 0:
                        head_curr = grad[-state.diff_dim:]
                        head_hist = state.hist_grads[k][-state.diff_dim:]
                        state.buf_hist.append(head_hist.detach().clone())
                        state.buf_targ.append((head_curr - head_hist).detach().clone())
                    state.hist_grads[k] = grad.detach().clone()

                elif epoch > args.warmup_rounds:
                    base_grad = state.hist_grads[k].clone()
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

        # --- 全局聚合 ---
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

        pbar.set_postfix({'Val Acc': f"{acc_v:.2f}%"})

    return t_acc, t_loss, v_acc, v_loss, history_avg_P_UE, history_avg_T_win


# ... [保留原本所有的 DMFL 训练逻辑，从开头直到 run_dmfl 函数结束] ...

if __name__ == '__main__':
    print("  实验一：运行【带 CR-SQP 优化器】")
    res_opt = run_dmfl(use_optimizer=True, skip_train_eval=True)
    _, _, v_acc_opt, v_loss_opt, hist_P_opt, hist_T_opt = res_opt

    print("  实验二：运行【无优化器】")
    res_base = run_dmfl(use_optimizer=False, skip_train_eval=True)
    _, _, v_acc_base, v_loss_base, hist_P_base, hist_T_base = res_base

    epochs = range(1, args.num_global_rounds + 1)

    # 图 1：Accuracy 和 Loss 对比 (1x2 子图)
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

    # 子图 1: Accuracy
    axes1[0].plot(epochs, v_acc_base, color='#d62728', linestyle='--', label='Baseline (No Optimizer)')
    axes1[0].plot(epochs, v_acc_opt, color='#2ca02c', linestyle='-', label='DMFL + CR-SQP')
    if args.warmup_rounds > 0:
        axes1[0].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes1[0].set_title(f'Test Accuracy Comparison ({args.dataset_name.upper()})')
    axes1[0].set_xlabel('Global Communication Rounds')
    axes1[0].set_ylabel('Accuracy (%)')
    axes1[0].legend()
    axes1[0].grid(True, linestyle='--', alpha=0.6)

    # 子图 2: Loss
    axes1[1].plot(epochs, v_loss_base, color='#d62728', linestyle='--', label='Baseline (No Optimizer)')
    axes1[1].plot(epochs, v_loss_opt, color='#2ca02c', linestyle='-', label='DMFL + CR-SQP')
    if args.warmup_rounds > 0:
        axes1[1].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes1[1].set_title('Test Loss Comparison')
    axes1[1].set_xlabel('Global Communication Rounds')
    axes1[1].set_ylabel('Loss')
    axes1[1].legend()
    axes1[1].grid(True, linestyle='--', alpha=0.6)

    fig1.tight_layout()
    plt.show(block=False)  # 保持窗口开启，继续绘制下一张图

    # 图 2：物理资源分配动态轨迹 (双 Y 轴单图)
    fig2, ax_p = plt.subplots(figsize=(10, 6))

    color_p = '#1f77b4'
    ax_p.set_xlabel('Global Communication Rounds', fontsize=11)
    ax_p.set_ylabel('Avg Allocation Power (W)', color=color_p, fontsize=11)
    ax_p.plot(epochs, hist_P_opt, color=color_p, linestyle='-', marker='o', markersize=3, label='P_UE (W)')
    ax_p.tick_params(axis='y', labelcolor=color_p)
    ax_p.grid(True, linestyle='--', alpha=0.4)

    # 创建双 Y 轴
    ax_t = ax_p.twinx()
    color_t = '#ff7f0e'
    ax_t.set_ylabel('Avg Time Window (s)', color=color_t, fontsize=11)
    ax_t.plot(epochs, hist_T_opt, color=color_t, linestyle='-', marker='^', markersize=3, label='T_win (s)')
    ax_t.tick_params(axis='y', labelcolor=color_t)

    # 合并两个轴的图例
    lines_1, labels_1 = ax_p.get_legend_handles_labels()
    lines_2, labels_2 = ax_t.get_legend_handles_labels()
    ax_p.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('CR-SQP Dynamic Resource Allocation Trajectory', fontsize=12)
    fig2.tight_layout()
    plt.show()