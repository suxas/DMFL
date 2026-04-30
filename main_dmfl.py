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

# 引入纯数学优化的求解器
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
    # 重置随机种子，确保开启/关闭优化器时，遇到的信道状态和数据划分绝对一致
    set_seed(42)

    mode_str = "带 CR-SQP 优化器" if use_optimizer else "无优化器 (满功率+常规截断)"
    print(f"\n>>> 正在进行仿真: DMFL - {mode_str} (Dataset: {args.dataset_name.upper()})")

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

    # 记录物理资源分配轨迹
    history_avg_P_UE = []
    history_avg_T_win = []

    best_val_acc = 0.0
    best_round = 0

    pbar = tqdm(range(1, args.num_global_rounds + 1), desc=f"Training", ncols=110, file=sys.stdout)

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()

        edge_grads = []
        edge_weights = []

        # ==========================================================
        # 1. 提取当前真实物理环境状态
        # ==========================================================
        N_sbs = args.num_edge_servers
        K_users = args.num_users // args.num_edge_servers

        T_train_mat = np.zeros((N_sbs, K_users))
        h_sq_mat = np.zeros((N_sbs, K_users))
        D_nk_mat = np.zeros((N_sbs, K_users))
        T_diff_mat = np.ones(N_sbs) * 0.5

        for n, server in enumerate(edge_servers):
            for k, client in enumerate(server.clients):
                t_train, _ = client.simulate_physical_conditions(param_dim)
                T_train_mat[n, k] = t_train
                # 模拟当前客观存在的瑞利信道衰落
                h_sq_mat[n, k] = np.random.exponential(scale=1.0)
                D_nk_mat[n, k] = client.dataset_len

        current_state = {
            'T_train': T_train_mat, 'h_sq': h_sq_mat,
            'D_nk': D_nk_mat, 'T_diff': T_diff_mat
        }

        # ==========================================================
        # 2. 物理资源分配决策 (使用控制变量法)
        # ==========================================================
        if use_optimizer:
            # 开启优化器：动态计算最优功率和截断时间
            plot_flag = (epoch <= 3)
            opt_P_UE, opt_T_win, final_state = run_resource_optimization(
                N=N_sbs, K=K_users, param_dim=param_dim, current_state=current_state, device=args.device
            )
            avg_P = np.mean(opt_P_UE)
            avg_T = np.mean(opt_T_win)
        else:
            # 不用优化器（基准线）：全员最大功率，采用排序淘汰机制决定截断时间
            final_state = np.zeros((N_sbs, K_users))
            avg_P = args.p_ue_max
            T_win_list = []

            for n in range(N_sbs):
                times = []
                for k in range(K_users):
                    # 按照香农公式计算满功率情况下的所需时间
                    SNR = (args.p_ue_max * h_sq_mat[n, k]) / (getattr(args, 'noise_scale', 0.05) ** 2)
                    R_UL = (args.bandwidth / K_users) * np.log2(1.0 + SNR)
                    T_up = (param_dim * 32) / (R_UL + 1e-9)
                    times.append(T_train_mat[n, k] + T_up)

                # 强制要求一定比例设备成功
                K_min = max(1, int(K_users * (1.0 - getattr(args, 'target_straggler_rate', 0.5))))
                t_win = min(getattr(args, 't_deadline', 1.0), sorted(times)[K_min - 1])
                T_win_list.append(t_win)

                # 生成判定状态
                for k in range(K_users):
                    if times[k] <= t_win:
                        final_state[n, k] = 1

            avg_T = np.mean(T_win_list)

        history_avg_P_UE.append(avg_P)
        history_avg_T_win.append(avg_T)
        tqdm.write(f"[资源调度 R{epoch}] 平均分配功率: {avg_P:.4f} W, 平均截断窗口: {avg_T:.4f} s")

        # ==========================================================
        # 3. 严格依据物理判定结果(final_state)执行联邦训练与扩散代偿
        # ==========================================================
        for n, edge_server in enumerate(edge_servers):
            state = states[edge_server.id]
            valid_grads = []

            for k, client in enumerate(edge_server.clients):
                if final_state[n, k] == 1:
                    # 允许上传
                    grad = client.train(global_weights)
                    valid_grads.append(grad)

                    if torch.norm(state.hist_grads[k]) > 0:
                        head_curr = grad[-state.diff_dim:]
                        head_hist = state.hist_grads[k][-state.diff_dim:]
                        state.buf_hist.append(head_hist.detach().clone())
                        state.buf_targ.append((head_curr - head_hist).detach().clone())
                    state.hist_grads[k] = grad.detach().clone()

                elif epoch > args.warmup_rounds:
                    # 被调度器或客观环境强制截断，调用扩散代偿
                    base_grad = state.hist_grads[k].clone()
                    if torch.norm(base_grad) > 0:
                        head_hist = base_grad[-state.diff_dim:].unsqueeze(0)
                        delta = state.diffusion.generate(head_hist).squeeze(0)

                        norm_d, norm_b = torch.norm(delta), torch.norm(head_hist)
                        if norm_d > norm_b:
                            delta = delta * (norm_b / (norm_d + 1e-6)) * 0.8

                        base_grad[-state.diff_dim:] += delta
                        valid_grads.append(base_grad)

            # 扩散模型训练逻辑
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

        # ==========================================================
        # 4. 宏基站全局聚合
        # ==========================================================
        if edge_grads:
            total_grad_sum = sum(edge_grads)
            total_weights = sum(edge_weights)
            global_grad = total_grad_sum / total_weights
            unflatten_params(global_model, flatten_params(global_model) - global_grad)

        acc_v, loss_v = evaluate(global_model, test_data)
        v_acc.append(acc_v)
        v_loss.append(loss_v)

        if acc_v > best_val_acc:
            best_val_acc = acc_v
            best_round = epoch

        if not skip_train_eval:
            acc_t, loss_t = evaluate(global_model, train_data)
            t_acc.append(acc_t)
            t_loss.append(loss_t)
        else:
            t_acc.append(0.0)
            t_loss.append(0.0)

        pbar.set_postfix({'Val Acc': f"{acc_v:.2f}%", 'Best': f"{best_val_acc:.2f}%"})

    print(f"\n✅ 训练结束! 最佳测试集精度: {best_val_acc:.2f}% (第 {best_round} 轮)")
    return t_acc, t_loss, v_acc, v_loss, history_avg_P_UE, history_avg_T_win


if __name__ == '__main__':
    # ==========================================================
    # 实验执行：先后进行带优化器与不带优化器的对比运行
    # ==========================================================
    print("==========================================================")
    print("  实验一：运行【带 CR-SQP 优化器】的 DMFL 系统")
    print("==========================================================")
    res_opt = run_dmfl(use_optimizer=True, skip_train_eval=True)
    _, _, v_acc_opt, v_loss_opt, hist_P_opt, hist_T_opt = res_opt

    print("\n==========================================================")
    print("  实验二：运行【无优化器 (满功率+常规截断)】的 DMFL 系统")
    print("==========================================================")
    res_base = run_dmfl(use_optimizer=False, skip_train_eval=True)
    _, _, v_acc_base, v_loss_base, hist_P_base, hist_T_base = res_base

    # ==========================================================
    # 绘制对比图表
    # ==========================================================
    epochs = range(1, args.num_global_rounds + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 图 1：测试集精度对比 ---
    axes[0].plot(epochs, v_acc_base, 'r--o', markersize=3, label='DMFL w/o Optimizer (Baseline)')
    axes[0].plot(epochs, v_acc_opt, 'g-^', markersize=3, label='DMFL with CR-SQP Optimizer')
    if args.warmup_rounds > 0:
        axes[0].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[0].set_title(f'Test Accuracy Comparison ({args.dataset_name.upper()})')
    axes[0].set_xlabel('Global Communication Rounds')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True)

    # --- 图 2：测试集 Loss 对比 ---
    axes[1].plot(epochs, v_loss_base, 'r--o', markersize=3, label='DMFL w/o Optimizer')
    axes[1].plot(epochs, v_loss_opt, 'g-^', markersize=3, label='DMFL with CR-SQP Optimizer')
    if args.warmup_rounds > 0:
        axes[1].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[1].set_title('Test Loss Comparison')
    axes[1].set_xlabel('Global Communication Rounds')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    # --- 图 3：优化器的物理资源分配动作轨迹 (双 Y 轴) ---
    color_p = 'tab:blue'
    axes[2].set_xlabel('Global Communication Rounds')
    axes[2].set_ylabel('Avg Allocation Power (W)', color=color_p)
    axes[2].plot(epochs, hist_P_opt, color=color_p, marker='o', markersize=4, linestyle='-', label='P_UE (W)')
    axes[2].tick_params(axis='y', labelcolor=color_p)

    ax3_twin = axes[2].twinx()
    color_t = 'tab:red'
    ax3_twin.set_ylabel('Avg Time Window (s)', color=color_t)
    ax3_twin.plot(epochs, hist_T_opt, color=color_t, marker='^', markersize=4, linestyle='--', label='T_win (s)')
    ax3_twin.tick_params(axis='y', labelcolor=color_t)

    axes[2].set_title('CR-SQP Resource Allocation Trajectory')
    fig.tight_layout()
    plt.show()