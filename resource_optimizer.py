import torch
import torch.nn as nn
import torch.optim as optim
from config import args

class DMFL_ResourceOptimizer(nn.Module):
    def __init__(self, N, K, param_dim):
        super(DMFL_ResourceOptimizer, self).__init__()
        self.N = N
        self.K = K

        # --- 物理约束与超参数 ---
        self.P_max = args.p_ue_max
        self.T_deadline = getattr(args, 't_deadline', 1.0)
        self.target_straggler_rate = getattr(args, 'target_straggler_rate', 0.5)
        self.K_min = max(1, int(K * (1.0 - self.target_straggler_rate)))
        self.sigma_noise_sq = getattr(args, 'noise_scale', 0.05) ** 2
        self.B = args.bandwidth / K
        self.S_omega = param_dim * 32

        self.Delta_diff_sq = getattr(args, 'delta_diff_sq', 0.05)
        self.rho_energy = 0.5  # 能耗惩罚权重：促使功率下降
        self.rho_time = 0.5    # 时间惩罚权重：在保证精度的前提下，逼迫 T_win 尽可能缩小

        # 利用sigmod函数留出优化空间
        self.raw_P_UE = nn.Parameter(torch.ones(N, K) * 2.0)
        self.raw_T_win = nn.Parameter(torch.ones(N) * 2.0)

    def get_actual_variables(self):
        # 使用 Sigmoid 保证严格在 (0, max) 之间，且全域可导，彻底解决 clamp 导致的梯度为 0 问题
        P_UE = self.P_max * torch.sigmoid(self.raw_P_UE) + 1e-6
        T_win = self.T_deadline * torch.sigmoid(self.raw_T_win) + 1e-6
        return P_UE, T_win

    def forward(self, T_train, h_sq, D_nk, T_diff, gamma):
        P_UE, T_win = self.get_actual_variables()

        # 物理计算：香农公式求传输时延
        SNR = (P_UE * h_sq) / self.sigma_noise_sq
        R_UL = self.B * torch.log2(1.0 + SNR)
        T_up = self.S_omega / (R_UL + 1e-9)
        T_req = T_train + T_up


        # 步骤 1：连续松弛
        T_win_expanded = T_win.unsqueeze(1).expand(self.N, self.K)
        gap = (T_win_expanded - T_req) / (self.T_deadline + 1e-6)
        State = torch.sigmoid(gamma * gap)
        # 1. 误差上界 L_bound
        comm_noise = self.sigma_noise_sq / (h_sq * P_UE + 1e-9)
        E_n_k = State * comm_noise + (1.0 - State) * (self.Delta_diff_sq + 1.0)
        E_n = torch.mean(E_n_k, dim=1)
        D_n = torch.sum(D_nk, dim=1)
        weights = D_n / (torch.sum(D_n) + 1e-9)
        L_bound = torch.sum(weights * E_n)

        # 2. 通信能耗 E_total
        E_energy_nk = State * (P_UE * T_up)
        E_total = torch.mean(torch.sum(E_energy_nk, dim=1))

        # 3. 时间窗口惩罚
        T_penalty = torch.mean(T_win)

        # 4. K_min 基础底线惩罚
        succ_devices_per_SBS = torch.sum(State, dim=1)
        penalty_Kmin = torch.sum(torch.relu(self.K_min - succ_devices_per_SBS) ** 2)

        loss = L_bound + self.rho_energy * E_total + self.rho_time * T_penalty + 10.0 * penalty_Kmin
        return loss


def run_resource_optimization(N, K, param_dim, current_state, device):
    T_train = torch.tensor(current_state['T_train'], dtype=torch.float32, device=device)
    h_sq = torch.tensor(current_state['h_sq'], dtype=torch.float32, device=device)
    D_nk = torch.tensor(current_state['D_nk'], dtype=torch.float32, device=device)
    T_diff = torch.tensor(current_state['T_diff'], dtype=torch.float32, device=device)

    model = DMFL_ResourceOptimizer(N, K, param_dim).to(device)
    # 步骤 2：序列二次规划寻优 (SQP)
    # 赋予足够大的学习率，让变量能迅速脱离初始状态
    optimizer = optim.Adam([
        {'params': model.raw_P_UE, 'lr': 0.1},
        {'params': model.raw_T_win, 'lr': 0.1}
    ])

    max_iter = 150
    model.train()
    for i in range(max_iter):
        optimizer.zero_grad()
        # 步骤 3：线性稳健退火
        gamma_current = 1.0 + (50.0 - 1.0) * (i / max_iter)
        loss = model(T_train, h_sq, D_nk, T_diff, gamma_current)
        loss.backward()
        optimizer.step()

    # 步骤 4：离散投影
    model.eval()
    with torch.no_grad():
        opt_P_UE, opt_T_win = model.get_actual_variables()
        R_UL = (args.bandwidth / K) * torch.log2(1.0 + (opt_P_UE * h_sq) / (getattr(args, 'noise_scale', 0.05) ** 2))
        T_up = (param_dim * 32) / (R_UL + 1e-9)
        T_req = T_train + T_up
        final_state = (T_req <= opt_T_win.unsqueeze(1).expand(N, K)).float()

    return opt_P_UE.cpu().numpy(), opt_T_win.cpu().numpy(), final_state.cpu().numpy()