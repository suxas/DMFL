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

        # 🌟 优化博弈权重 (调整策略以保精度为主)
        self.Delta_diff_sq = getattr(args, 'delta_diff_sq', 0.05)
        self.rho_energy = 0.5  # 降低能耗权重，防止过度压低功率导致通信噪声过大
        self.tau_time = 0.3    # 新增：时间惩罚权重。赋予等待时间代价，激活 T_win 的动态变化

        # 初始化：T_win 初始化为中间值，给予其上下浮动的空间
        self.raw_P_UE = nn.Parameter(torch.ones(N, K) * self.P_max)
        self.raw_T_win = nn.Parameter(torch.ones(N) * (self.T_deadline * 0.8))

    def get_actual_variables(self):
        # 严格控制在物理域内
        P_UE = torch.clamp(self.raw_P_UE, min=1e-6, max=self.P_max)
        T_win = torch.clamp(self.raw_T_win, min=1e-6, max=self.T_deadline)
        return P_UE, T_win

    def forward(self, T_train, h_sq, D_nk, T_diff, gamma):
        P_UE, T_win = self.get_actual_variables()

        # 物理计算：时延
        SNR = (P_UE * h_sq) / self.sigma_noise_sq
        R_UL = self.B * torch.log2(1.0 + SNR)
        T_up = self.S_omega / (R_UL + 1e-9)
        T_req = T_train + T_up

        # 连续松弛
        T_win_expanded = T_win.unsqueeze(1).expand(self.N, self.K)
        gap = (T_win_expanded - T_req) / (self.T_deadline + 1e-6)
        State = torch.sigmoid(gamma * gap)

        # 1. 误差上界 L_bound (真实数据极度重要，掉队惩罚设为常数 1.0 压制通信噪声)
        comm_noise = self.sigma_noise_sq / (h_sq * P_UE + 1e-9)
        E_n_k = State * comm_noise + (1.0 - State) * 1.0
        E_n = torch.mean(E_n_k, dim=1)
        D_n = torch.sum(D_nk, dim=1)
        weights = D_n / (torch.sum(D_n) + 1e-9)
        L_bound = torch.sum(weights * E_n)

        # 2. 通信能耗 E_total
        E_energy_nk = State * (P_UE * T_up)
        E_total = torch.mean(torch.sum(E_energy_nk, dim=1))

        # 3. K_min 底线惩罚
        succ_devices_per_SBS = torch.sum(State, dim=1)
        penalty_Kmin = torch.sum(torch.relu(self.K_min - succ_devices_per_SBS) ** 2)

        # 🌟 最终目标：重塑 3D 博弈 (误差 + 能耗 + 时间)
        loss = L_bound + self.rho_energy * E_total + self.tau_time * torch.mean(T_win) + 10.0 * penalty_Kmin
        return loss

def run_resource_optimization(N, K, param_dim, current_state, device):
    T_train = torch.tensor(current_state['T_train'], dtype=torch.float32, device=device)
    h_sq = torch.tensor(current_state['h_sq'], dtype=torch.float32, device=device)
    D_nk = torch.tensor(current_state['D_nk'], dtype=torch.float32, device=device)
    T_diff = torch.tensor(current_state['T_diff'], dtype=torch.float32, device=device)

    model = DMFL_ResourceOptimizer(N, K, param_dim).to(device)

    # 给 T_win 更大的学习率，让其有足够的动力去寻找最优窗口
    optimizer = optim.Adam([
        {'params': model.raw_P_UE, 'lr': 0.05},
        {'params': model.raw_T_win, 'lr': 0.05}
    ])

    max_iter = 150
    model.train()
    for i in range(max_iter):
        optimizer.zero_grad()
        gamma_current = 1.0 + (50.0 - 1.0) * (i / max_iter)
        loss = model(T_train, h_sq, D_nk, T_diff, gamma_current)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        opt_P_UE, opt_T_win = model.get_actual_variables()
        R_UL = (args.bandwidth / K) * torch.log2(1.0 + (opt_P_UE * h_sq) / (getattr(args, 'noise_scale', 0.05) ** 2))
        T_up = (param_dim * 32) / (R_UL + 1e-9)
        T_req = T_train + T_up
        final_state = (T_req <= opt_T_win.unsqueeze(1).expand(N, K)).float()

    return opt_P_UE.cpu().numpy(), opt_T_win.cpu().numpy(), final_state.cpu().numpy()