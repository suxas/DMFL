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

        # 误差界限与能耗权重
        self.Delta_diff_sq = getattr(args, 'delta_diff_sq', 0.05)
        self.rho_energy = 0.5  #  ##能耗惩罚权重## 调大它，系统更省电(倾向于截断)；调小它，系统更保精度(倾向于满功率)

        # 初始化：默认全员满功率，窗口给满，防止梯度过早消失
        self.raw_P_UE = nn.Parameter(torch.ones(N, K) * self.P_max)
        self.raw_T_win = nn.Parameter(torch.ones(N) * self.T_deadline)

    def get_actual_variables(self):
        #  硬截断：彻底抛弃 Sigmoid，消除梯度饱和，严格控制在物理域内
        P_UE = torch.clamp(self.raw_P_UE, min=1e-6, max=self.P_max)
        T_win = torch.clamp(self.raw_T_win, min=1e-6, max=self.T_deadline)
        return P_UE, T_win

    def forward(self, T_train, h_sq, D_nk, T_diff, gamma):
        P_UE, T_win = self.get_actual_variables()

        # 物理计算：香农公式求传输时延
        SNR = (P_UE * h_sq) / self.sigma_noise_sq
        R_UL = self.B * torch.log2(1.0 + SNR)
        T_up = self.S_omega / (R_UL + 1e-9)
        T_req = T_train + T_up

        # ==========================================================
        # 步骤 1：连续松弛 (Continuous Relaxation)
        # ==========================================================
        T_win_expanded = T_win.unsqueeze(1).expand(self.N, self.K)
        # 使用归一化 gap，防止 gamma 放大时引发数值溢出
        gap = (T_win_expanded - T_req) / (self.T_deadline + 1e-6)
        State = torch.sigmoid(gamma * gap)

        # ==========================================================
        # 新型优化目标构建
        # ==========================================================
        # 1. 误差上界 L_bound (放大截断惩罚，彰显真实 Non-IID 数据的重要性)
        comm_noise = self.sigma_noise_sq / (h_sq * P_UE + 1e-9)
        # 若截断，不仅承受扩散误差，还加上 1.0 的 Non-IID 缺失惩罚，逼迫优化器保住设备
        E_n_k = State * comm_noise + (1.0 - State) * (self.Delta_diff_sq + 1.0)
        E_n = torch.mean(E_n_k, dim=1)
        D_n = torch.sum(D_nk, dim=1)
        weights = D_n / (torch.sum(D_n) + 1e-9)
        L_bound = torch.sum(weights * E_n)

        # 2. 通信能耗 E_total (只有实际上传的设备才消耗能量)
        E_energy_nk = State * (P_UE * T_up)
        E_total = torch.mean(torch.sum(E_energy_nk, dim=1))

        # 3. K_min 基础底线惩罚
        succ_devices_per_SBS = torch.sum(State, dim=1)
        penalty_Kmin = torch.sum(torch.relu(self.K_min - succ_devices_per_SBS) ** 2)

        # 目标：最小化误差 + 最小化能耗
        loss = L_bound + self.rho_energy * E_total + 10.0 * penalty_Kmin
        return loss


def run_resource_optimization(N, K, param_dim, current_state, device):
    T_train = torch.tensor(current_state['T_train'], dtype=torch.float32, device=device)
    h_sq = torch.tensor(current_state['h_sq'], dtype=torch.float32, device=device)
    D_nk = torch.tensor(current_state['D_nk'], dtype=torch.float32, device=device)
    T_diff = torch.tensor(current_state['T_diff'], dtype=torch.float32, device=device)

    model = DMFL_ResourceOptimizer(N, K, param_dim).to(device)

    # ==========================================================
    # 步骤 2：序列二次规划寻优 (SQP)
    # ==========================================================
    optimizer = optim.Adam([
        {'params': model.raw_P_UE, 'lr': 0.05},
        {'params': model.raw_T_win, 'lr': 0.01}
    ])

    max_iter = 150
    model.train()
    for i in range(max_iter):
        optimizer.zero_grad()
        # ==========================================================
        # 步骤 3：退火逼近 (Annealing Strategy) - 改为线性稳健退火
        # ==========================================================
        gamma_current = 1.0 + (50.0 - 1.0) * (i / max_iter)
        loss = model(T_train, h_sq, D_nk, T_diff, gamma_current)
        loss.backward()
        optimizer.step()

    # ==========================================================
    # 步骤 4：离散投影 (Discrete Projection)
    # ==========================================================
    model.eval()
    with torch.no_grad():
        opt_P_UE, opt_T_win = model.get_actual_variables()
        R_UL = (args.bandwidth / K) * torch.log2(1.0 + (opt_P_UE * h_sq) / (getattr(args, 'noise_scale', 0.05) ** 2))
        T_up = (param_dim * 32) / (R_UL + 1e-9)
        T_req = T_train + T_up

        final_state = (T_req <= opt_T_win.unsqueeze(1).expand(N, K)).float()

    return opt_P_UE.cpu().numpy(), opt_T_win.cpu().numpy(), final_state.cpu().numpy()