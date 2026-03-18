import torch
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config import args
from models.diffusion import GradientDiffusion
from SALF import get_salf_partial_grad


class EdgeServer:
    def __init__(self, id, assigned_clients, input_dim):
        self.id = id
        self.clients = assigned_clients
        self.param_dim = input_dim

        self.diffusion_dim = 510
        self.diffusion = GradientDiffusion(
            param_dim=self.diffusion_dim,
            hidden_dim=args.diff_hidden_dim,
            timesteps=args.diff_timesteps
        ).to(args.device)

        self.diff_optimizer = optim.SGD(
            self.diffusion.parameters(),
            lr=args.diff_lr,
            momentum=args.momentum,
            weight_decay=5e-4
        )
        self.historical_grads = {
            i: torch.zeros(input_dim).to(args.device)
            for i in range(len(assigned_clients))
        }
        self.current_round = 0

        # 经验回放池
        self.replay_buffer_history = []
        self.replay_buffer_target = []
        self.w_hist = 5  # 假设滑动窗口保存过去 5 轮的数据
        self.max_buffer_size = len(assigned_clients) * self.w_hist

    def aggregate(self, global_weights, method='baseline', global_model=None):
        valid_grads = []
        self.current_round += 1

        enable_diffusion = (method == 'dmfl')
        is_warmup = self.current_round <= args.warmup_rounds

        # 模拟物理层状态评估
        client_conditions = [client.simulate_physical_conditions(self.param_dim) for client in self.clients]
        total_times = [c[0] + c[1] for c in client_conditions]

        # 设定动态接收窗口 Twin
        K_min = max(2, int(len(self.clients) * 0.5))  # 至少收集50%的梯度
        sorted_times = sorted(total_times)
        dynamic_t_win = min(args.t_deadline, sorted_times[K_min - 1])

        round_energy = 0.0
        max_time_in_round = 0.0

        for local_idx, client in enumerate(self.clients):
            # 因为 client.py 废弃了精准 snr 计算，这里用 _ 忽略它
            t_train, t_up, e_comp, e_comm = client_conditions[local_idx]
            t_total = t_train + t_up

            # 判断掉队状态：超过动态时间窗口
            is_straggler = (t_total > dynamic_t_win)

            if not is_straggler:
                # 记录有效能耗和时间
                round_energy += (e_comp + e_comm)
                max_time_in_round = max(max_time_in_round, t_total)

                grad = client.train(global_weights)

                valid_grads.append(grad)

                if enable_diffusion and torch.norm(self.historical_grads[local_idx]) > 0:
                    current_head = grad[-self.diffusion_dim:]
                    history_head = self.historical_grads[local_idx][-self.diffusion_dim:]
                    target_delta = current_head - history_head

                    # 经验回放池
                    self.replay_buffer_history.append(history_head.detach().clone())
                    self.replay_buffer_target.append(target_delta.detach().clone())

                self.historical_grads[local_idx] = grad.detach().clone()

            else:
                if method == 'salf':
                    # SALF 将允许计算部分梯度，所以产生部分计算能耗
                    allowable_train_time = max(0, dynamic_t_win - t_up)
                    ratio = min(1.0, allowable_train_time / t_train) if t_train > 0 else 0
                    round_energy += (e_comp * ratio + e_comm)
                    max_time_in_round = max(max_time_in_round, dynamic_t_win)

                    full_grad = client.train(global_weights)
                    partial_grad = get_salf_partial_grad(global_model, full_grad)

                    valid_grads.append(partial_grad)
                    self.historical_grads[local_idx] = full_grad.detach().clone()

                elif enable_diffusion and not is_warmup:
                    # 扩散模型产生额外的计算补偿时间与能耗
                    base_grad = self.historical_grads[local_idx].clone()
                    if torch.norm(base_grad) > 0:
                        history_head = base_grad[-self.diffusion_dim:].unsqueeze(0)
                        predicted_delta = self.diffusion.generate(history_head).squeeze(0)

                        delta_norm = torch.norm(predicted_delta)
                        base_norm = torch.norm(history_head)
                        if delta_norm > base_norm:
                            predicted_delta = predicted_delta * (base_norm / (delta_norm + 1e-6)) * 0.5

                        base_grad[-self.diffusion_dim:] += predicted_delta
                        valid_grads.append(base_grad)

                        # 补偿计算微小能耗
                        round_energy += (0.05 * e_comp)
                        max_time_in_round = max(max_time_in_round, dynamic_t_win + 0.05)
                else:
                    max_time_in_round = max(max_time_in_round, dynamic_t_win)

        # 滑动时间窗口
        if enable_diffusion and len(self.replay_buffer_history) > 0:
            # 1. 维护滑动时间窗口大小 W_hist
            if len(self.replay_buffer_history) > self.max_buffer_size:
                self.replay_buffer_history = self.replay_buffer_history[-self.max_buffer_size:]
                self.replay_buffer_target = self.replay_buffer_target[-self.max_buffer_size:]

            # 2. 构建 Dataset 并通过 DataLoader 划分 Batch
            hist_tensor = torch.stack(self.replay_buffer_history)
            targ_tensor = torch.stack(self.replay_buffer_target)
            dataset = TensorDataset(targ_tensor, hist_tensor)

            # batch_size 本地集中训练
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            self.diffusion.train()
            e_diff = 5  # 本地训练 Epoch 数

            for _ in range(e_diff):
                for batch_targ, batch_hist in dataloader:
                    loss = self.diffusion.train_step(batch_targ, batch_hist)
                    self.diff_optimizer.zero_grad()
                    loss.backward()
                    self.diff_optimizer.step()

            # 3. 记录基站本地集中训练产生的额外耗时与功耗
            n_samples = len(self.replay_buffer_history)
            t_diff = n_samples * e_diff * 0.0005  # 假设单样本单次计算耗时 0.5 毫秒
            max_time_in_round += t_diff
            round_energy += (t_diff * 15.0)  # 假设小基站工作功率为 15W

        agg_grad = torch.stack(valid_grads).mean(dim=0) if len(valid_grads) > 0 else None
        return agg_grad, max_time_in_round, round_energy