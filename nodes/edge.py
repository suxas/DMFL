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
        self.w_hist = 5
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
        K_min = max(2, int(len(self.clients) * 0.5))
        sorted_times = sorted(total_times)
        dynamic_t_win = min(args.t_deadline, sorted_times[K_min - 1])

        round_energy = 0.0
        max_time_in_round = 0.0

        for local_idx, client in enumerate(self.clients):
            t_train, t_up, e_comp, e_comm = client_conditions[local_idx]
            t_total = t_train + t_up

            is_straggler = (t_total > dynamic_t_win)

            if not is_straggler:
                # 记录 UE 端有效能耗和时间
                round_energy += (e_comp + e_comm)
                max_time_in_round = max(max_time_in_round, t_total)

                grad = client.train(global_weights)
                valid_grads.append(grad)

                if enable_diffusion and torch.norm(self.historical_grads[local_idx]) > 0:
                    current_head = grad[-self.diffusion_dim:]
                    history_head = self.historical_grads[local_idx][-self.diffusion_dim:]
                    target_delta = current_head - history_head

                    self.replay_buffer_history.append(history_head.detach().clone())
                    self.replay_buffer_target.append(target_delta.detach().clone())

                self.historical_grads[local_idx] = grad.detach().clone()

            else:
                if method == 'salf':
                    allowable_train_time = max(0, dynamic_t_win - t_up)
                    ratio = min(1.0, allowable_train_time / t_train) if t_train > 0 else 0
                    round_energy += (e_comp * ratio + e_comm)
                    max_time_in_round = max(max_time_in_round, dynamic_t_win)

                    full_grad = client.train(global_weights)
                    partial_grad = get_salf_partial_grad(global_model, full_grad)

                    valid_grads.append(partial_grad)
                    self.historical_grads[local_idx] = full_grad.detach().clone()

                elif enable_diffusion and not is_warmup:
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
                        # 推断产生微小系统延时
                        max_time_in_round = max(max_time_in_round, dynamic_t_win + 0.05)
                else:
                    max_time_in_round = max(max_time_in_round, dynamic_t_win)

        if enable_diffusion and len(self.replay_buffer_history) > 0:
            if len(self.replay_buffer_history) > self.max_buffer_size:
                self.replay_buffer_history = self.replay_buffer_history[-self.max_buffer_size:]
                self.replay_buffer_target = self.replay_buffer_target[-self.max_buffer_size:]

            hist_tensor = torch.stack(self.replay_buffer_history)
            targ_tensor = torch.stack(self.replay_buffer_target)
            dataset = TensorDataset(targ_tensor, hist_tensor)

            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            self.diffusion.train()
            e_diff = 5

            for _ in range(e_diff):
                for batch_targ, batch_hist in dataloader:
                    loss = self.diffusion.train_step(batch_targ, batch_hist)
                    self.diff_optimizer.zero_grad()
                    loss.backward()
                    self.diff_optimizer.step()

            # 增加扩散模型训练用时
            n_samples = len(self.replay_buffer_history)
            t_diff = n_samples * e_diff * 0.0005
            max_time_in_round += t_diff

        agg_grad = torch.stack(valid_grads).mean(dim=0) if len(valid_grads) > 0 else None
        return agg_grad, max_time_in_round, round_energy