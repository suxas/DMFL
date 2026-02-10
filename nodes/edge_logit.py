import torch
import torch.optim as optim
import random
from collections import deque
from config import args
from models.logit import LogitLSTM


class LogitEdgeServer:
    def __init__(self, id, assigned_clients, input_dim):
        self.id = id
        self.clients = assigned_clients

        # 策略保持与 DMFL 一致：只预测分类头 (Head) 的参数
        # 硬编码 510 以匹配 SimpleCNN 的全连接层参数量 (500 weights + 10 bias)
        self.target_dim = 510

        # LOGIT 模型: LSTM
        # 使用 args.diff_hidden_dim 保证与 Diffusion 模型规模相当
        self.model = LogitLSTM(
            input_dim=self.target_dim,
            hidden_dim=args.diff_hidden_dim,
            num_layers=1
        ).to(args.device)

        # 优化器配置保持与 DMFL 一致
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.diff_lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        # LOGIT 需要时间序列历史 (Sequence Length = 3)
        self.seq_len = 3

        # 为每个客户端维护一个滑动窗口历史
        self.history_buffers = {
            i: deque(maxlen=self.seq_len)
            for i in range(len(assigned_clients))
        }

        # 用于存储上一轮的完整梯度，以便在预测时作为 Base
        self.last_full_grads = {
            i: torch.zeros(input_dim).to(args.device)
            for i in range(len(assigned_clients))
        }

        self.current_round = 0
        self.criterion = torch.nn.MSELoss()

    def aggregate(self, global_weights, enable_logit=False):
        valid_grads = []
        self.current_round += 1
        # 预热期不进行预测
        is_warmup = self.current_round <= args.warmup_rounds

        for local_idx, client in enumerate(self.clients):
            # 模拟掉队: 使用 config.py 中的 straggler_prob
            is_straggler = random.random() < args.straggler_prob

            if not is_straggler:
                # === Case 1: 客户端在线 ===
                # 正常训练并获取梯度
                grad = client.train(global_weights)
                valid_grads.append(grad)

                # 获取梯度的 Head 部分用于训练
                current_head = grad[-self.target_dim:].detach().clone()

                # --- LOGIT 在线训练 ---
                # 只有当历史数据积累足够 (满足 seq_len) 时才训练
                if enable_logit and len(self.history_buffers[local_idx]) == self.seq_len:
                    # Input: 历史序列 [t-3, t-2, t-1]
                    # Shape: (1, 3, 510)
                    history_seq = torch.stack(list(self.history_buffers[local_idx])).unsqueeze(0)
                    # Target: 当前真实梯度 t
                    target = current_head.unsqueeze(0)

                    self.optimizer.zero_grad()
                    prediction = self.model(history_seq)
                    loss = self.criterion(prediction, target)
                    loss.backward()
                    self.optimizer.step()

                # 更新历史缓冲
                self.history_buffers[local_idx].append(current_head)
                self.last_full_grads[local_idx] = grad.detach().clone()

            else:
                # === Case 2: 客户端掉队 ===
                if enable_logit and not is_warmup and len(self.history_buffers[local_idx]) == self.seq_len:
                    # 准备历史数据
                    history_seq = torch.stack(list(self.history_buffers[local_idx])).unsqueeze(0)

                    # 使用 LSTM 预测下一时刻的 Head
                    self.model.eval()
                    with torch.no_grad():
                        predicted_head = self.model(history_seq).squeeze(0)
                    self.model.train()

                    # 组合: Base (上一轮完整梯度) + Correction (新预测的 Head)
                    # 这是一个简单的替换策略
                    proxy_grad = self.last_full_grads[local_idx].clone()

                    # 安全缩放 (防止预测值数值爆炸影响聚合)
                    pred_norm = torch.norm(predicted_head)
                    # 取最近一次真实梯度的范数作为参考
                    hist_norm = torch.norm(self.history_buffers[local_idx][-1])

                    # 如果预测值异常大，进行缩放
                    if pred_norm > hist_norm * 2.0:
                        predicted_head = predicted_head * (hist_norm / (pred_norm + 1e-6))

                    # 替换 Head 部分
                    proxy_grad[-self.target_dim:] = predicted_head
                    valid_grads.append(proxy_grad)

                    # 自回归更新: 将预测值加入历史，以便下一轮继续预测 (如果连续掉线)
                    self.history_buffers[local_idx].append(predicted_head)
                else:
                    # 预热期或无历史数据，直接丢弃 (Baseline 行为)
                    pass

        # 聚合本轮所有有效梯度 (包括预测的)
        if len(valid_grads) == 0:
            return None
        return torch.stack(valid_grads).mean(dim=0)