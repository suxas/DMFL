import torch
import torch.optim as optim
import random
import numpy as np
from config import args
from models.diffusion import GradientDiffusion


class EdgeServer:
    def __init__(self, id, assigned_clients, input_dim):
        self.id = id
        self.clients = assigned_clients

        # 只预测分类头 (Head) 的参数 (最后510个参数)
        # 这种 "Head-Only" 策略最稳健
        self.diffusion_dim = 510

        self.diffusion = GradientDiffusion(
            param_dim=self.diffusion_dim,
            hidden_dim=args.diff_hidden_dim,
            timesteps=args.diff_timesteps
        ).to(args.device)

        #SGD优化器
        self.diff_optimizer = optim.SGD(
            self.diffusion.parameters(),
            lr=args.diff_lr,
            momentum=0.9,  # 推荐加入动量
            weight_decay=5e-4  # SGD通常配合一定的权重衰减防止过拟合
        )

        # 历史梯度
        self.historical_grads = {
            i: torch.zeros(input_dim).to(args.device)
            for i in range(len(assigned_clients))
        }
        self.current_round = 0

    def aggregate(self, global_weights, enable_diffusion=False):
        valid_grads = []
        self.current_round += 1

        is_warmup = self.current_round <= args.warmup_rounds

        for local_idx, client in enumerate(self.clients):
            is_straggler = random.random() < args.straggler_prob

            if not is_straggler:
                # --- 正常更新 ---
                grad = client.train(global_weights)
                valid_grads.append(grad)

                # --- 训练扩散模型 (Residual Mode) ---
                if enable_diffusion and torch.norm(self.historical_grads[local_idx]) > 0:
                    # 目标：预测 "当前梯度" 与 "历史梯度" 的差值 (Delta)
                    # 这样扩散模型只需要学习 "变化量"，比学习全量更容易
                    current_head = grad[-self.diffusion_dim:]
                    history_head = self.historical_grads[local_idx][-self.diffusion_dim:]

                    target_delta = current_head - history_head

                    # Condition 就是历史头部
                    loss = self.diffusion.train_step(
                        target_delta.unsqueeze(0),
                        history_head.unsqueeze(0)
                    )
                    self.diff_optimizer.zero_grad()
                    loss.backward()
                    self.diff_optimizer.step()

                self.historical_grads[local_idx] = grad.detach().clone()

            else:
                # --- 掉队处理 ---
                if enable_diffusion and not is_warmup:
                    # 1. 取出历史梯度 (Base)
                    base_grad = self.historical_grads[local_idx].clone()

                    if torch.norm(base_grad) > 0:
                        # 2. 用扩散模型预测 "变化量" (Correction)
                        history_head = base_grad[-self.diffusion_dim:].unsqueeze(0)
                        predicted_delta = self.diffusion.generate(history_head).squeeze(0)

                        # 3. 安全检查与缩放 (防止噪声爆炸)
                        # 如果预测的变化量比历史梯度本体还大，说明预测失控了，强制缩小
                        delta_norm = torch.norm(predicted_delta)
                        base_norm = torch.norm(history_head)
                        if delta_norm > base_norm:
                            predicted_delta = predicted_delta * (base_norm / (delta_norm + 1e-6)) * 0.5

                        # 4. 组合: Final = History + Predicted_Delta
                        # 即使 predicted_delta 是垃圾，我们至少还有 History (FedAsync 效果)
                        # 这保证了蓝线至少不会比 "完全瞎猜" 差
                        base_grad[-self.diffusion_dim:] += predicted_delta

                        valid_grads.append(base_grad)
                    else:
                        pass  # 无历史记录，无法补偿
                else:
                    pass  # Baseline 行为：丢弃

        if len(valid_grads) == 0:
            return None
        return torch.stack(valid_grads).mean(dim=0)