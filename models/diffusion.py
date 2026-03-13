import torch
import torch.nn as nn
from config import args


class GradientDiffusion(nn.Module):
    def __init__(self, param_dim, hidden_dim, timesteps):
        super(GradientDiffusion, self).__init__()
        self.param_dim = param_dim
        self.timesteps = timesteps

        # 输入维度: x_t(param_dim) + t(1) + condition(param_dim)
        self.net = nn.Sequential(
            nn.Linear(param_dim + 1 + param_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim)
        )

        # 预计算 Beta Schedule
        self.betas = torch.linspace(0.0001, 0.02, timesteps).to(args.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def forward(self, x, t, condition):
        # 归一化时间步
        t_in = t.float().view(-1, 1) / self.timesteps
        x_in = torch.cat((x, t_in, condition), dim=1)
        return self.net(x_in)

    def p_sample(self, model, x, t, condition, t_index):
        """反向采样一步"""
        beta_t = self.betas[t_index]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t_index])

        model_mean = sqrt_recip_alpha_t * (
                x - beta_t * model(x, t, condition) / sqrt_one_minus_alpha_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance = beta_t * (1. - self.alphas_cumprod[t_index - 1]) / (1. - self.alphas_cumprod[t_index])
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance) * noise

    def generate(self, condition):
        """生成补足梯度"""
        x = torch.randn_like(condition).to(args.device)
        self.eval()
        with torch.no_grad():
            for i in reversed(range(self.timesteps)):
                t = torch.full((1,), i, device=args.device, dtype=torch.long)
                x = self.p_sample(self.forward, x, t, condition, i)
        self.train()
        return x

    def train_step(self, target_grad, condition_grad):
        """训练扩散模型"""
        self.train()

        # 1. 动态获取当前 batch 的大小
        batch_size = target_grad.shape[0]

        # 2. 为 batch 中的每个样本独立采样一个时间步，大小变为 (batch_size,)
        t = torch.randint(0, self.timesteps, (batch_size,), device=args.device).long()
        noise = torch.randn_like(target_grad)

        # Forward Process
        x_start = target_grad

        # 3. 提取对应时间步的系数，并增加一个维度 (变成 [batch_size, 1]) 以便与 [batch_size, param_dim] 相乘
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)

        x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

        # Predict Noise
        predicted_noise = self.forward(x_t, t, condition_grad)
        return nn.MSELoss()(predicted_noise, noise)