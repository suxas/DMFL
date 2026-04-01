import torch
import torch.nn as nn
from config import args


class GradientDiffusion(nn.Module):
    def __init__(self, param_dim, hidden_dim, timesteps):
        super().__init__()
        self.timesteps = timesteps

        # 输入维度: x_t(param_dim) + t(1) + condition(param_dim)
        self.net = nn.Sequential(
            nn.Linear(param_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim)
        )

        # 预计算 Beta Schedule
        self.betas = torch.linspace(1e-4, 2e-2, timesteps).to(args.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def forward(self, x, t, condition):
        t_in = t.float().view(-1, 1) / self.timesteps
        return self.net(torch.cat((x, t_in, condition), dim=1))

    def p_sample(self, x, t, condition, t_index):
        """反向采样一步 (移除了多余的 model 参数)"""
        beta_t = self.betas[t_index]
        sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas[t_index])
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t_index]

        model_mean = sqrt_recip_alpha * (
                x - beta_t * self.forward(x, t, condition) / sqrt_one_minus_alpha_cumprod
        )

        if t_index == 0:
            return model_mean

        posterior_variance = beta_t * (1. - self.alphas_cumprod[t_index - 1]) / (1. - self.alphas_cumprod[t_index])
        return model_mean + torch.sqrt(posterior_variance) * torch.randn_like(x)

    def generate(self, condition):
        """生成补足梯度"""
        x = torch.randn_like(condition)
        self.eval()
        with torch.no_grad():
            for i in reversed(range(self.timesteps)):
                t = torch.full((1,), i, device=args.device, dtype=torch.long)
                x = self.p_sample(x, t, condition, i)
        self.train()
        return x

    def train_step(self, target_grad, condition_grad):
        """训练扩散模型"""
        self.train()
        batch_size = target_grad.shape[0]

        t = torch.randint(0, self.timesteps, (batch_size,), device=args.device).long()
        noise = torch.randn_like(target_grad)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)

        # 前向加噪
        x_t = sqrt_alpha * target_grad + sqrt_one_minus_alpha * noise

        # 预测噪声并计算损失
        return nn.MSELoss()(self.forward(x_t, t, condition_grad), noise)    #扩散模型使用MSE