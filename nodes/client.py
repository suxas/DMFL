import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
import random
from config import args
from utils import flatten_params


class LocalClient:
    def __init__(self, dataset, idxs, model):
        self.ldr_train = DataLoader(Subset(dataset, list(idxs)), batch_size=args.batch_size, shuffle=True)
        self.model = copy.deepcopy(model)
        self.dataset_len = len(idxs)

        # 初始化物理属性 (随机分配计算能力)
        self.f_cpu = random.uniform(args.f_cpu_min, args.f_cpu_max)

    def simulate_physical_time(self, param_dim):
        """计算当前轮次的通信与计算总用时，供边缘服务器评估是否掉队"""
        rate = random.uniform(1e6, 5e6)  # 随机速率 1-5 Mbps
        grad_size_bits = param_dim * 32

        t_up = grad_size_bits / rate
        t_train = (args.num_local_epochs * args.cycles_per_sample * self.dataset_len) / self.f_cpu

        return t_train, t_up

    def train(self, global_weights, add_noise=True):
        """本地训练并返回计算出的梯度"""
        self.model.load_state_dict(global_weights)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
        initial_params = flatten_params(self.model).detach().clone()

        # 执行本地 Epochs
        for epoch in range(args.num_local_epochs):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                output = self.model(images)
                loss = nn.CrossEntropyLoss()(output, labels)
                loss.backward()
                optimizer.step()

        final_params = flatten_params(self.model).detach().clone()
        grad_vec = initial_params - final_params

        # 模拟 6G 通信噪声的附加
        if add_noise:
            noise_std = random.uniform(0.5, 1.5) * args.noise_scale
            channel_noise = torch.randn_like(grad_vec) * noise_std
            grad_vec = grad_vec + channel_noise

        return grad_vec