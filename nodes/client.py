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

        # 初始化设备物理属性
        # CPU 频率，用于计算设备的计算时间与计算能耗
        self.f_cpu = random.uniform(args.f_cpu_min, args.f_cpu_max)

    def simulate_physical_conditions(self, param_dim):
        # 随机生成一个客观的传输速率 (1 Mbps - 5 Mbps)
        rate = random.uniform(1e6, 5e6)

        # 梯度大小 S(w) (位 = 参数量 * 32 float)
        grad_size_bits = param_dim * 32

        # 1. 传输时间与计算时间
        t_up = grad_size_bits / rate
        t_train = (args.num_local_epochs * args.cycles_per_sample * self.dataset_len) / self.f_cpu

        # 2. 传输耗能与计算耗能
        e_comm = args.p_ue_max * t_up
        e_comp = args.kappa * (self.f_cpu ** 2) * (args.num_local_epochs * args.cycles_per_sample * self.dataset_len)

        return t_train, t_up, e_comp, e_comm

    def train(self, global_weights):
        self.model.load_state_dict(global_weights)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

        initial_params = flatten_params(self.model).detach().clone()

        # 本地训练
        for epoch in range(args.num_local_epochs):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                output = self.model(images)
                loss = nn.CrossEntropyLoss()(output, labels)
                loss.backward()
                optimizer.step()

        final_params = flatten_params(self.model).detach().clone()

        # 返回梯度向量
        grad_vec = initial_params - final_params

        # 随机噪声
        base_noise_scale = getattr(args, 'noise_scale', 0.02)
        noise_std = random.uniform(0.5, 1.5) * base_noise_scale
        channel_noise = torch.randn_like(grad_vec) * noise_std

        # 噪声附加在成功计算的梯度上
        grad_vec = grad_vec + channel_noise

        return grad_vec