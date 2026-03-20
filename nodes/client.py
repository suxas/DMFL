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
        self.f_cpu = random.uniform(args.f_cpu_min, args.f_cpu_max)

    def simulate_physical_conditions(self, param_dim):
        """仅返回影响状态判断的时间参数，剔除多余功耗数据"""
        rate = random.uniform(1e6, 5e6)
        t_up = (param_dim * 32) / rate
        t_train = (args.num_local_epochs * args.cycles_per_sample * self.dataset_len) / self.f_cpu

        return t_train, t_up

    def train(self, global_weights):
        self.model.load_state_dict(global_weights)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

        initial_params = flatten_params(self.model).detach().clone()

        for _ in range(args.num_local_epochs):
            for images, labels in self.ldr_train:
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(images), labels)
                loss.backward()
                optimizer.step()

        grad_vec = initial_params - flatten_params(self.model).detach().clone()

        # 附加信道噪声
        noise_std = random.uniform(0.5, 1.5) * getattr(args, 'noise_scale', 0.02)
        return grad_vec + torch.randn_like(grad_vec) * noise_std