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
        rate = random.uniform(1e6, 5e6)
        t_up = (param_dim * 32) / rate
        actual_samples = min(self.dataset_len, getattr(args, 'local_iterations', 9999) * args.batch_size)
        t_train = (args.num_local_epochs * args.cycles_per_sample * actual_samples) / self.f_cpu
        return t_train, t_up

    def train(self, global_weights):
        self.model.load_state_dict(global_weights)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)

        initial_params = flatten_params(self.model).detach().clone()

        for _ in range(args.num_local_epochs):
            for images, labels in self.ldr_train:
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(images), labels)  # 本地客户端使用交叉熵损失
                loss.backward()
                optimizer.step()

        grad_vec = initial_params - flatten_params(self.model).detach().clone()

        noise_std = args.noise_scale
        if noise_std > 0:
            return grad_vec + torch.randn_like(grad_vec) * noise_std
        return grad_vec