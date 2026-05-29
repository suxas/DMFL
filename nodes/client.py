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
        self.loader = DataLoader(Subset(dataset, list(idxs)),
                                 batch_size=args.batch_size, shuffle=True)
        self.model = copy.deepcopy(model)
        self.n_samples = len(idxs)
        self.freq = random.uniform(args.f_cpu_min, args.f_cpu_max)

    def sim_time(self, grad_size):
        rate = random.uniform(1e6, 5e6)
        t_up = (grad_size * 32) / rate
        t_cpu = (args.num_local_epochs * args.cycles_per_sample * self.n_samples) / self.freq
        return t_cpu, t_up

    def train(self, global_weights):
        self.model.load_state_dict(global_weights)
        self.model.train()
        opt = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)

        p0 = flatten_params(self.model).detach().clone()

        for _ in range(args.num_local_epochs):
            for imgs, labels in self.loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                opt.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(imgs), labels)
                loss.backward()
                opt.step()

        grad = p0 - flatten_params(self.model).detach().clone()
        if args.noise_scale > 0:
            grad = grad + torch.randn_like(grad) * args.noise_scale
        return grad
