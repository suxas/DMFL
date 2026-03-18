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
        grad_size_bits = param_dim * 32

        t_up = grad_size_bits / rate
        t_train = (args.num_local_epochs * args.cycles_per_sample * self.dataset_len) / self.f_cpu

        e_comm = args.p_ue_max * t_up
        e_comp = args.kappa * (self.f_cpu ** 2) * (args.num_local_epochs * args.cycles_per_sample * self.dataset_len)

        return t_train, t_up, e_comp, e_comm

    def train(self, global_weights, add_noise=True):
        self.model.load_state_dict(global_weights)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
        initial_params = flatten_params(self.model).detach().clone()

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


        if add_noise:
            base_noise_scale = getattr(args, 'noise_scale', 0.02)
            noise_std = random.uniform(0.5, 1.5) * base_noise_scale
            channel_noise = torch.randn_like(grad_vec) * noise_std
            grad_vec = grad_vec + channel_noise

        return grad_vec