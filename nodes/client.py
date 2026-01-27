import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from config import args
from utils import flatten_params


class LocalClient:
    def __init__(self, dataset, idxs, model):
        self.ldr_train = DataLoader(Subset(dataset, list(idxs)), batch_size=args.batch_size, shuffle=True)
        self.model = copy.deepcopy(model)
        self.dataset_len = len(idxs)

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

        # 返回梯度向量 (W_old - W_new)
        grad_vec = initial_params - final_params
        return grad_vec