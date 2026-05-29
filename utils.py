import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def unflatten_params(model, flat):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view_as(p))
        offset += n


def evaluate(model, dataset, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct, total = 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
            out = model(x)
            total += F.cross_entropy(out, y, reduction='sum').item()
            correct += out.argmax(dim=1).eq(y).sum().item()
    return 100. * correct / len(dataset), total / len(dataset)


def plot_shadow(ax, x, y, color, label, window=20):
    y = np.array(y)
    mean = np.zeros_like(y)
    std = np.zeros_like(y)
    for i in range(len(y)):
        lo = max(0, i - window // 2)
        hi = min(len(y), i + window // 2 + 1)
        mean[i] = np.mean(y[lo:hi])
        std[i] = np.std(y[lo:hi])
    ax.plot(x, mean, color=color, linestyle='-', linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
