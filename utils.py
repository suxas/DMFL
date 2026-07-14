import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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


def plot_shadow(ax, x, y, color, label, window=20, marker=None, linestyle='-'):
    y = np.array(y)
    mean = np.zeros_like(y)
    std = np.zeros_like(y)
    for i in range(len(y)):
        lo = max(0, i - window // 2)
        hi = min(len(y), i + window // 2 + 1)
        mean[i] = np.mean(y[lo:hi])
        std[i] = np.std(y[lo:hi])
    ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=2.0, marker=marker,
            markevery=max(1, len(x) // 10), markersize=8, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def set_plot_format():
    """
    配置全局 matplotlib 画图格式：英文，Times New Roman 字体，字号稍大。
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.rcParams['font.size'] = 14  # 基础字号放大
    plt.rcParams['axes.titlesize'] = 16  # 标题字号
    plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签
    plt.rcParams['xtick.labelsize'] = 12  # X轴刻度
    plt.rcParams['ytick.labelsize'] = 12  # Y轴刻度
    plt.rcParams['legend.fontsize'] = 12  # 图例

    # 防止因字体放大导致标签被截断
    plt.rcParams['figure.autolayout'] = True