import torch
import torch.nn.functional as F


def flatten_params(model):
    """将模型所有参数展平为一维张量"""
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def unflatten_params(model, flat_params):
    """将一维张量还原并覆盖回模型参数"""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[offset:offset + numel].view_as(param))
        offset += numel


def evaluate_model(model, dataloader, device):
    """评估模型，返回准确率(%)和平均Loss"""
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(dataloader.dataset)
    acc = 100. * correct / len(dataloader.dataset)
    return acc, loss