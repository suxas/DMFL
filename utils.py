import torch

def flatten_params(model):
    """将模型参数展平为一维向量"""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def unflatten_params(model, params_vec):
    """将一维向量恢复为模型参数"""
    idx = 0
    for p in model.parameters():
        numel = p.data.numel()
        p.data.copy_(params_vec[idx:idx+numel].view_as(p.data))
        idx += numel