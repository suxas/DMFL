import torch
import numpy as np


def get_salf_partial_grad(model, full_grad_vec):
    """
    参考 SALF 算法实现：
    当节点成为 straggler 时，只上传从输出层向后传播的部分层梯度。
    由于 DMFL 中使用的是展平的一维梯度向量 (full_grad_vec)，
    通过计算每层参数的数量，将前面未完成计算的层的梯度置为 0，从而模拟局部更新。
    """
    # 获取模型中每层参数的数量 (按照从输入到输出的顺序)
    param_sizes = [p.numel() for p in model.parameters()]
    num_of_layers = len(param_sizes)

    # 随机决定该 straggler 完成了多少层的反向传播 (1 到 num_of_layers)
    # 对应 SALF 源码中的 up_to_layer = np.random.randint(1, num_of_layers + 1)
    up_to_layer = np.random.randint(1, num_of_layers + 1)

    # 从后往前保留 up_to_layer 层，这意味着前面的 num_of_layers - up_to_layer 层梯度被置为0
    layers_to_zero = num_of_layers - up_to_layer

    # 计算需要置零的参数总数 (展平后的索引)
    num_zeros = sum(param_sizes[:layers_to_zero])

    salf_grad_vec = full_grad_vec.clone()
    if num_zeros > 0:
        # 将前段未计算完的层对应的梯度设为0
        salf_grad_vec[:num_zeros] = 0.0

    return salf_grad_vec