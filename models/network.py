import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    基于 SALF 论文中的 CNN2Layer 结构，用于 MNIST
    """

    def __init__(self, in_channels=1, output_size=10, data='mnist', kernel_size=5, intemidiate_size_1=6,
                 intemidiate_size_2=50):
        super(SimpleCNN, self).__init__()
        self.intemidiate_size_1 = intemidiate_size_1
        self.data = data
        self.data_size = 28 if data == 'mnist' else 32

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intemidiate_size_1, kernel_size=kernel_size)
        self.data_size = (self.data_size - kernel_size + 1) / 2

        self.conv2 = nn.Conv2d(in_channels=intemidiate_size_1, out_channels=intemidiate_size_1, kernel_size=kernel_size)
        self.data_size = int((self.data_size - kernel_size + 1) / 2)

        # 全连接层 1
        self.fc1 = nn.Linear(intemidiate_size_1 * int(self.data_size) * int(self.data_size), intemidiate_size_2)
        # 最后一层全连接分类头： 50 * 10 (weight) + 10 (bias) = 510
        self.fc2 = nn.Linear(intemidiate_size_2, output_size)

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(-1, self.intemidiate_size_1 * int(self.data_size) * int(self.data_size))
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    """
    基于 SALF 论文中的 CNN2Layer 结构，用于 CIFAR-10
    """

    def __init__(self, in_channels=3, output_size=10, kernel_size=5, intemidiate_size_1=6,
                 intemidiate_size_2=50):
        super(CNNCifar, self).__init__()
        self.intemidiate_size_1 = intemidiate_size_1
        self.data_size = 32

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intemidiate_size_1, kernel_size=kernel_size)
        self.data_size = (self.data_size - kernel_size + 1) / 2  # 变为 14

        self.conv2 = nn.Conv2d(in_channels=intemidiate_size_1, out_channels=intemidiate_size_1, kernel_size=kernel_size)
        self.data_size = int((self.data_size - kernel_size + 1) / 2)  # 变为 5

        # 第一层全连接：6 * 5 * 5 -> 50
        self.fc1 = nn.Linear(intemidiate_size_1 * int(self.data_size) * int(self.data_size), intemidiate_size_2)
        # 最后一层全连接维度: 50*10 + 10 = 510 (必须在主函数同步修改 diff_dim)
        self.fc2 = nn.Linear(intemidiate_size_2, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(-1, self.intemidiate_size_1 * int(self.data_size) * int(self.data_size))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x