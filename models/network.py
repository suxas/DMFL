import torch.nn as nn
import torch.nn.functional as F

# 参考SALF论文中的任务模型
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # SALF 针对 MNIST 的参数设置:
        # kernel_size = 5, intemidiate_size_1 = 6, intemidiate_size_2 = 50

        self.intemidiate_size_1 = 6
        self.data_size = 28  # MNIST 原始图像大小为 28x28

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.intemidiate_size_1, kernel_size=5)
        self.data_size = int((self.data_size - 5 + 1) / 2)  # 变为 12
        self.conv2 = nn.Conv2d(in_channels=self.intemidiate_size_1, out_channels=self.intemidiate_size_1, kernel_size=5)
        self.data_size = int((self.data_size - 5 + 1) / 2)  # 变为 4

        # 全连接层
        flattened_size = self.intemidiate_size_1 * self.data_size * self.data_size
        self.fc1 = nn.Linear(flattened_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, verbose=False):
        # Conv -> ReLU -> MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # 展平
        x = x.view(-1, self.intemidiate_size_1 * self.data_size * self.data_size)

        # 全连接层 1
        x = self.fc1(x)
        x = F.relu(x)

        # 全连接层 2 (输出层)
        x = self.fc2(x)

        return x