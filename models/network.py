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
        self.data_size = 32

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intemidiate_size_1, kernel_size=kernel_size)
        self.data_size = (self.data_size - kernel_size + 1) / 2

        self.conv2 = nn.Conv2d(in_channels=intemidiate_size_1, out_channels=intemidiate_size_1, kernel_size=kernel_size)
        self.data_size = int((self.data_size - kernel_size + 1) / 2)

        self.fc1 = nn.Linear(intemidiate_size_1 * int(self.data_size) * int(self.data_size), intemidiate_size_2)
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
        self.data_size = (self.data_size - kernel_size + 1) / 2

        self.conv2 = nn.Conv2d(in_channels=intemidiate_size_1, out_channels=intemidiate_size_1, kernel_size=kernel_size)
        self.data_size = int((self.data_size - kernel_size + 1) / 2)

        self.fc1 = nn.Linear(intemidiate_size_1 * int(self.data_size) * int(self.data_size), intemidiate_size_2)
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

# CIFAR-10 的 VGG 网络结构配置
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGGBase(nn.Module):
    """
    VGG 基础类，适配 CIFAR-10 数据集 (32x32 RGB)
    """
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        self.features = self._make_layers(cfg_vgg[vgg_name])
        # CIFAR-10 经过 5 次 MaxPool (2x2) 后，32x32 会变成 1x1
        # 所以进入全连接层的特征图大小为 512 * 1 * 1
        self.classifier = nn.Linear(512, 10)

    def forward(self, x, verbose=False):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3  # CIFAR 数据集是 3 通道 (RGB)
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 【核心修复】：使用 GroupNorm 完美平替 BatchNorm，彻底解决 2.3 死锁问题
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=32, num_channels=x),  # 加入组归一化
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG11CIFAR(VGGBase):
    def __init__(self):
        super(VGG11CIFAR, self).__init__('VGG11')


class VGG13CIFAR(VGGBase):
    def __init__(self):
        super(VGG13CIFAR, self).__init__('VGG13')


class VGG16CIFAR(VGGBase):
    def __init__(self):
        super(VGG16CIFAR, self).__init__('VGG16')


class VGG19CIFAR(VGGBase):
    def __init__(self):
        super(VGG19CIFAR, self).__init__('VGG19')