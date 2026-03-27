import numpy as np
from torchvision import datasets, transforms


def get_cifar10_data():
    """获取并预处理 CIFAR-10 数据集"""
    # 训练集加入基础的数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset


def split_cifar10_data(dataset, num_users):
    """Pathological Non-IID 数据划分：每个用户随机分配 2 个 Shard (即两种主要特征/标签)"""
    num_shards = 2 * num_users
    num_imgs_per_shard = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(num_shards * num_imgs_per_shard)
    labels = np.array(dataset.targets)[:num_shards * num_imgs_per_shard]

    # 根据标签对数据索引进行排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 为每个用户随机分配 2 个 shard
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            start_idx = rand * num_imgs_per_shard
            end_idx = (rand + 1) * num_imgs_per_shard
            dict_users[i] = np.concatenate((dict_users[i], idxs[start_idx:end_idx]), axis=0)

    return dict_users