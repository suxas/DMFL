import numpy as np
from torchvision import datasets, transforms
from config import args


def get_dataset():
    """根据 config 自动下载并返回对应数据集"""
    if args.dataset_name == 'cifar10':
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
        train_dataset = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)

    elif args.dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(args.data_path, train=False, download=True, transform=transform)

    else:
        raise ValueError("不支持的数据集。请在 config 中设置 'mnist' 或 'cifar10'")

    return train_dataset, test_dataset


def split_data(dataset, num_users):
    """Pathological Non-IID 数据划分：每个用户严格分配 2 种标签的数据块"""
    num_shards = 2 * num_users
    num_imgs_per_shard = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(num_shards * num_imgs_per_shard)
    # 兼容 MNIST 和 CIFAR-10 的 targets 格式差异
    labels = np.array(dataset.targets)[:num_shards * num_imgs_per_shard]

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            start_idx = rand * num_imgs_per_shard
            end_idx = (rand + 1) * num_imgs_per_shard
            dict_users[i] = np.concatenate((dict_users[i], idxs[start_idx:end_idx]), axis=0)

    return dict_users