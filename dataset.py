import numpy as np
from torchvision import datasets, transforms
from config import args


def get_dataset():
    """根据 config 自动下载并返回对应数据集"""
    if args.dataset_name == 'cifar10':
        # 恢复随机裁剪和水平翻转
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)

    elif args.dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(args.data_path, train=False, download=True, transform=transform)

    else:
        raise ValueError("不支持的数据集。请在 config 中设置 'mnist' 或 'cifar10'")

    return train_dataset, test_dataset


def split_data(dataset, num_users):
    """层级数据划分 (Hierarchical Data Distribution)"""
    num_edges = args.num_edge_servers
    users_per_edge = num_users // num_edges

    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    sorted_idxs = idxs_labels[0, :]

    macro_shards = np.array_split(sorted_idxs, num_edges)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    np.random.seed(1234)

    for edge_idx in range(num_edges):
        edge_data_idxs = macro_shards[edge_idx]
        np.random.shuffle(edge_data_idxs)
        edge_clients_idxs = np.array_split(edge_data_idxs, users_per_edge)

        for local_i, client_data in enumerate(edge_clients_idxs):
            global_client_idx = edge_idx * users_per_edge + local_i
            dict_users[global_client_idx] = client_data

    return dict_users