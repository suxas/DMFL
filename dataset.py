import numpy as np
from torchvision import datasets, transforms
from config import args


def get_dataset():
    """根据 config 自动下载并返回对应数据集"""
    if args.dataset_name == 'cifar10':
        # 对齐 SALF，使用 0.5 标准化，不使用复杂数据增强
        transform_train = transforms.Compose([
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
    """
    层级数据划分 (Hierarchical Data Distribution):
    - 全局 Non-IID：不同的 Edge Server (基站) 拥有完全不同的数据标签。
    - 簇内 IID：同一个 Edge Server (基站) 下属的客户端拥有均匀、同分布的数据。
    """
    num_edges = args.num_edge_servers
    users_per_edge = num_users // num_edges

    # 1. 提取所有图片的索引和真实标签
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    # 2. 按照标签大小进行排序 (将 0-9 的同类图片聚在一起)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    sorted_idxs = idxs_labels[0, :]

    # 3. 将排好序的数据切分成 num_edges 个"宏数据块"
    # 由于数据已按标签排序，每个基站只会分到一部分标签种类 (实现 全局 Non-IID)
    macro_shards = np.array_split(sorted_idxs, num_edges)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # 设定随机种子保证实验可复现
    np.random.seed(1234)

    # 4. 为每个基站内部的客户端分配数据
    for edge_idx in range(num_edges):
        edge_data_idxs = macro_shards[edge_idx]

        # 在该基站内部，将"宏数据块"彻底打乱 (实现 簇内 IID)
        np.random.shuffle(edge_data_idxs)

        # 将打乱后的数据均匀切分给该基站下的所有客户端
        edge_clients_idxs = np.array_split(edge_data_idxs, users_per_edge)

        for local_i, client_data in enumerate(edge_clients_idxs):
            # 计算全局客户端的真实 ID (0 ~ 29)
            global_client_idx = edge_idx * users_per_edge + local_i
            dict_users[global_client_idx] = client_data

    return dict_users