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
        raise ValueError("不支持的数据集。")
    return train_dataset, test_dataset


def split_data(dataset, num_users):
    """
    层级混合数据划分 (Hierarchical Hybrid Distribution):
    - 全局 Non-IID：基站间标签高度异构。
    - 簇内可调 IID：通过 inner_iid_degree 控制簇内客户端的异构程度。
    """
    num_edges = args.num_edge_servers
    users_per_edge = num_users // num_edges
    iid_deg = args.inner_client_iid

    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    # 1. 按照标签排序 (全局 Non-IID 基础)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    sorted_idxs = idxs_labels[0, :]

    # 2. 将全局数据切分为 3 个宏分片给 3 个基站
    macro_shards = np.array_split(sorted_idxs, num_edges)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    np.random.seed(1234)

    for edge_idx in range(num_edges):
        edge_data = macro_shards[edge_idx]
        total_edge_samples = len(edge_data)

        # 计算该基站下 IID 和 Non-IID 部分的数据量
        num_iid = int(total_edge_samples * iid_deg)

        # 3. 提取 IID 部分并打乱
        iid_pool = edge_data[:num_iid]
        np.random.shuffle(iid_pool)
        iid_client_shards = np.array_split(iid_pool, users_per_edge)

        # 4. 提取 Non-IID 部分并保持排序 (以保证内部客户端标签不平衡)
        non_iid_pool = edge_data[num_iid:]
        # 为了增加难度，Non-IID 部分不打乱，直接按顺序切分
        niid_client_shards = np.array_split(non_iid_pool, users_per_edge)

        for local_i in range(users_per_edge):
            global_idx = edge_idx * users_per_edge + local_i

            # 合并该客户端的两部分数据
            combined_idxs = np.concatenate((iid_client_shards[local_i], niid_client_shards[local_i]))
            # 最终打乱该客户端的本地数据，模拟本地训练的随机性
            np.random.shuffle(combined_idxs)

            dict_users[global_idx] = combined_idxs

    return dict_users