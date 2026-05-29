import numpy as np
from torchvision import datasets, transforms
from config import args


def get_dataset():
    if args.dataset_name == 'cifar10':
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = datasets.CIFAR10(args.data_path, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(args.data_path, train=False, download=True, transform=test_tf)
    elif args.dataset_name == 'mnist':
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_ds = datasets.MNIST(args.data_path, train=True, download=True, transform=tf)
        test_ds = datasets.MNIST(args.data_path, train=False, download=True, transform=tf)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset_name}")
    return train_ds, test_ds


def split_data(dataset, num_users):
    """层级混合划分：edge 间 Non-IID，edge 内按 inner_client_iid 混合。"""
    n_edge = args.num_edge_servers
    u_per = num_users // n_edge
    iid_deg = args.inner_client_iid

    labels = np.array(dataset.targets)
    idx = np.arange(len(labels))

    # 按标签排序后均匀切分给各 edge（保证 edge 间 Non-IID）
    sorted_idx = idx[labels.argsort()]
    shards = np.array_split(sorted_idx, n_edge)

    user_map = {i: np.array([], dtype='int64') for i in range(num_users)}

    for eid in range(n_edge):
        shard = shards[eid]
        total = len(shard)
        n_iid = int(total * iid_deg)

        iid_pool = shard[:n_iid].copy()
        np.random.shuffle(iid_pool)
        iid_splits = np.array_split(iid_pool, u_per)

        niid_pool = shard[n_iid:]
        niid_splits = np.array_split(niid_pool, u_per)

        for j in range(u_per):
            uid = eid * u_per + j
            combined = np.concatenate([iid_splits[j], niid_splits[j]])
            np.random.shuffle(combined)
            user_map[uid] = combined

    return user_map
