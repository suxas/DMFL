import numpy as np
from torchvision import datasets, transforms
from config import args


def get_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(args.data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_path, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def split_data(dataset, num_users):
    #Non-IID 划分
    num_shards = 2 * num_users  # 例如 10个用户 -> 20个分片
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()[:len(idxs)]  # 截断以匹配

    # 排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 分配
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users