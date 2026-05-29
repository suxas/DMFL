import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import args
from dataset import get_dataset, split_data
from models.network import SimpleCNN, VGG11CIFAR
from models.diffusion import GradientDiffusion
from utils import set_seed, flatten_params, unflatten_params, evaluate
from nodes.client import LocalClient
from nodes.edge import EdgeServer


class EdgeState:
    def __init__(self, n_clients, p_dim, d_dim):
        self.diffusion = GradientDiffusion(d_dim, args.diff_hidden_dim, args.diff_timesteps).to(args.device)
        self.opt = optim.SGD(self.diffusion.parameters(), lr=args.diff_lr,
                             momentum=args.momentum, weight_decay=5e-4)
        self.hist = {i: torch.zeros(p_dim, device=args.device) for i in range(n_clients)}
        self.buf_x, self.buf_y = [], []   # x: 历史分类头, y: 增量
        self.max_buf = n_clients * 5


def run_dmfl(skip_train_eval=False):
    set_seed(42)
    print(f"\n>>> DMFL 仿真 ({args.dataset_name.upper()})")
    train_ds, test_ds = get_dataset()
    groups = split_data(train_ds, args.num_users)

    if args.dataset_name == 'cifar10':
        model = VGG11CIFAR().to(args.device)
        diff_dim = args.diff_dim_cifar
    else:
        model = SimpleCNN().to(args.device)
        diff_dim = args.diff_dim_mnist

    p_dim = flatten_params(model).numel()
    feat_dim = p_dim - diff_dim   # 特征提取器维度

    clients = [LocalClient(train_ds, groups[i], model) for i in range(args.num_users)]
    u_per = args.num_users // args.num_edge_servers
    servers = [EdgeServer(i, clients[i * u_per:(i + 1) * u_per], p_dim)
               for i in range(args.num_edge_servers)]
    states = {s.id: EdgeState(len(s.clients), p_dim, diff_dim) for s in servers}

    t_acc, t_loss, v_acc, v_loss = [], [], [], []
    pbar = tqdm(range(1, args.num_global_rounds + 1), desc="DMFL", ncols=100, file=sys.stdout)

    for rnd in pbar:
        model.train()
        w_global = model.state_dict()
        edge_grads = []

        for srv in servers:
            st = states[srv.id]

            times = [sum(c.sim_time(p_dim)) for c in srv.clients]
            k = max(1, int(len(srv.clients) * (1.0 - args.target_straggler_rate))) - 1
            t_win = min(args.t_deadline, sorted(times)[k])

            active = []        # 成功上传的真实梯度
            dropped = []       # 掉队者索引

            for i, c in enumerate(srv.clients):
                if times[i] <= t_win:
                    g = c.train(w_global)
                    active.append(g)
                    # 更新扩散模型经验池
                    if torch.norm(st.hist[i]) > 0:
                        st.buf_x.append(st.hist[i][-diff_dim:].detach().clone())
                        st.buf_y.append((g[-diff_dim:] - st.hist[i][-diff_dim:]).detach().clone())
                    st.hist[i] = g.detach().clone()
                else:
                    dropped.append(i)

            # 存活者特征提取器平均梯度
            avg_feat = torch.stack([g[:feat_dim] for g in active]).mean(dim=0) if active else \
                torch.zeros(feat_dim, device=args.device)

            all_grads = list(active)

            for i in dropped:
                base = st.hist[i].clone()
                if rnd > args.warmup_rounds and torch.norm(base) > 0:
                    # 扩散模型预测分类头增量
                    head_hist = base[-diff_dim:].unsqueeze(0)
                    delta = st.diffusion.generate(head_hist).squeeze(0)
                    if torch.norm(delta) > torch.norm(head_hist):
                        delta = delta * (torch.norm(head_hist) / (torch.norm(delta) + 1e-6)) * 0.8
                    fake_head = base[-diff_dim:] + delta

                    # 特征提取器按 IID 比例插值
                    hist_feat = base[:feat_dim]
                    fake_feat = (args.inner_client_iid * avg_feat +
                                 (1.0 - args.inner_client_iid) * hist_feat) if active else hist_feat

                    all_grads.append(torch.cat([fake_feat, fake_head]))
                else:
                    all_grads.append(base)

            # 扩散模型训练
            if st.buf_x:
                st.buf_x = st.buf_x[-st.max_buf:]
                st.buf_y = st.buf_y[-st.max_buf:]
                ds = TensorDataset(torch.stack(st.buf_y), torch.stack(st.buf_x))
                loader = DataLoader(ds, batch_size=32, shuffle=True)
                st.diffusion.train()
                for _ in range(10):
                    for targ, hist in loader:
                        loss = st.diffusion.train_step(targ, hist)
                        st.opt.zero_grad()
                        loss.backward()
                        st.opt.step()

            if all_grads:
                edge_grads.append(torch.stack(all_grads).sum(dim=0))

        if edge_grads:
            g_global = sum(edge_grads) / args.num_users
            unflatten_params(model, flatten_params(model) - g_global)

        acc_v, loss_v = evaluate(model, test_ds, args.batch_size)
        v_acc.append(acc_v)
        v_loss.append(loss_v)

        if not skip_train_eval:
            acc_t, loss_t = evaluate(model, train_ds, args.batch_size)
            t_acc.append(acc_t)
            t_loss.append(loss_t)
        else:
            t_acc.append(0.0)
            t_loss.append(0.0)

        pbar.set_postfix({'Val Acc': f'{acc_v:.2f}%', 'Val Loss': f'{loss_v:.4f}'})

    return t_acc, t_loss, v_acc, v_loss
