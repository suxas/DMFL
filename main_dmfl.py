import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from config import args
from dataset import get_mnist_data, split_data
from models.network import SimpleCNN
from models.diffusion import GradientDiffusion
from utils import flatten_params, unflatten_params, evaluate_model
from nodes.client import LocalClient


class DMFLEdgeServer:
    def __init__(self, clients, param_dim):
        self.clients = clients
        self.param_dim = param_dim
        self.diffusion_dim = 510  # CNN最后一层的参数维度

        self.diffusion = GradientDiffusion(
            param_dim=self.diffusion_dim, hidden_dim=args.diff_hidden_dim, timesteps=args.diff_timesteps
        ).to(args.device)
        self.diff_optimizer = optim.SGD(self.diffusion.parameters(), lr=args.diff_lr, momentum=args.momentum,
                                        weight_decay=5e-4)

        # 经验回放设置
        self.historical_grads = {i: torch.zeros(param_dim).to(args.device) for i in range(len(clients))}
        self.replay_buffer_history = []
        self.replay_buffer_target = []
        self.max_buffer_size = len(clients) * 5

    def aggregate(self, global_weights, current_round):
        is_warmup = current_round <= args.warmup_rounds
        client_times = [c.simulate_physical_time(self.param_dim) for c in self.clients]
        total_times = [c[0] + c[1] for c in client_times]

        # 计算需要成功接收的保底设备数
        success_rate = 1.0 - args.target_straggler_rate
        K_min = max(1, int(len(assigned_clients) * success_rate))
        # 动态时间窗口受限于排序后的时间和绝对死线
        dynamic_t_win = min(args.t_deadline, sorted(total_times)[K_min - 1])

        # 收集或预测设备梯度
        for idx, client in enumerate(self.clients):
            is_straggler = (total_times[idx] > dynamic_t_win)

            if not is_straggler:
                grad = client.train(global_weights, add_noise=True)
                valid_grads.append(grad)

                if torch.norm(self.historical_grads[idx]) > 0:
                    current_head = grad[-self.diffusion_dim:]
                    history_head = self.historical_grads[idx][-self.diffusion_dim:]
                    self.replay_buffer_history.append(history_head.detach().clone())
                    self.replay_buffer_target.append((current_head - history_head).detach().clone())

                self.historical_grads[idx] = grad.detach().clone()
            else:
                if not is_warmup:
                    base_grad = self.historical_grads[idx].clone()
                    if torch.norm(base_grad) > 0:
                        history_head = base_grad[-self.diffusion_dim:].unsqueeze(0)
                        predicted_delta = self.diffusion.generate(history_head).squeeze(0)
                        base_grad[-self.diffusion_dim:] += predicted_delta
                        valid_grads.append(base_grad)

        # 后台经验池训练
        if len(self.replay_buffer_history) > 0:
            if len(self.replay_buffer_history) > self.max_buffer_size:
                self.replay_buffer_history = self.replay_buffer_history[-self.max_buffer_size:]
                self.replay_buffer_target = self.replay_buffer_target[-self.max_buffer_size:]

            if is_warmup or current_round % 3 == 0:
                hist_tensor = torch.stack(self.replay_buffer_history)
                targ_tensor = torch.stack(self.replay_buffer_target)
                dataset = TensorDataset(targ_tensor, hist_tensor)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                self.diffusion.train()
                e_diff = 5 if is_warmup else 2
                for _ in range(e_diff):
                    for batch_targ, batch_hist in dataloader:
                        loss = self.diffusion.train_step(batch_targ, batch_hist)
                        self.diff_optimizer.zero_grad()
                        loss.backward()
                        self.diff_optimizer.step()

        return torch.stack(valid_grads).mean(dim=0) if valid_grads else None


def run_dmfl():
    print("\n>>> [3/3] 正在运行 DMFL ")
    train_data, test_data = get_mnist_data()
    user_groups = split_data(train_data, args.num_users)
    global_model = SimpleCNN().to(args.device)
    param_dim = flatten_params(global_model).numel()

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]

    edge_servers = []
    users_per_edge = args.num_users // args.num_edge_servers
    for i in range(args.num_edge_servers):
        edge_servers.append(DMFLEdgeServer(clients[i * users_per_edge: (i + 1) * users_per_edge], param_dim))

    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    pbar = tqdm(range(args.num_global_rounds), desc="Training [DMFL]", file=sys.stdout, colour='green')

    for epoch in pbar:
        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []

        for edge in edge_servers:
            agg_grad = edge.aggregate(global_weights, epoch + 1)
            if agg_grad is not None: edge_grads.append(agg_grad)

        if edge_grads:
            global_grad = torch.stack(edge_grads).mean(dim=0)
            curr_params = flatten_params(global_model)
            unflatten_params(global_model, curr_params - global_grad)

        train_acc, train_loss = evaluate_model(global_model, train_loader, args.device)
        val_acc, val_loss = evaluate_model(global_model, test_loader, args.device)
        history['train_acc'].append(train_acc);
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc);
        history['val_loss'].append(val_loss)
        pbar.set_postfix({'Val Acc': f"{val_acc:.2f}%"})

    return history


if __name__ == '__main__':
    history = run_dmfl()
    epochs = range(1, args.num_global_rounds + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0].plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    axes[0].set_title('DMFL Accuracy (Train vs Val)')
    axes[0].legend();
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[1].plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    axes[1].set_title('DMFL Loss (Train vs Val)')
    axes[1].legend();
    axes[1].grid(True)
    plt.show()