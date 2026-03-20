import torch
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import args
from dataset import get_mnist_data, split_data
from models.network import SimpleCNN
from models.diffusion import GradientDiffusion
from utils import flatten_params, unflatten_params
from nodes.client import LocalClient
from nodes.edge import EdgeServer

class DMFLEdgeState:
    """管理属于每个边缘基站的 DMFL 扩散模型相关状态"""
    def __init__(self, num_clients, input_dim):
        self.diffusion_dim = 510
        self.diffusion = GradientDiffusion(
            param_dim=self.diffusion_dim,
            hidden_dim=args.diff_hidden_dim,
            timesteps=args.diff_timesteps
        ).to(args.device)

        self.diff_optimizer = optim.SGD(
            self.diffusion.parameters(),
            lr=args.diff_lr,
            momentum=args.momentum,
            weight_decay=5e-4
        )
        self.historical_grads = {i: torch.zeros(input_dim).to(args.device) for i in range(num_clients)}
        self.replay_buffer_history = []
        self.replay_buffer_target = []
        self.w_hist = 5
        self.max_buffer_size = num_clients * self.w_hist

def evaluate_model(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size)
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(dataset), total_loss / len(dataset)

def run_dmfl():
    print("\n>>> 正在进行仿真: Method = dmfl")
    train_data, test_data = get_mnist_data()
    user_groups = split_data(train_data, args.num_users)

    global_model = SimpleCNN().to(args.device)
    param_dim = flatten_params(global_model).numel()

    clients = [LocalClient(train_data, user_groups[i], global_model) for i in range(args.num_users)]
    edge_servers = []
    users_per_edge = args.num_users // args.num_edge_servers
    for i in range(args.num_edge_servers):
        edge_servers.append(EdgeServer(i, clients[i * users_per_edge: (i + 1) * users_per_edge], param_dim))

    # 初始化每个基站的 DMFL 状态 (脱离了 edge.py)
    dmfl_states = {server.id: DMFLEdgeState(len(server.clients), param_dim) for server in edge_servers}

    t_acc_hist, t_loss_hist, v_acc_hist, v_loss_hist = [], [], [], []
    pbar = tqdm(range(args.num_global_rounds), desc="Training [dmfl]", ncols=100, file=sys.stdout)

    for epoch in pbar:
        current_round = epoch + 1
        is_warmup = current_round <= args.warmup_rounds

        global_model.train()
        global_weights = global_model.state_dict()
        edge_grads = []

        for edge_server in edge_servers:
            state = dmfl_states[edge_server.id]
            valid_grads = []

            # --- DMFL 聚合与训练扩散模型 ---
            client_conditions = [client.simulate_physical_conditions(param_dim) for client in edge_server.clients]
            total_times = [c[0] + c[1] for c in client_conditions]

            K_min = max(2, int(len(edge_server.clients) * 0.5))
            sorted_times = sorted(total_times)
            dynamic_t_win = min(args.t_deadline, sorted_times[K_min - 1])

            for local_idx, client in enumerate(edge_server.clients):
                t_train, t_up, e_comp, e_comm = client_conditions[local_idx]
                t_total = t_train + t_up
                is_straggler = (t_total > dynamic_t_win)

                if not is_straggler:
                    grad = client.train(global_weights)
                    valid_grads.append(grad)

                    # 提取特征存入池中
                    if torch.norm(state.historical_grads[local_idx]) > 0:
                        current_head = grad[-state.diffusion_dim:]
                        history_head = state.historical_grads[local_idx][-state.diffusion_dim:]
                        target_delta = current_head - history_head

                        state.replay_buffer_history.append(history_head.detach().clone())
                        state.replay_buffer_target.append(target_delta.detach().clone())

                    state.historical_grads[local_idx] = grad.detach().clone()

                else:
                    if not is_warmup:
                        # 扩散模型预测掉队梯度
                        base_grad = state.historical_grads[local_idx].clone()
                        if torch.norm(base_grad) > 0:
                            history_head = base_grad[-state.diffusion_dim:].unsqueeze(0)
                            predicted_delta = state.diffusion.generate(history_head).squeeze(0)

                            delta_norm = torch.norm(predicted_delta)
                            base_norm = torch.norm(history_head)
                            if delta_norm > base_norm:
                                predicted_delta = predicted_delta * (base_norm / (delta_norm + 1e-6)) * 0.5

                            base_grad[-state.diffusion_dim:] += predicted_delta
                            valid_grads.append(base_grad)

            # 更新当前基站的扩散模型
            if len(state.replay_buffer_history) > 0:
                if len(state.replay_buffer_history) > state.max_buffer_size:
                    state.replay_buffer_history = state.replay_buffer_history[-state.max_buffer_size:]
                    state.replay_buffer_target = state.replay_buffer_target[-state.max_buffer_size:]

                hist_tensor = torch.stack(state.replay_buffer_history)
                targ_tensor = torch.stack(state.replay_buffer_target)
                dataset = TensorDataset(targ_tensor, hist_tensor)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                state.diffusion.train()
                for _ in range(5):  # e_diff = 5
                    for batch_targ, batch_hist in dataloader:
                        loss = state.diffusion.train_step(batch_targ, batch_hist)
                        state.diff_optimizer.zero_grad()
                        loss.backward()
                        state.diff_optimizer.step()

            agg_grad = torch.stack(valid_grads).mean(dim=0) if len(valid_grads) > 0 else None

            if agg_grad is not None:
                edge_grads.append(agg_grad)

        if edge_grads:
            global_grad = torch.stack(edge_grads).mean(dim=0)
            new_params = flatten_params(global_model) - global_grad
            unflatten_params(global_model, new_params)

        train_acc, train_loss = evaluate_model(global_model, train_data)
        val_acc, val_loss = evaluate_model(global_model, test_data)

        t_acc_hist.append(train_acc)
        t_loss_hist.append(train_loss)
        v_acc_hist.append(val_acc)
        v_loss_hist.append(val_loss)
        pbar.set_postfix({'Val Acc': f"{val_acc:.2f}%", 'Val Loss': f"{val_loss:.4f}"})

    return t_acc_hist, t_loss_hist, v_acc_hist, v_loss_hist


if __name__ == '__main__':
    t_acc, t_loss, v_acc, v_loss = run_dmfl()
    epochs = range(1, args.num_global_rounds + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, t_acc, 'b--o', label='Train Accuracy')
    axes[0].plot(epochs, v_acc, 'r-^', label='Validation Accuracy')
    if args.warmup_rounds > 0:
        axes[0].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[0].set_title('DMFL: Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, t_loss, 'b--o', label='Train Loss')
    axes[1].plot(epochs, v_loss, 'r-^', label='Validation Loss')
    if args.warmup_rounds > 0:
        axes[1].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[1].set_title('DMFL: Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()