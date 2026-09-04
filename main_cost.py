import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from config import args
from models.network import SimpleCNN, VGG11CIFAR


def set_modern_style():
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.facecolor'] = '#FFFFFF'  # 纯白绘图区背景
    plt.rcParams['figure.facecolor'] = '#FFFFFF'  # 纯白画布背景
    plt.rcParams['grid.color'] = '#DEE2E6'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.edgecolor'] = '#CED4DA'
    plt.rcParams['text.color'] = '#2B2D42'
    plt.rcParams['xtick.color'] = '#495057'
    plt.rcParams['ytick.color'] = '#495057'

    # 坐标轴文字大小配置
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 18


def draw_3d_bars(ax, x_pos, values, colors, width=0.5):
    """绘制具有3D厚度感和高光质感的柱状图"""
    for i in range(10, 0, -1):
        ax.bar(x_pos + i * 0.005, values, width,
               color='black', alpha=0.015, zorder=1, edgecolor='none')

    bars = ax.bar(x_pos, values, width,
                  color=colors, edgecolor='#343A40', linewidth=1.2, zorder=3)

    ax.bar(x_pos - width / 2 + 0.02, values, 0.03,
           color='white', alpha=0.35, zorder=4, edgecolor='none')

    return bars


def calc_costs():
    args.dataset_name = 'cifar10'
    args.num_local_epochs = 1

    if args.dataset_name == 'cifar10':
        model = VGG11CIFAR()
        total_samples = 50000
    else:
        model = SimpleCNN()
        total_samples = 60000

    sizes = [p.numel() for p in model.parameters()]
    p_dim = sum(sizes)
    model_mb = p_dim * 4 / (1024 * 1024)

    surv_rate = 1.0 - args.target_straggler_rate
    n_surv = int(args.num_users * surv_rate)
    n_drop = args.num_users - n_surv

    # [A] 上行通信开销 (MB/轮)
    comm_fedavg = n_surv * model_mb
    comm_dmfl = n_surv * model_mb

    avg_partial = sum(sum(sizes[-i:]) for i in range(1, len(sizes) + 1)) / len(sizes)
    partial_mb = avg_partial * 4 / (1024 * 1024)
    comm_salf = n_surv * model_mb + n_drop * partial_mb

    # [B] 客户端终端计算开销 (MFLOPs/轮)
    samples_per_client = total_samples / args.num_users
    comp_per_client_full = (args.num_local_epochs * samples_per_client * (3 * p_dim)) / 1e6
    comp_per_client_partial = (args.num_local_epochs * samples_per_client * (3 * avg_partial)) / 1e6

    client_comp_fedavg = args.num_users * comp_per_client_full
    client_comp_dmfl = client_comp_fedavg
    client_comp_salf = n_surv * comp_per_client_full + n_drop * comp_per_client_partial

    set_modern_style()
    labels = ['FedAvg', 'SALF', 'DMFL']
    colors = ['#FF4D6D', '#06D6A0', '#118AB2']
    x_pos = np.arange(len(labels))
    width = 0.55

    # ------------------ 图表 1 ------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    comm_vals = [comm_fedavg, comm_salf, comm_dmfl]
    b1 = draw_3d_bars(ax1, x_pos, comm_vals, colors, width)

    ax1.set_title(f'Upload Communication Cost ({args.dataset_name.upper()})', pad=15)
    ax1.set_ylabel('Data Transfer (MB)', fontsize=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=20)
    ax1.tick_params(axis='y', labelsize=16)

    # 仅将 'DMFL' 刻度标签加粗
    for tick_label in ax1.get_xticklabels():
        if tick_label.get_text() == 'DMFL':
            tick_label.set_fontweight('bold')
        else:
            tick_label.set_fontweight('normal')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(comm_vals) * 1.15)

    # 仅加粗 DMFL 的柱顶数值
    for label, bar in zip(labels, b1):
        weight = 'bold' if label == 'DMFL' else 'normal'
        txt = ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                       f'{bar.get_height():.2f}', ha='center', va='bottom',
                       fontsize=18, fontweight=weight, color='#2B2D42', zorder=5)
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

    fig1.tight_layout()
    fig1.savefig(f'Results_Cost1_Comm_{args.dataset_name}_VGG.png', dpi=400, bbox_inches='tight')

    # ------------------ 图表 2 ------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    client_comps = [client_comp_fedavg, client_comp_salf, client_comp_dmfl]
    b2 = draw_3d_bars(ax2, x_pos, client_comps, colors, width)

    ax2.set_title(f'Client Computation Cost ({args.dataset_name.upper()})', pad=15)
    ax2.set_ylabel('Operations (MFLOPs)', fontsize=20)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=20)
    ax2.tick_params(axis='y', labelsize=16)

    # 仅将 'DMFL' 刻度标签加粗
    for tick_label in ax2.get_xticklabels():
        if tick_label.get_text() == 'DMFL':
            tick_label.set_fontweight('bold')
        else:
            tick_label.set_fontweight('normal')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(client_comps) * 1.15)

    for label, bar in zip(labels, b2):
        weight = 'bold' if label == 'DMFL' else 'normal'
        txt = ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                       f'{bar.get_height():,.0f}', ha='center', va='bottom',
                       fontsize=18, fontweight=weight, color='#2B2D42', zorder=5)
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

    fig2.tight_layout()
    fig2.savefig(f'Results_Cost2_Comp_{args.dataset_name}_VGG.png', dpi=400, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    calc_costs()