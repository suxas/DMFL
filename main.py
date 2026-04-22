import matplotlib.pyplot as plt
import torch
import numpy as np

from config import args

print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"当前分配的设备: {args.device}")

from main_fedavg import run_fedavg
from main_salf import run_salf
from main_dmfl import run_dmfl


# 🌟 绘制阴影折线图的辅助函数
def plot_with_shadow(ax, x, y, color, label, window=20):
    """
    使用滑动窗口计算均值（主线）和标准差（阴影），将震荡转化为置信区间
    """
    y_arr = np.array(y)
    y_mean = np.zeros_like(y_arr)
    y_std = np.zeros_like(y_arr)

    for i in range(len(y_arr)):
        start = max(0, i - window // 2)
        end = min(len(y_arr), i + window // 2 + 1)
        y_mean[i] = np.mean(y_arr[start:end])
        y_std[i] = np.std(y_arr[start:end])

    # 绘制平滑主线
    ax.plot(x, y_mean, color=color, linestyle='-', label=label, linewidth=2.0)
    # 绘制波动阴影 (上下一个标准差)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)


if __name__ == '__main__':
    original_lr = args.lr

    # 1. 运行 FedAvg
    args.lr = original_lr
    _, _, v_acc_base, v_loss_base = run_fedavg(skip_train_eval=True)

    # 2. 运行 SALF
    args.lr = original_lr
    _, _, v_acc_salf, v_loss_salf = run_salf(skip_train_eval=True)

    # 3. 运行 DMFL
    args.lr = original_lr
    _, _, v_acc_dmfl, v_loss_dmfl = run_dmfl(skip_train_eval=True)

    epochs = range(1, args.num_global_rounds + 1)

    # ================= 图1: 验证集精度 (阴影图) =================
    plt.figure(1, figsize=(10, 6))
    plot_with_shadow(plt, epochs, v_acc_base, 'red', 'FedAvg', window=20)
    plot_with_shadow(plt, epochs, v_acc_salf, 'green', 'SALF', window=20)
    plot_with_shadow(plt, epochs, v_acc_dmfl, 'blue', 'DMFL', window=20)

    if args.warmup_rounds > 0:
        plt.axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    plt.xlabel('Global Communication Rounds')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Comparison (Smoothed w/ Variance Shadow)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Results_Accuracy_{args.dataset_name}_Rounds{args.num_global_rounds}_Shadow.png", dpi=300)

    # ================= 图2: 验证集 Loss (阴影图) =================
    plt.figure(2, figsize=(10, 6))
    plot_with_shadow(plt, epochs, v_loss_base, 'red', 'FedAvg', window=20)
    plot_with_shadow(plt, epochs, v_loss_salf, 'green', 'SALF', window=20)
    plot_with_shadow(plt, epochs, v_loss_dmfl, 'blue', 'DMFL', window=20)

    if args.warmup_rounds > 0:
        plt.axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    plt.xlabel('Global Communication Rounds')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison (Smoothed w/ Variance Shadow)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Results_Loss_{args.dataset_name}_Rounds{args.num_global_rounds}_Shadow.png", dpi=300)

    plt.show()