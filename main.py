import matplotlib.pyplot as plt
import torch

from config import args
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"当前分配的设备: {args.device}")

# 从各个独立脚本导入仿真运行函数
from main_fedavg import run_fedavg
from main_salf import run_salf
from main_dmfl import run_dmfl

if __name__ == '__main__':
    _, _, v_acc_base, v_loss_base = run_fedavg(skip_train_eval=True)
    _, _, v_acc_salf, v_loss_salf = run_salf(skip_train_eval=True)
    _, _, v_acc_dmfl, v_loss_dmfl = run_dmfl(skip_train_eval=True)

    epochs = range(1, args.num_global_rounds + 1)
    ms = 2

    # ================= 图1: 验证集精度 =================
    plt.figure(1, figsize=(8, 6))
    plt.plot(epochs, v_acc_base, 'r--o', markersize=ms, label='FedAvg')
    plt.plot(epochs, v_acc_salf, 'g-^', markersize=ms, label='SALF')
    plt.plot(epochs, v_acc_dmfl, 'b-s', markersize=ms, label='DMFL')
    if args.warmup_rounds > 0:
        plt.axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    plt.xlabel('Global Communication Rounds')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    acc_filename = f"Results_Accuracy_{args.dataset_name}_Rounds{args.num_global_rounds}.png"
    plt.savefig(acc_filename, dpi=300, bbox_inches='tight')

    # ================= 图2: 验证集 Loss =================
    plt.figure(2, figsize=(8, 6))
    plt.plot(epochs, v_loss_base, 'r--o', markersize=ms, label='FedAvg')
    plt.plot(epochs, v_loss_salf, 'g-^', markersize=ms, label='SALF')
    plt.plot(epochs, v_loss_dmfl, 'b-s', markersize=ms, label='DMFL')
    if args.warmup_rounds > 0:
        plt.axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    plt.xlabel('Global Communication Rounds')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_filename = f"Results_Loss_{args.dataset_name}_Rounds{args.num_global_rounds}.png"
    plt.savefig(loss_filename, dpi=300, bbox_inches='tight')

    plt.show()