import matplotlib.pyplot as plt
from config import args

# 从各个独立脚本导入仿真运行函数
from main_fedavg import run_fedavg
from main_salf import run_salf
from main_dmfl import run_dmfl

if __name__ == '__main__':
    _, _, v_acc_base, v_loss_base = run_fedavg(skip_train_eval=True)
    _, _, v_acc_salf, v_loss_salf = run_salf(skip_train_eval=True)
    _, _, v_acc_dmfl, v_loss_dmfl = run_dmfl(skip_train_eval=True)

    # 绘制三大算法对照图 (对比验证集效果)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, args.num_global_rounds + 1)

    # 子图1: 验证集精度对比
    axes[0].plot(epochs, v_acc_base, 'r--o', label='FedAvg')
    axes[0].plot(epochs, v_acc_salf, 'g-^', label='SALF')
    axes[0].plot(epochs, v_acc_dmfl, 'b-s', label='DMFL')
    if args.warmup_rounds > 0:
        axes[0].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[0].set_xlabel('Global Communication Rounds')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Validation Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True)

    # 子图2: 验证集Loss对比
    axes[1].plot(epochs, v_loss_base, 'r--o', label='FedAvg')
    axes[1].plot(epochs, v_loss_salf, 'g-^', label='SALF')
    axes[1].plot(epochs, v_loss_dmfl, 'b-s', label='DMFL')
    if args.warmup_rounds > 0:
        axes[1].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')
    axes[1].set_xlabel('Global Communication Rounds')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss Comparison')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('Comparison of FL Methods: Validation Set Only')
    plt.tight_layout()
    plt.show()