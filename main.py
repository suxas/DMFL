import matplotlib.pyplot as plt
from config import args

# 从各个独立脚本导入仿真运行函数
from main_fedavg import run_fedavg
from main_salf import run_salf
from main_dmfl import run_dmfl

if __name__ == '__main__':
    # 分别运行并获取三个独立算法的结果
    _, _, v_acc_base, v_loss_base = run_fedavg()
    _, _, v_acc_salf, v_loss_salf = run_salf()
    _, _, v_acc_dmfl, v_loss_dmfl = run_dmfl()

    # 绘制三大算法对照图 (只对比验证集效果)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, args.num_global_rounds + 1)

    # 子图1: 验证集精度对比
    axes[0].plot(epochs, v_acc_base, 'r--o', label='fedavg')
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
    axes[1].plot(epochs, v_loss_base, 'r--o', label='fedavg')
    axes[1].plot(epochs, v_loss_salf, 'g-^', label='SALF')
    axes[1].plot(epochs, v_loss_dmfl, 'b-s', label='DMFL')
    axes[1].set_xlabel('Global Communication Rounds')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss Comparison')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('Comparison of FL Methods: Accuracy & Loss')
    plt.tight_layout()
    plt.show()