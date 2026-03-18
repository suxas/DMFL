import matplotlib.pyplot as plt
from config import args

# 解耦导入各算法跑通模块
from main_fedavg import run_fedavg
from main_salf import run_salf
from main_dmfl import run_dmfl

if __name__ == '__main__':

    hist_fedavg = run_fedavg()
    hist_salf = run_salf()
    hist_dmfl = run_dmfl()

    epochs = range(1, args.num_global_rounds + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 准确度 ---
    axes[0].plot(epochs, hist_fedavg['val_acc'], 'r--o', label='FedAvg')
    axes[0].plot(epochs, hist_salf['val_acc'], 'g-^', label='SALF')
    axes[0].plot(epochs, hist_dmfl['val_acc'], 'b-s', label='DMFL (Ours)')
    if args.warmup_rounds > 0:
        axes[0].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')

    axes[0].set_xlabel('Global Communication Rounds')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Validation Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True)

    # --- Loss  ---
    axes[1].plot(epochs, hist_fedavg['val_loss'], 'r--o', label='FedAvg')
    axes[1].plot(epochs, hist_salf['val_loss'], 'g-^', label='SALF')
    axes[1].plot(epochs, hist_dmfl['val_loss'], 'b-s', label='DMFL (Ours)')
    if args.warmup_rounds > 0:
        axes[1].axvline(x=args.warmup_rounds, color='gray', linestyle=':', label='Warm-up End')

    axes[1].set_xlabel('Global Communication Rounds')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss Comparison')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('Algorithm Comparison: Accuracy and Loss', fontsize=16)
    plt.tight_layout()
    plt.show()