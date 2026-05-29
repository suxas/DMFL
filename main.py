import matplotlib.pyplot as plt
from config import args
from utils import plot_shadow
from main_fedavg import run_fedavg
from main_salf import run_salf
from main_dmfl import run_dmfl


if __name__ == '__main__':
    # 依次运行三种算法
    _, _, acc_fed, loss_fed = run_fedavg(skip_train_eval=True)
    _, _, acc_salf, loss_salf = run_salf(skip_train_eval=True)
    _, _, acc_dmfl, loss_dmfl = run_dmfl(skip_train_eval=True)

    rnds = range(1, args.num_global_rounds + 1)

    # 精度对比
    fig1 = plt.figure(1, figsize=(10, 6))
    plot_shadow(plt, rnds, acc_fed, 'red', 'FedAvg')
    plot_shadow(plt, rnds, acc_salf, 'green', 'SALF')
    plot_shadow(plt, rnds, acc_dmfl, 'blue', 'DMFL')
    if args.warmup_rounds > 0:
        plt.axvline(args.warmup_rounds, color='gray', linestyle=':', label='Warm-up')
    plt.xlabel('Rounds')
    plt.ylabel('Val Accuracy (%)')
    plt.title(f'{args.dataset_name.upper()} Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Results_Accuracy_{args.dataset_name}_Rounds{args.num_global_rounds}_Shadow.png', dpi=300)

    # Loss 对比
    fig2 = plt.figure(2, figsize=(10, 6))
    plot_shadow(plt, rnds, loss_fed, 'red', 'FedAvg')
    plot_shadow(plt, rnds, loss_salf, 'green', 'SALF')
    plot_shadow(plt, rnds, loss_dmfl, 'blue', 'DMFL')
    if args.warmup_rounds > 0:
        plt.axvline(args.warmup_rounds, color='gray', linestyle=':', label='Warm-up')
    plt.xlabel('Rounds')
    plt.ylabel('Val Loss')
    plt.title(f'{args.dataset_name.upper()} Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Results_Loss_{args.dataset_name}_Rounds{args.num_global_rounds}_Shadow.png', dpi=300)

    plt.show()
