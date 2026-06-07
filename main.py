import time
import matplotlib.pyplot as plt
from config import args
from utils import plot_shadow

# 导入这四种算法的运行入口
from main_fedavg import run_fedavg
from main_fedprox import run_fedprox
from main_salf import run_salf
from main_dmfl import run_dmfl

if __name__ == '__main__':
    print(f">>> 训练集：{args.dataset_name.upper()}")

    t_start = time.time()
    _, _, acc_fed, loss_fed = run_fedavg(skip_train_eval=True)
    time_fed = time.time() - t_start

    t_start = time.time()
    _, _, acc_prox, loss_prox = run_fedprox(skip_train_eval=True, mu=0.5)
    time_prox = time.time() - t_start

    t_start = time.time()
    _, _, acc_salf, loss_salf = run_salf(skip_train_eval=True)
    time_salf = time.time() - t_start

    t_start = time.time()
    _, _, acc_dmfl, loss_dmfl = run_dmfl(skip_train_eval=True)
    time_dmfl = time.time() - t_start

    print("\n" + "=" * 55)
    print(f"  [1] FedAvg   耗时: {time_fed:8.2f} s")
    print(f"  [2] FedProx  耗时: {time_prox:8.2f} s")
    print(f"  [3] SALF     耗时: {time_salf:8.2f} s")
    print(f"  [4] DMFL     耗时: {time_dmfl:8.2f} s")
    print("=" * 55 + "\n")

    rnds = range(1, args.num_global_rounds + 1)

    TITLE_SIZE = 20  # 标题字体大小
    LABEL_SIZE = 18  # 坐标轴标签字体大小
    TICK_SIZE = 16  # 坐标轴刻度数字大小
    LEGEND_SIZE = 16  # 图例字体大小

    # ==== 精度对比图 ====
    fig1 = plt.figure(1, figsize=(10, 6))
    plot_shadow(plt, rnds, acc_fed, 'red', 'FedAvg')
    plot_shadow(plt, rnds, acc_prox, 'brown', 'FedProx')
    plot_shadow(plt, rnds, acc_salf, 'green', 'SALF')
    plot_shadow(plt, rnds, acc_dmfl, 'blue', 'DMFL')

    if args.warmup_rounds > 0:
        plt.axvline(args.warmup_rounds, color='gray', linestyle=':', label='Warm-up')

    plt.xlabel('Rounds', fontsize=LABEL_SIZE)
    plt.ylabel('Accuracy (%)', fontsize=LABEL_SIZE)
    plt.title(f'{args.dataset_name.upper()} Accuracy Comparison', fontsize=TITLE_SIZE)

    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)

    plt.legend(fontsize=LEGEND_SIZE)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Results_Accuracy_{args.dataset_name}_Rounds{args.num_global_rounds}_Shadow.png', dpi=300)

    # ==== Loss 对比图 ====
    fig2 = plt.figure(2, figsize=(10, 6))
    plot_shadow(plt, rnds, loss_fed, 'red', 'FedAvg')
    plot_shadow(plt, rnds, loss_prox, 'brown', 'FedProx')
    plot_shadow(plt, rnds, loss_salf, 'green', 'SALF')
    plot_shadow(plt, rnds, loss_dmfl, 'blue', 'DMFL')

    if args.warmup_rounds > 0:
        plt.axvline(args.warmup_rounds, color='gray', linestyle=':', label='Warm-up')

    plt.xlabel('Rounds', fontsize=LABEL_SIZE)
    plt.ylabel('Val Loss', fontsize=LABEL_SIZE)
    plt.title(f'{args.dataset_name.upper()} Loss Comparison', fontsize=TITLE_SIZE, fontweight='bold')

    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Results_Loss_{args.dataset_name}_Rounds{args.num_global_rounds}_Shadow.png', dpi=300)

    plt.show()