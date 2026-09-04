import os
import time
import matplotlib.pyplot as plt
from config import args
from utils import plot_shadow
from utils import set_plot_format
from logger import DataLogger

from main_fedavg import run_fedavg
from main_fedprox import run_fedprox
from main_salf import run_salf
from main_dmfl import run_dmfl


def resolve_algo(logger, run_fn, name, **kwargs):
    """检查历史数据，有则跳过训练，无则运行训练并保存"""
    has_history, data = logger.check_history()
    expected = args.num_global_rounds
    if has_history and len(data) == expected:
        print(f"[{name}] 使用历史数据，共 {len(data)} 轮")
        acc = [d['accuracy'] for d in data]
        loss = [d['loss'] for d in data]
        return acc, loss, 0.0

    if has_history:
        print(f"[{name}] 历史轮次({len(data)})≠当前({expected})，重新训练")

    t_start = time.time()
    _, _, acc, loss = run_fn(**kwargs)
    elapsed = time.time() - t_start

    for i, (a, l) in enumerate(zip(acc, loss)):
        logger.log_epoch(i + 1, a, l)

    return acc, loss, elapsed


if __name__ == '__main__':
    set_plot_format()

    print(f">>> 训练集：{args.dataset_name.upper()}")

    if args.inner_client_iid == 1.0:
        iid_type = "IID"
    elif args.inner_client_iid == 0.0:
        iid_type = "Non-IID"
    else:
        iid_type = f"Partial-IID-{args.inner_client_iid}"

    acc_fed, loss_fed, time_fed = resolve_algo(
        DataLogger(args, 'FedAvg', iid_type), run_fedavg, 'FedAvg', skip_train_eval=True)
    acc_prox, loss_prox, time_prox = resolve_algo(
        DataLogger(args, 'FedProx', iid_type), run_fedprox, 'FedProx', skip_train_eval=True, mu=0.5)
    acc_salf, loss_salf, time_salf = resolve_algo(
        DataLogger(args, 'SALF', iid_type), run_salf, 'SALF', skip_train_eval=True)
    acc_dmfl, loss_dmfl, time_dmfl = resolve_algo(
        DataLogger(args, 'DMFL', iid_type), run_dmfl, 'DMFL', skip_train_eval=True)

    print("\n" + "=" * 55)
    print(f"  [1] FedAvg   耗时: {time_fed:8.2f} s")
    print(f"  [2] FedProx  耗时: {time_prox:8.2f} s")
    print(f"  [3] SALF     耗时: {time_salf:8.2f} s")
    print(f"  [4] DMFL     耗时: {time_dmfl:8.2f} s")
    print("=" * 55 + "\n")

    # 创建保存目录
    png_dir = os.path.join('result', 'png')
    eps_dir = os.path.join('result', 'eps')
    pdf_dir = os.path.join('result', 'pdf')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(eps_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    rnds = range(1, args.num_global_rounds + 1)

    TITLE_SIZE = 20
    LABEL_SIZE = 18
    TICK_SIZE = 16
    LEGEND_SIZE = 16

    # ==== Accuracy ====
    fig1 = plt.figure(1, figsize=(10, 6))
    plot_shadow(plt, rnds, acc_fed, 'red', 'FedAvg', marker='o', linestyle='-')
    plot_shadow(plt, rnds, acc_prox, 'brown', 'FedProx', marker='s', linestyle='--')
    plot_shadow(plt, rnds, acc_salf, 'green', 'SALF', marker='^', linestyle='-.')
    plot_shadow(plt, rnds, acc_dmfl, 'blue', 'DMFL', marker='D', linestyle=':')

    if args.warmup_rounds > 0:
        plt.axvline(args.warmup_rounds, color='gray', linestyle=':', label='Warm-up')

    plt.xlabel('Global Rounds', fontsize=LABEL_SIZE)
    plt.ylabel('Testing Accuracy (%)', fontsize=LABEL_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.grid(True)
    plt.tight_layout()

    # 保存各格式图片
    acc_base_name = f'Acc_{args.dataset_name}_Rounds{args.num_global_rounds}_Strag{args.target_straggler_rate}_IID{args.inner_client_iid}_Shadow'
    plt.savefig(os.path.join(png_dir, f'{acc_base_name}.png'), format='png', dpi=300)
    plt.savefig(os.path.join(eps_dir, f'{acc_base_name}.eps'), format='eps', dpi=300)
    plt.savefig(os.path.join(pdf_dir, f'{acc_base_name}.pdf'), format='pdf', dpi=300)

    # ==== Loss ====
    fig2 = plt.figure(2, figsize=(10, 6))
    plot_shadow(plt, rnds, loss_fed, 'red', 'FedAvg', marker='o', linestyle='-')
    plot_shadow(plt, rnds, loss_prox, 'brown', 'FedProx', marker='s', linestyle='--')
    plot_shadow(plt, rnds, loss_salf, 'green', 'SALF', marker='^', linestyle='-.')
    plot_shadow(plt, rnds, loss_dmfl, 'blue', 'DMFL', marker='D', linestyle=':')

    if args.warmup_rounds > 0:
        plt.axvline(args.warmup_rounds, color='gray', linestyle=':', label='Warm-up')

    plt.xlabel('Global Rounds', fontsize=LABEL_SIZE)
    plt.ylabel('Testing Loss', fontsize=LABEL_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.grid(True)
    plt.tight_layout()

    # 保存各格式图片
    loss_base_name = f'Loss_{args.dataset_name}_Rounds{args.num_global_rounds}_Strag{args.target_straggler_rate}_IID{args.inner_client_iid}_Shadow'
    plt.savefig(os.path.join(png_dir, f'{loss_base_name}.png'), format='png', dpi=300)
    plt.savefig(os.path.join(eps_dir, f'{loss_base_name}.eps'), format='eps', dpi=300)
    plt.savefig(os.path.join(pdf_dir, f'{loss_base_name}.pdf'), format='pdf', dpi=300)

    plt.show()