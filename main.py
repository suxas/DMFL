import time
import matplotlib.pyplot as plt
from config import args
from utils import plot_shadow
from utils import set_plot_format
from logger import DataLogger

# 导入这四种算法的运行入口
from main_fedavg import run_fedavg
from main_fedprox import run_fedprox
from main_salf import run_salf
from main_dmfl import run_dmfl

if __name__ == '__main__':
    # 配置全局绘图格式
    set_plot_format()
    
    print(f">>> 训练集：{args.dataset_name.upper()}")

    # 确定IID类型
    if args.inner_client_iid == 1.0:
        iid_type = "IID"
    elif args.inner_client_iid == 0.0:
        iid_type = "Non-IID"
    else:
        iid_type = f"Partial-IID-{args.inner_client_iid}"

    # 为每个算法创建DataLogger实例
    logger_fedavg = DataLogger(args, 'FedAvg', iid_type)
    logger_fedprox = DataLogger(args, 'FedProx', iid_type)
    logger_salf = DataLogger(args, 'SALF', iid_type)
    logger_dmfl = DataLogger(args, 'DMFL', iid_type)

    # 检查历史数据并运行算法（轮次必须匹配才使用历史数据）
    expected_rounds = args.num_global_rounds

    has_fedavg, data_fedavg = logger_fedavg.check_history()
    if has_fedavg and len(data_fedavg) == expected_rounds:
        print(f"[FedAvg] 使用历史数据，共 {len(data_fedavg)} 轮")
        acc_fed = [d['accuracy'] for d in data_fedavg]
        loss_fed = [d['loss'] for d in data_fedavg]
        time_fed = 0.0
    else:
        if has_fedavg:
            print(f"[FedAvg] 历史数据轮次({len(data_fedavg)})与当前配置({expected_rounds})不匹配，重新训练")
        t_start = time.time()
        _, _, acc_fed, loss_fed = run_fedavg(skip_train_eval=True)
        time_fed = time.time() - t_start
        logger_fedavg = DataLogger(args, 'FedAvg', iid_type)
        logger_fedavg.check_history()
        for i, (acc, loss) in enumerate(zip(acc_fed, loss_fed)):
            logger_fedavg.log_epoch(i + 1, acc, loss)

    has_fedprox, data_fedprox = logger_fedprox.check_history()
    if has_fedprox and len(data_fedprox) == expected_rounds:
        print(f"[FedProx] 使用历史数据，共 {len(data_fedprox)} 轮")
        acc_prox = [d['accuracy'] for d in data_fedprox]
        loss_prox = [d['loss'] for d in data_fedprox]
        time_prox = 0.0
    else:
        if has_fedprox:
            print(f"[FedProx] 历史数据轮次({len(data_fedprox)})与当前配置({expected_rounds})不匹配，重新训练")
        t_start = time.time()
        _, _, acc_prox, loss_prox = run_fedprox(skip_train_eval=True, mu=0.5)
        time_prox = time.time() - t_start
        logger_fedprox = DataLogger(args, 'FedProx', iid_type)
        logger_fedprox.check_history()
        for i, (acc, loss) in enumerate(zip(acc_prox, loss_prox)):
            logger_fedprox.log_epoch(i + 1, acc, loss)

    has_salf, data_salf = logger_salf.check_history()
    if has_salf and len(data_salf) == expected_rounds:
        print(f"[SALF] 使用历史数据，共 {len(data_salf)} 轮")
        acc_salf = [d['accuracy'] for d in data_salf]
        loss_salf = [d['loss'] for d in data_salf]
        time_salf = 0.0
    else:
        if has_salf:
            print(f"[SALF] 历史数据轮次({len(data_salf)})与当前配置({expected_rounds})不匹配，重新训练")
        t_start = time.time()
        _, _, acc_salf, loss_salf = run_salf(skip_train_eval=True)
        time_salf = time.time() - t_start
        logger_salf = DataLogger(args, 'SALF', iid_type)
        logger_salf.check_history()
        for i, (acc, loss) in enumerate(zip(acc_salf, loss_salf)):
            logger_salf.log_epoch(i + 1, acc, loss)

    has_dmfl, data_dmfl = logger_dmfl.check_history()
    if has_dmfl and len(data_dmfl) == expected_rounds:
        print(f"[DMFL] 使用历史数据，共 {len(data_dmfl)} 轮")
        acc_dmfl = [d['accuracy'] for d in data_dmfl]
        loss_dmfl = [d['loss'] for d in data_dmfl]
        time_dmfl = 0.0
    else:
        if has_dmfl:
            print(f"[DMFL] 历史数据轮次({len(data_dmfl)})与当前配置({expected_rounds})不匹配，重新训练")
        t_start = time.time()
        _, _, acc_dmfl, loss_dmfl = run_dmfl(skip_train_eval=True)
        time_dmfl = time.time() - t_start
        logger_dmfl = DataLogger(args, 'DMFL', iid_type)
        logger_dmfl.check_history()
        for i, (acc, loss) in enumerate(zip(acc_dmfl, loss_dmfl)):
            logger_dmfl.log_epoch(i + 1, acc, loss)

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

    # 样式配置：markers = ['o', 's', '^', 'D'], lines = ['-', '--', '-.', ':']
    # ==== 精度对比图 ====
    fig1 = plt.figure(1, figsize=(10, 6))
    plot_shadow(plt, rnds, acc_fed, 'red', 'FedAvg', marker='o', linestyle='-')
    plot_shadow(plt, rnds, acc_prox, 'brown', 'FedProx', marker='s', linestyle='--')
    plot_shadow(plt, rnds, acc_salf, 'green', 'SALF', marker='^', linestyle='-.')
    plot_shadow(plt, rnds, acc_dmfl, 'blue', 'DMFL', marker='D', linestyle=':')

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
    plot_shadow(plt, rnds, loss_fed, 'red', 'FedAvg', marker='o', linestyle='-')
    plot_shadow(plt, rnds, loss_prox, 'brown', 'FedProx', marker='s', linestyle='--')
    plot_shadow(plt, rnds, loss_salf, 'green', 'SALF', marker='^', linestyle='-.')
    plot_shadow(plt, rnds, loss_dmfl, 'blue', 'DMFL', marker='D', linestyle=':')

    if args.warmup_rounds > 0:
        plt.axvline(args.warmup_rounds, color='gray', linestyle=':', label='Warm-up')

    plt.xlabel('Rounds', fontsize=LABEL_SIZE)
    plt.ylabel('Val Loss', fontsize=LABEL_SIZE)
    plt.title(f'{args.dataset_name.upper()} Loss Comparison', fontsize=TITLE_SIZE)

    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Results_Loss_{args.dataset_name}_Rounds{args.num_global_rounds}_Shadow.png', dpi=300)

    plt.show()