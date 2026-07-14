import os
import json
import hashlib
from datetime import datetime


class DataLogger:
    def __init__(self, config, algo_type, iid_type, base_dir="log"):
        """
        初始化 Logger，自动建立文件夹、计算参数哈希，并判断是否已有历史实验
        :param config: 实验的配置参数 (通常为 args 或 config 对象/字典)
        :param algo_type: 算法种类 (如 'DMFL', 'FedAvg')
        :param iid_type: 数据分布类型 (如 'IID', 'Non-IID')
        """
        self.algo_type = algo_type
        self.iid_type = iid_type
        self.base_dir = base_dir

        # 提取可序列化的参数并生成 MD5 哈希
        self.config_dict = self._get_serializable_dict(config)
        self.epochs = self.config_dict.get('num_global_rounds', 100)  # 尝试获取轮次，默认 100
        config_str = json.dumps(self.config_dict, sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()

        # 建立算法专属文件夹
        self.log_dir = os.path.join(self.base_dir, self.algo_type)
        os.makedirs(self.log_dir, exist_ok=True)

        self.filepath = None
        self.history_data = []

    def _get_serializable_dict(self, config):
        """过滤配置中的不可序列化对象(如 device, model等)"""
        if hasattr(config, '__dict__'):
            config_dict = vars(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}

        safe_dict = {}
        for k, v in config_dict.items():
            if isinstance(v, (int, float, str, bool, tuple, list)):
                safe_dict[k] = v
        return safe_dict

    def check_history(self):
        """检查是否存在相同参数配置的历史 JSON 文件"""
        for filename in os.listdir(self.log_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.log_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get("config_hash") == self.config_hash:
                            print(f"[{self.algo_type}] 匹配到相同配置的历史数据: {filename}")
                            self.history_data = data.get("data", [])
                            self.filepath = filepath
                            return True, self.history_data
                except Exception:
                    continue

        # 未找到匹配项，新建文件
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.epochs}_{self.iid_type}_{current_time}.json"
        self.filepath = os.path.join(self.log_dir, filename)

        # 初始化空文件
        self._save()
        print(f"[{self.algo_type}] 未找到历史数据，新建仿真记录: {self.filepath}")
        return False, []

    def log_epoch(self, epoch, accuracy, loss):
        """记录每轮数据并缓存到 JSON 文件"""
        self.history_data.append({
            "epoch": epoch,
            "accuracy": accuracy,
            "loss": loss
        })
        self._save()

    def _save(self):
        """写入 JSON 文件"""
        save_dict = {
            "config": self.config_dict,
            "config_hash": self.config_hash,
            "data": self.history_data
        }
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)