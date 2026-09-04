import os
import re
import json
from datetime import datetime


class DataLogger:
    def __init__(self, config, algo_type, iid_type, base_dir="log"):
        self.algo_type = algo_type
        self.iid_type = iid_type
        self.base_dir = base_dir

        self.dataset = getattr(config, 'dataset_name', 'unknown')
        self.straggler_rate = getattr(config, 'target_straggler_rate', 0.0)
        self.rounds = getattr(config, 'num_global_rounds', 100)

        self.log_dir = os.path.join(self.base_dir, self.algo_type)
        os.makedirs(self.log_dir, exist_ok=True)

        self.filepath = None
        self.history_data = []

    def _make_prefix(self):
        return f"{self.dataset}_{self.iid_type}_strag{self.straggler_rate}_R{self.rounds}"

    def _make_filename(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self._make_prefix()}_{ts}.json"

    def _parse_prefix(self, filename):
        """从文件名中解析参数前缀，返回 (dataset, iid, straggler, rounds) 或 None"""
        m = re.match(
            r'^(.+)_(.+?)_strag([\d.]+)_R(\d+)_\d{8}_\d{6}\.json$',
            filename
        )
        if m:
            return m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
        return None

    def check_history(self):
        """通过文件名前缀参数匹配历史实验数据"""
        prefix = (self.dataset, self.iid_type, self.straggler_rate, self.rounds)
        for filename in os.listdir(self.log_dir):
            if not filename.endswith(".json"):
                continue
            parsed = self._parse_prefix(filename)
            if parsed == prefix:
                filepath = os.path.join(self.log_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"[{self.algo_type}] 匹配到历史数据: {filename}")
                    self.history_data = data.get("data", [])
                    self.filepath = filepath
                    return True, self.history_data
                except Exception:
                    continue

        self.filepath = os.path.join(self.log_dir, self._make_filename())
        self._save()
        print(f"[{self.algo_type}] 未找到历史数据，新建: {os.path.basename(self.filepath)}")
        return False, []

    def log_epoch(self, epoch, accuracy, loss):
        self.history_data.append({
            "epoch": epoch,
            "accuracy": accuracy,
            "loss": loss
        })
        self._save()

    def _save(self):
        save_dict = {
            "dataset": self.dataset,
            "iid_type": self.iid_type,
            "straggler_rate": self.straggler_rate,
            "rounds": self.rounds,
            "data": self.history_data
        }
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)
