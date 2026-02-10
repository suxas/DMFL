import torch
import torch.nn as nn
# 直接复用您 config.py 中的 args，保证设备一致性
from config import args


class LogitLSTM(nn.Module):
    """
    LOGIT 算法的核心模型：基于 LSTM 的梯度轨迹预测器
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LogitLSTM, self).__init__()
        # input_dim 对应 Head 部分的参数维度 (例如 510)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # LSTM 输出 shape: (batch, seq_len, hidden)
        out, (hn, cn) = self.lstm(x)

        # 取最后一个时间步的隐状态进行预测
        last_hidden = out[:, -1, :]
        prediction = self.fc(last_hidden)
        return prediction