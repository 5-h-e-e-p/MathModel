import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Q1_step1_preprocess import create_sequences
import joblib
from loguru import logger
import sys
import os

# 引入已有模块
from Q1_step2_create_model import CNNLSTMAttentionModel        # 你的 Attention 模型
from Q1_step1_preprocess import (get_dataframe, create_sequences,
                                 TimeSeriesStandardScaler, TargetScaler)
import config

# ---------------------------- 全局设置 ----------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------- 预测步长（可调） ----------------------------
PRED_HORIZON = 30          # 预测未来第30步的SOC
SEQ_LEN = config.SEQ_LENGTH

# ---------------------------- 1. 定义纯 LSTM 模型 ----------------------------
class LSTMModel(nn.Module):
    """
    纯粹 LSTM + 全连接回归器，结构与 CNN‑LSTM‑Attention 可比
    输入: (batch, input_channels, seq_len)  需要转换到 (seq_len, batch, input_size)
    """
    def __init__(self, input_channels, lstm_hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,          # 我们将手动排列为 (seq_len, batch, input)
            dropout=dropout,
            bidirectional=False
        )
        self.fc = nn.Linear(lstm_hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, channels, seq_len) -> (seq_len, batch, channels)
        x = x.permute(2, 0, 1)          # (seq_len, batch, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[-1]         # (batch, lstm_hidden)
        last_out = self.dropout(last_out)
        pred = self.fc(last_out).squeeze(-1)   # (batch,)
        return pred


# ---------------------------- 2. 加载数据并划分 ----------------------------
raw_data = get_dataframe(config.DATA_FILE)
X_raw, y_raw = create_sequences(
    raw_data, SEQ_LEN, config.FEATURE_COLS, config.TARGET_COLS
)
logger.info(f"序列数据 X: {X_raw.shape}, y: {y_raw.shape}")

total = len(X_raw)
train_end = int(total * 0.7)
val_end   = int(total * 0.85)
test_end  = int(total * 1.0)

X_train, y_train = X_raw[:train_end], y_raw[:train_end]
X_val, y_val     = X_raw[train_end:val_end], y_raw[train_end:val_end]
X_test, y_test   = X_raw[val_end:test_end], y_raw[val_end:test_end]

# ---------------------------- 4. 加载已保存的标准化器（与训练 CNN 时完全相同） ----------------------------
x_scaler = joblib.load(config.X_SCALER_PATH)          # 如果文件名不同请修改
y_scaler = joblib.load(config.Y_SCALER_PATH)
logger.info("已加载标准化器")

# 只 transform，不重新 fit（确保与 CNN 训练时一致的分布）
X_train_scaled = x_scaler.transform(X_train)
X_val_scaled   = x_scaler.transform(X_val)
X_test_scaled  = x_scaler.transform(X_test)

# 注意：y 不做 transform，只用于训练 LSTM 时计算损失（需要标准化）和最终评估（原始值）
y_train_scaled = y_scaler.transform(y_train)
y_val_scaled   = y_scaler.transform(y_val)
# y_test 保持原始值，最终评估用

# 转 tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
X_val_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val_scaled, dtype=torch.float32)

from torch.utils.data import DataLoader, TensorDataset
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                          batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                          batch_size=batch_size, shuffle=False)

# ---------------------------- 5. 加载已训好的 CNN‑LSTM‑Attention 模型 ----------------------------
cnn_attn_model = CNNLSTMAttentionModel(
    input_channels=config.INPUT_CHANNELS,
    cnn_hidden=64,
    lstm_hidden=128,
    lstm_layers=2,
    num_heads=4
)
cnn_attn_model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=DEVICE))
cnn_attn_model.to(DEVICE)
cnn_attn_model.eval()
logger.success("已加载 CNN‑LSTM‑Attention 模型")

# ---------------------------- 6. 训练纯 LSTM 模型 ----------------------------
def train_model(model, train_loader, val_loader, model_name, epochs=5, lr=0.001):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
        val_loss /= len(val_loader)

        logger.info(f"[{model_name}] Epoch {epoch+1:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

    model.load_state_dict(best_weights)
    logger.success(f"[{model_name}] 训练完成，最佳验证损失: {best_val_loss:.6f}")
    return model, best_val_loss

lstm_model = LSTMModel(
    input_channels=config.INPUT_CHANNELS,
    lstm_hidden=128,
    lstm_layers=2
)
LSTM_MODEL_PATH = "lstm_best_model.pth"
if os.path.exists(LSTM_MODEL_PATH):
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
    lstm_model.to(DEVICE)
    lstm_model.eval()
    logger.success(f"已加载缓存的 LSTM 模型：{LSTM_MODEL_PATH}")
else:
    logger.info("未找到缓存的 LSTM 模型，开始训练...")
    lstm_model, best_loss = train_model(lstm_model, train_loader, val_loader, LSTM_MODEL_PATH)
    torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
    logger.success(f"LSTM 模型训练完成（最佳验证损失 {best_loss:.6f}），已保存至 {LSTM_MODEL_PATH}")
# ---------------------------- 7. 测试评估（原始 SOC 域） ----------------------------
def evaluate_model(model, X_test_tensor, y_test_true, scaler, model_name):
    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor.to(DEVICE))
        # 如果输出是 tuple（如 CNN 返回 pred, attn），只取第一个元素
        if isinstance(output, tuple):
            preds_scaled = output[0].cpu().numpy()
        else:
            preds_scaled = output.cpu().numpy()
    preds = scaler.inverse_transform(preds_scaled)
    rmse = np.sqrt(mean_squared_error(y_test_true, preds))
    mae  = mean_absolute_error(y_test_true, preds)
    r2   = r2_score(y_test_true, preds)
    logger.info(f"[{model_name}] Test RMSE: {rmse:.4f}%, MAE: {mae:.4f}%, R²: {r2:.4f}")
    return preds, rmse, mae, r2

preds_cnn, rmse_cnn, mae_cnn, r2_cnn = evaluate_model(
    cnn_attn_model, X_test_tensor, y_test, y_scaler, "CNN-LSTM-Attn")
preds_lstm, rmse_lstm, mae_lstm, r2_lstm = evaluate_model(
    lstm_model, X_test_tensor, y_test, y_scaler, "Pure LSTM")

# ---------------------------- 8. 可视化对比 ----------------------------
# 8.1 预测 vs 真值散点图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, name, rmse_val in zip(axes,
                                     [preds_cnn, preds_lstm],
                                     ["CNN-LSTM-Attention", "Pure LSTM"],
                                     [rmse_cnn, rmse_lstm]):
    ax.scatter(y_test, preds, alpha=0.4, s=15, edgecolors='none')
    ax.plot([0, 100], [0, 100], 'r--')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('True SOC (%)')
    ax.set_ylabel('Predicted SOC (%)')
    ax.set_title(name)
    ax.set_aspect('equal')
    ax.text(5, 95, f'RMSE={rmse_val:.2f}%', fontsize=10, color='red', va='top')
plt.tight_layout()
plt.savefig('comparison_pred_vs_true.png', dpi=150)
plt.show()

# 8.2 误差分布直方图
fig, ax = plt.subplots(figsize=(8, 5))
errors_cnn = preds_cnn - y_test
errors_lstm = preds_lstm - y_test
ax.hist(errors_cnn, bins=30, alpha=0.6, label='CNN-LSTM-Attn', color='steelblue')
ax.hist(errors_lstm, bins=30, alpha=0.6, label='Pure LSTM', color='darkorange')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Prediction Error (SOC %)')
ax.set_ylabel('Frequency')
ax.set_title('Error Distribution Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('comparison_error_dist.png', dpi=150)
plt.show()

# 8.3 时序对比（测试集前200点）
sample_len = min(200, len(y_test))
t = np.arange(sample_len)
plt.figure(figsize=(14, 5))
plt.plot(t, y_test[:sample_len], 'k-', linewidth=2, label='True SOC')
plt.plot(t, preds_cnn[:sample_len], 'b-', alpha=0.8, label='CNN-LSTM-Attn')
plt.plot(t, preds_lstm[:sample_len], 'orange', alpha=0.8, label='Pure LSTM')
plt.xlabel('Sequence Index')
plt.ylabel('SOC (%)')
plt.title('Prediction Sequence Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparison_sequence.png', dpi=150)
plt.show()

logger.success("对比完成，图表已保存。")
