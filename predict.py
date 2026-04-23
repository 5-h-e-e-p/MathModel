import torch
import numpy as np
import pandas as pd
import joblib
from preprocess import create_sequences
from CNN import CNNLSTMAttentionModel
import config

# ---------- 1. 加载模型结构与权重 ----------
model = CNNLSTMAttentionModel(
    input_channels=config.INPUT_CHANNELS,
    cnn_hidden=64,
    lstm_hidden=128,
    lstm_layers=2,
    num_heads=4
)
model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
model.eval()

# ---------- 2. 加载标准化器 ----------
x_scaler = joblib.load("x_scaler.joblib")
y_scaler = joblib.load("y_scaler.joblib")

# ---------- 3. 读取新数据并构造序列 ----------
# !!! csv文件至少有 config.SEQ_LENGTH+1 行数据
new_data_path = r"testdata\vin.csv"
new_data = pd.read_csv(new_data_path)

# 保证列顺序与 config.FEATURE_COLS 一致
X_new_raw, _ = create_sequences(
    new_data,
    config.SEQ_LENGTH,
    config.FEATURE_COLS,
    config.TARGET_COLS
)
# 此时 X_new_raw 形状 (N, SEQ_LENGTH, N_FEATURES)

# ---------- 4. 对新特征进行标准化（用训练好的 x_scaler） ----------
X_new_scaled = x_scaler.transform(X_new_raw)   # 形状不变

# ---------- 5. 转为 PyTorch 张量并调整维度 ----------
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).permute(0, 2, 1)


# ---------- 6. 模型预测 ----------
with torch.no_grad():
    predictions_scaled, attn_weights = model(X_new_tensor)


# ---------- 7. 反标准化得到真实 SOC（%） ----------
predictions_real = y_scaler.inverse_transform(predictions_scaled.cpu().numpy())
# predictions_real 是一维 numpy 数组，单位与原始 SOC 相同（%）

print("预测 SOC 值（前 10 个）:", predictions_real[:10])