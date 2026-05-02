import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from loguru import logger
import sys

# 引入你自己的模块
from Q1_step2_create_model import CNNLSTMAttentionModel
from Q1_step1_preprocess import get_dataframe, create_sequences, TimeSeriesStandardScaler, TargetScaler
import config

if __name__ == "__main__":
    # ---------------------------- 全局设置 ----------------------------
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------------------------- 1. 加载模型与数据 ----------------------------
    model = CNNLSTMAttentionModel(
        input_channels=config.INPUT_CHANNELS,
        cnn_hidden=64,
        lstm_hidden=128,
        lstm_layers=2,
        num_heads=4
    )
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    x_scaler = joblib.load(config.X_SCALER_PATH)
    y_scaler = joblib.load(config.Y_SCALER_PATH)

    # 读取原始数据用于工况划分
    raw_data = get_dataframe(config.DATA_FILE)   # 包含 terminaltime 等列
    features_orig = raw_data[config.FEATURE_COLS].values
    target_orig = raw_data[config.TARGET_COLS].values.squeeze()   # SOC

    # 生成序列（同训练时一致），同时记录每个序列的起始时间索引
    X_seq, y_seq = create_sequences(raw_data, config.SEQ_LENGTH, config.FEATURE_COLS, config.TARGET_COLS)
    # 序列对应的起始行索引
    start_indices = np.arange(len(raw_data) - config.SEQ_LENGTH)

    # 按训练/验证/测试划分（与 train.py 同一比例）
    total = len(X_seq)
    train_end = int(total * config.TRAIN_RATE)
    val_end = int(total * (config.TRAIN_RATE + config.VAL_RATE))
    test_end = int(total * (config.TRAIN_RATE + config.VAL_RATE + config.TEST_RATE))

    X_test_seq = X_seq[val_end:test_end]
    y_test_seq = y_seq[val_end:test_end]
    test_start_idx = start_indices[val_end:test_end]

    # 标准化测试特征
    X_test_scaled = x_scaler.transform(X_test_seq)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)

    # 模型预测（标准化值 -> 反标准化）
    with torch.no_grad():
        pred_scaled, _ = model(X_test_tensor)
    pred_test = y_scaler.inverse_transform(pred_scaled.cpu().numpy())
    y_test_real = y_test_seq   # 真实值（标准化 -> 原始域）

    # 计算误差
    errors = pred_test - y_test_real
    abs_errors = np.abs(errors)

    # ---------------------------- 4. 误差分布直方图与精度指标 ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 误差直方图
    axes[0].hist(errors, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predict Error (SOC %)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution Histogram')

    # 绝对误差直方图
    axes[1].hist(abs_errors, bins=40, color='darkorange', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Absolute Error (SOC %)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Absolute Error Distribution Histogram')

    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=150)
    plt.show()

    # 精度指标打印
    rmse = np.sqrt(mean_squared_error(y_test_real, pred_test))
    mae = mean_absolute_error(y_test_real, pred_test)
    r2 = r2_score(y_test_real, pred_test)
    logger.info(f"测试集精度: RMSE={rmse:.4f}%, MAE={mae:.4f}%, R²={r2:.4f}")
