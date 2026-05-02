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

    # ---------------------------- 2. 定义工况标签 ----------------------------
    # 为每个测试序列计算工况指标
    speeds = []
    temps = []
    currents = []
    for i in range(len(test_start_idx)):
        start = test_start_idx[i]
        window = raw_data.iloc[start : start + config.SEQ_LENGTH]  # 输入窗口（含特征+时间）
        # 平均速度（单位取决于 terminaltime 单位，按 odometer diff / time diff）
        odo_diff = window['totalodometer'].iloc[-1] - window['totalodometer'].iloc[0]
        time_diff = window['terminaltime'].iloc[-1] - window['terminaltime'].iloc[0]
        speed = odo_diff / time_diff if time_diff > 0 else 0.0   # 若时间差为0，速度为0
        speeds.append(speed)
        # 平均温度：取 mintemperaturevalue 和 maxtemperaturevalue 的平均值再在窗口内取平均
        avg_temp = (window['mintemperaturevalue'] + window['maxtemperaturevalue']).mean() / 2.0
        temps.append(avg_temp)
        # 平均电流（绝对值）
        avg_current = window['totalcurrent'].abs().mean()
        currents.append(avg_current)

    speeds = np.array(speeds)
    temps = np.array(temps)
    currents = np.array(currents)

    # 高速/低速：按平均电流中位数划分（电流越大通常对应高功率/高速工况）
    current_median = np.median(currents)
    high_speed_idx = currents >= current_median
    low_speed_idx = currents < current_median

    # 高温/低温：按平均温度中位数划分
    temp_median = np.median(temps)
    high_temp_idx = temps >= temp_median
    low_temp_idx = temps < temp_median

    logger.info(f"工况划分完成: 高速样本 {high_speed_idx.sum()}, 低速样本 {low_speed_idx.sum()}")
    logger.info(f"高温样本 {high_temp_idx.sum()}, 低温样本 {low_temp_idx.sum()}")

    # ---------------------------- 3. 预测 vs 真值 对比图 ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_pairs = [
        (low_speed_idx, 'low_speed'),
        (high_speed_idx, 'high_speed'),
        (low_temp_idx, 'low_temperature'),
        (high_temp_idx, 'high_temperature')
    ]
    for ax, (mask, title) in zip(axes.flat, plot_pairs):
        ax.scatter(y_test_real[mask], pred_test[mask], alpha=0.5, edgecolors='none', s=20)
        ax.plot([0, 100], [0, 100], 'r--', linewidth=1)  # 对角线
        ax.set_xlabel('real SOC (%)')
        ax.set_ylabel('predict SOC (%)')
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('pred_vs_true_by_condition.png', dpi=150)
    plt.show()
    logger.success("工况对比图已保存 pred_vs_true_by_condition.png")
