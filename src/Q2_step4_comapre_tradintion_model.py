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

    # ---------------------------- 6. 传统模型（能量消耗外推） ----------------------------
    # 从原始数据中获取时间列（已处理过的 terminaltime）
    time_col = 'terminaltime'

    # 然后在传统模型中计算 ΔSOC 也要用同样的 horizon
    def compute_energy_and_delta_soc_horizon(start_idx, raw_df, seq_len, horizon = config.PRED_HORIZON):
        win = raw_df.iloc[start_idx : start_idx + seq_len + horizon]  # 扩展到目标点
        # 能量计算延用前面修正过的梯形公式（只用 seq_len 个特征）
        v = raw_df['totalvoltage'].values[start_idx : start_idx + seq_len + 1]
        i = raw_df['totalcurrent'].values[start_idx : start_idx + seq_len + 1]
        t = raw_df[time_col].values[start_idx : start_idx + seq_len + 1]
        dt = np.diff(t)
        p = v[:-1] * i[:-1] + v[1:] * i[1:]
        energy = np.sum(p * dt) / 2.0
        # SOC 变化：从窗口最后时刻到目标时刻
        soc_last = raw_df['soc'].values[start_idx + seq_len - 1]
        soc_target = raw_df['soc'].values[start_idx + seq_len + horizon - 1]
        delta_soc = soc_target - soc_last
        return energy, delta_soc

    # 在训练集上计算所有序列的能量和ΔSOC
    train_start_indices = np.arange(len(raw_data) - config.SEQ_LENGTH)[:train_end]
    energies_train = []
    deltas_train = []
    for idx in train_start_indices:
        e, d = compute_energy_and_delta_soc_horizon(idx, raw_data, config.SEQ_LENGTH)
        energies_train.append(e)
        deltas_train.append(d)

    energies_train = np.array(energies_train)
    deltas_train = np.array(deltas_train)

    # 线性回归：ΔSOC = k * energy + b
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(energies_train.reshape(-1, 1), deltas_train)
    k = lin_reg.coef_[0]
    b = lin_reg.intercept_
    logger.info(f"传统模型拟合: ΔSOC = {k:.6f} * 时间 + {b:.4f}")

    # 在测试集上应用
    test_start_indices_all = start_indices[val_end:test_end]
    energies_test = []
    last_soc_test = []
    for idx in test_start_indices_all:
        e, _ = compute_energy_and_delta_soc_horizon(idx, raw_data, config.SEQ_LENGTH)  # 第二个返回值不用
        energies_test.append(e)
        last_soc_test.append(raw_data['soc'].values[idx + config.SEQ_LENGTH - 1])

    energies_test = np.array(energies_test)
    last_soc_test = np.array(last_soc_test)

    # 传统模型预测
    pred_delta = lin_reg.predict(energies_test.reshape(-1, 1))
    pred_traditional = last_soc_test + pred_delta
    pred_traditional = np.clip(pred_traditional, 0, 100)

    # 评估传统模型
    rmse_trad = np.sqrt(mean_squared_error(y_test_real, pred_traditional))
    mae_trad = mean_absolute_error(y_test_real, pred_traditional)
    r2_trad = r2_score(y_test_real, pred_traditional)
    logger.info(f"传统能量模型: RMSE={rmse_trad:.4f}%, MAE={mae_trad:.4f}%, R²={r2_trad:.4f}")

    def eval_model(y_true, y_pred, name):
        rmse_ = np.sqrt(mean_squared_error(y_true, y_pred))
        mae_ = mean_absolute_error(y_true, y_pred)
        r2_ = r2_score(y_true, y_pred)
        logger.info(f"{name}: RMSE={rmse_:.4f}%, MAE={mae_:.4f}%, R²={r2_:.4f}")
        return rmse_, mae_, r2_

    # 与深度学习模型对比
    eval_model(y_test_real, pred_test, "CNNLSTMAttention")
    logger.info("------ 以下为新增传统模型 ------")
    eval_model(y_test_real, pred_traditional, "能量外推模型")

    # 误差分布对比图（保留深度学习模型和传统模型）
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=30, alpha=0.6, label='CNNLSTMAttention', color='steelblue')
    ax.hist(pred_traditional - y_test_real, bins=30, alpha=0.6, label='Tradition', color='darkorange')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('SOC(after 180min) (SOC %)')
    ax.set_ylabel('Frequency')
    ax.set_title('Deep Learning vs Traditional Energy Model Error Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig('model_comparison_with_energy_180min.png', dpi=150)
    plt.show()
    logger.success("传统能量模型对比图已保存 model_comparison_with_energy_60min.png")