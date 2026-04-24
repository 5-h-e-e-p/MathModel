import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from CNN import CNNLSTMAttentionModel
import config
import joblib
from preprocess import get_dataframe, create_sequences, TimeSeriesStandardScaler, TargetScaler
from loguru import logger

# ============================
# 1. 加载模型与数据
# ============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型
model = CNNLSTMAttentionModel(
    input_channels=config.INPUT_CHANNELS,
    cnn_hidden=64,
    lstm_hidden=128,
    lstm_layers=2,
    num_heads=4
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# 标准化器（训练时保存的）
x_scaler = joblib.load("x_scaler.joblib")
y_scaler = joblib.load("y_scaler.joblib")

# 加载训练数据张量（你在 main.py 中生成的 X_train_tensor）
# 假设保存在文件里，或用同样的流程重新生成，这里直接引用训练时的变量。
# 如果你的训练脚本没有保存张量，可以用同样的 create_sequences 流程重新生成。
# 这里示例：从保存的 .pt 文件加载
# 标签不需要参与 SHAP 计算

# 背景样本：取一小部分训练数据作为 SHAP 的背景（推荐100~200个）

# 测试样本：要解释的样本（取验证集或测试集前几十个）

data = get_dataframe(config.DATA_FILE)
X_raw, y_raw = create_sequences(data, config.SEQ_LENGTH, config.FEATURE_COLS, config.TARGET_COLS)
logger.info(f"原始数据 X: {X_raw.shape}, y: {y_raw.shape}")

total = len(X_raw)
train_end = int(total * 0.07)
val_end = int(total * 0.085)
test_end = int(total * 0.1)

X_train, y_train = X_raw[:train_end], y_raw[:train_end]
X_val, y_val = X_raw[train_end:val_end], y_raw[train_end:val_end]
X_test, y_test = X_raw[val_end:test_end], y_raw[val_end:test_end]
logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

x_scaler = TimeSeriesStandardScaler()
x_scaler.fit(X_train)
X_train_scaled = x_scaler.transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = TargetScaler()
y_scaler.fit(y_train)
y_train_scaled = y_scaler.transform(y_train)
y_val_scaled = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

n_background = 200
background = X_train_tensor[:n_background]

n_explain = 50
test_samples = X_test_tensor[:n_explain]

C, L = config.INPUT_CHANNELS, config.SEQ_LENGTH       # 通道数、时间步数
flat_dim = C * L                                       # 展平后的特征总数（如 8*10=80）

# 展平为 2D 数组
flat_background = background.cpu().numpy().reshape(n_background, flat_dim)
flat_test       = test_samples.cpu().numpy().reshape(n_explain, flat_dim)

# ============================
# 2. 包装模型（支持展平输入）
# ============================
def model_predict(x_flat):
    """
    x_flat: numpy array, shape (batch, flat_dim)
    返回: numpy array, shape (batch,)
    """
    batch = x_flat.shape[0]
    # 恢复为 (batch, channels, seq_len)
    x_3d = x_flat.reshape(batch, C, L)
    tensor = torch.tensor(x_3d, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred, _ = model(tensor)
    return pred.cpu().numpy()

# ============================
# 3. 创建 SHAP 解释器（Permutation，基于展平表格）
# ============================
explainer = shap.PermutationExplainer(
    model_predict,
    masker=shap.maskers.Independent(flat_background),
    max_evals=500                        # 每个样本随机置换 500 次
)

# 计算 SHAP 值（展平格式）
shap_result = explainer(flat_test)
shap_values_flat = shap_result.values    # (n_explain, flat_dim)
print("SHAP values shape (flat):", shap_values_flat.shape)

# 恢复为 (n_explain, channels, seq_len) 用于时间步分析
shap_values_3d = shap_values_flat.reshape(n_explain, C, L)

# ============================
# 4. 全局特征重要性（按原始特征聚合）
# ============================
feature_names = config.FEATURE_COLS
# 对每个通道的所有时间步 SHAP 绝对值取平均
global_importance = np.abs(shap_values_3d).mean(axis=(0, 2))  # (C,)
sorted_idx = np.argsort(global_importance)[::-1]

print("\n===== 特征全局重要性（所有时间步平均 |SHAP|）=====")
for i in sorted_idx:
    print(f"{feature_names[i]:25s}: {global_importance[i]:.6f}")

# 条形图
plt.figure(figsize=(10, 5))
plt.barh(range(len(sorted_idx)), global_importance[sorted_idx], color='skyblue')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Mean |SHAP|')
plt.title('Feature Importance (Global)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=150)
plt.show()

# ============================
# 5. 单样本时间步热力图
# ============================
sample_idx = 0
sample_shap_3d = shap_values_3d[sample_idx]   # (C, L)

plt.figure(figsize=(12, 6))
plt.imshow(sample_shap_3d, aspect='auto', cmap='RdBu')
plt.colorbar(label='SHAP value')
plt.xticks(range(L), [f't-{L-i}' for i in range(L)])
plt.yticks(range(C), feature_names)
plt.title(f'SHAP Time-Step Heatmap for Sample {sample_idx}')
plt.tight_layout()
plt.savefig('shap_time_heatmap.png', dpi=150)
plt.show()

# ============================
# 6. summary_plot（绘制展平特征的重要性）
# ============================
# 构建展平特征名
flat_names = [f"{feat}_t{t}" for t in range(L) for feat in feature_names]

plt.figure(figsize=(15, 10))
shap.summary_plot(shap_values_flat, flat_test, feature_names=flat_names, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=150)
plt.show()

# 保存结果
np.savez('shap_results.npz',
         shap_values_flat=shap_values_flat,
         shap_values_3d=shap_values_3d,
         feature_names=np.array(feature_names),
         global_importance=global_importance,
         flat_test=flat_test)
