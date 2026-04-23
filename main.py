from preprocess import get_dataframe, create_sequences, TimeSeriesStandardScaler, TargetScaler
from CNN import CNNLSTMAttentionModel
import config
from loguru import logger
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import joblib

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

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                        batch_size=batch_size, shuffle=False)

model = CNNLSTMAttentionModel(
    input_channels=config.INPUT_CHANNELS,
    cnn_hidden=64,
    lstm_hidden=128,
    lstm_layers=2,
    num_heads=4
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
for epoch in range(50):
    # 训练
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        pred, _ = model(batch_x)           # pred shape: (batch,)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred, _ = model(batch_x)
            val_loss += criterion(pred, batch_y).item()
    val_loss /= len(val_loader)

    logger.info(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        joblib.dump(x_scaler, "x_scaler.joblib")
        joblib.dump(y_scaler, "y_scaler.joblib")

logger.success(f"最终训练出的模型Val Loss: {best_val_loss}")