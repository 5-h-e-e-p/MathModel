import torch.nn as nn
import torch.nn.functional as F
import config
from Q1_step1_preprocess import get_dataframe, create_sequences, TimeSeriesStandardScaler, TargetScaler
import torch
from loguru import logger
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import joblib

class FeatureExtractorCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels=64, kernel_size=5):
        super().__init__()
        # 第一层卷积：提取基础局部模式
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,   # 原始特征数（如5）
            out_channels=hidden_channels, # 卷积核个数（输出通道数）
            kernel_size=kernel_size,      # 一次看5个时间步
            padding=2                # 保持时间长度不变（方便后续LSTM）
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # 第二层卷积：组合更抽象的特征
        self.conv2 = nn.Conv1d(
            hidden_channels, 
            hidden_channels*2, 
            kernel_size=kernel_size, 
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels*2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        # 输出 shape: (batch, hidden_channels*2, seq_len)
        return x
    
class CNNLSTMAttentionModel(nn.Module):
    """
    输入原始数据形状：(batch, input_channels, seq_len)
    输出：续航预测值 (batch,)
    同时返回注意力权重用于可解释性分析
    """
    def __init__(self, input_channels, cnn_hidden=64, lstm_hidden=128, lstm_layers=2, num_heads=4):
        super().__init__()
        # CNN 部分
        self.cnn = FeatureExtractorCNN(input_channels, cnn_hidden)
        
        # LSTM 部分：输入维度 = CNN输出通道数 = cnn_hidden*2
        self.lstm = nn.LSTM(
            input_size=cnn_hidden*2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,      # 我们将手动设置为 (seq_len, batch, feature)
            dropout=0.2,
            bidirectional=False     # 单层LSTM；若用双向需调整后续维度
        )
        
        # Attention 部分：对 LSTM 所有时间步输出做自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=False       # 输入格式 (seq_len, batch, embed_dim)
        )
        
        # 最终回归层
        self.fc = nn.Linear(lstm_hidden, 1)
        
    def forward(self, x):
        """
        x: 原始输入，形状 (batch, input_channels, seq_len)
        """
        # ---------- Step 1: CNN ----------
        cnn_out = self.cnn(x)                     # (batch, cnn_features, seq_len)
        
        # ---------- Step 2: 维度转换，喂给 LSTM ----------
        # LSTM 期望输入 (seq_len, batch, input_size)
        lstm_input = cnn_out.permute(2, 0, 1)     # (seq_len, batch, cnn_features)
        
        # ---------- Step 3: LSTM ----------
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)  
    
        # ---------- Step 4: Attention（对时间步加权）----------
        # MultiheadAttention 输入格式：query, key, value 均为 (seq_len, batch, embed_dim)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # 方式 B：对 Attention 输出做全局平均池化（推荐，保留整体时序信息）
        context = attn_out.mean(dim=0)    # (batch, lstm_hidden)
        
        pred = self.fc(context).squeeze(-1)   # (batch,)
        
        return pred, attn_weights
    
if __name__ == "__main__":
    data = get_dataframe(config.DATA_FILE)
    X_raw, y_raw = create_sequences(data, config.SEQ_LENGTH, config.FEATURE_COLS, config.TARGET_COLS)
    logger.info(f"原始数据 X: {X_raw.shape}, y: {y_raw.shape}")

    total = len(X_raw)
    train_end = int(total * config.TRAIN_RATE)
    val_end = int(total * (config.TRAIN_RATE + config.VAL_RATE))
    test_end = int(total * (config.TRAIN_RATE + config.VAL_RATE + config.TEST_RATE))

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
    for epoch in range(5):
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
            torch.save(model.state_dict(), f"horizon={config.PRED_HORIZON}/best_model.pth")
            joblib.dump(x_scaler, "horizon={config.PRED_HORIZON}/x_scaler.joblib")
            joblib.dump(y_scaler, "horizon={config.PRED_HORIZON}/y_scaler.joblib")

    logger.success(f"最终训练出的模型Val Loss: {best_val_loss}")