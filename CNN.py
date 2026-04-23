import torch.nn as nn
import torch.nn.functional as F

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
        # lstm_out: (seq_len, batch, lstm_hidden)
        
        # ---------- Step 4: Attention（对时间步加权）----------
        # MultiheadAttention 输入格式：query, key, value 均为 (seq_len, batch, embed_dim)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_out: (seq_len, batch, lstm_hidden)
        # attn_weights: (batch, seq_len, seq_len)  (若 batch_first=False 时权重形状？实际返回 (L, B, S)？这里以官方文档为准，我们需要转换)
        # 注：PyTorch MultiheadAttention 在 batch_first=False 时返回权重形状 (seq_len, batch, seq_len)
        
        # ---------- Step 5: 聚合时间步特征（两种方式任选）----------
        # 方式 A：取最后一个时间步（简单，适合短序列）
        # context = lstm_out[-1, :, :]   # (batch, lstm_hidden)
        
        # 方式 B：对 Attention 输出做全局平均池化（推荐，保留整体时序信息）
        context = attn_out.mean(dim=0)    # (batch, lstm_hidden)
        
        # 方式 C：直接用 Attention 加权和（需手动计算加权和，但 MultiheadAttention 内部已加权）
        # 实际上 attn_out 已经是加权后的结果，取平均或取最后均可
        
        # ---------- Step 6: 全连接层输出续航 ----------
        pred = self.fc(context).squeeze(-1)   # (batch,)
        
        return pred, attn_weights
