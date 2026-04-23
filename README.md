# 预处理

1. 删除最后两列（batteryvoltage,probetemperatures）
2. 删除所有含空白单元格的数据行
3. 删除所有chargestatus=255的数据行(电量状态异常)
4. 将第一列terminaltime全部减去第二行第一列（B1）的数据'

```python
def process_csv(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    print(f"读取{input_file}成功")

    # 1. 删除最后两列
    cols_to_drop = ['batteryvoltage', 'probetemperatures']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"已删除多余数据")

    # 2. 删除所有含空白单元格的数据行
    df.dropna(inplace=True)
    print(f"已删除空白数据")

    # 3. 删除 chargestatus = 255 的数据行
    if 'chargestatus' in df.columns:
        df = df[df['chargestatus'] != 255]
    print(f"已删除异常数据")

    # 4. 将 terminaltime 列的全部值减去第二个数据行（第一行数据）的 terminaltime 值
    if not df.empty and 'terminaltime' in df.columns:
        base_time = df['terminaltime'].iloc[0]   # 第一个数据行的terminaltime
        df['terminaltime'] = df['terminaltime'] - base_time
    print(f"已处理完时间戳")

    # 保存处理后的文件
    df.to_csv(output_file, index=False)
    print(f"处理完成，输出文件: {output_file}")
```

# 数据标准化

由于温度、电压等特征幅值范围大，所以进行 Z-score 标准化处理：
    
## 时序数据标准化器

仅用训练集的统计量进行 fit，然后 transform 任意集。

```python
class TimeSeriesStandardScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """在训练集上用 (N*seq_len, features) 的方式拟合"""
        N, seq, f = X.shape
        # 将前两个维度合并，按特征拟合
        self.scaler.fit(X.reshape(-1, f))
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """变换，保持形状不变"""
        N, seq, f = X.shape
        return self.scaler.transform(X.reshape(-1, f)).reshape(N, seq, f)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """逆变换，同上"""
        N, seq, f = X_scaled.shape
        return self.scaler.inverse_transform(X_scaled.reshape(-1, f)).reshape(N, seq, f)
```

## SOC标准化

```python
class TargetScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, y: np.ndarray):
        self.scaler.fit(y.reshape(-1, 1))
        self.is_fitted = True

    def transform(self, y: np.ndarray) -> np.ndarray:
        return self.scaler.transform(y.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
```

# 组织输入张量

读取csv与构建滑动窗口样本函数

```python
def get_dataframe(data_file_path: str) -> pd.DataFrame:
    data_file_path = os.path.join(config.DATA_FOLDER, data_file_path)
    data:pd.DataFrame = None
    try:
        if not os.path.exists(data_file_path):
            raise ValueError("文件不存在")
        try:
            if data_file_path[-4:] == ".csv":
                data = pd.read_csv(data_file_path)
            elif data_file_path[-5:] == ".xlsx":
                data = pd.read_excel(data_file_path)
            else:
                raise ValueError(f"{data_file_path}文件格式超出处理范围")
        except Exception as e:
            logger.error(f"文件格式错误: {e}")
    except Exception as e:
        logger.error(f"获取DataFrame时出错: {e}")
    return data

def create_sequences(data:pd.DataFrame, seq_length:int, feature_cols:list[str], target_cols:list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_data = data[feature_cols].values
    y_data = data[target_cols].values
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(x_data[i:i+seq_length])
        y.append(y_data[i+seq_length])
    return np.array(X), np.array(y).reshape(-1)
```

使用示例:

```python
X_raw, y_raw = create_sequences(data, config.SEQ_LENGTH, config.FEATURE_COLS, config.TARGET_COLS)
logger.info(f"原始数据 X: {X_raw.shape}, y: {y_raw.shape}")

total = len(X_raw)
train_end = int(total * 0.07)
val_end = int(total * 0.085)

X_train, y_train = X_raw[:train_end], y_raw[:train_end]
X_val, y_val = X_raw[train_end:val_end], y_raw[train_end:val_end]
X_test, y_test = X_raw[val_end:], y_raw[val_end:]
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
```

# CNN

```python
class FeatureExtractorCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels=64, kernel_size=5):
        super().__init__()
        # 第一层卷积
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # 第二层卷积
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

```

# CNN-LSTM-Attention

```python
class CNNLSTMAttentionModel(nn.Module):
    """
    输入原始数据形状：(batch, input_channels, seq_len)
    输出：续航预测值 (batch,)
    同时返回注意力权重用于可解释性分析
    """
    def __init__(self, input_channels, cnn_hidden=64, lstm_hidden=128, lstm_layers=2, num_heads=4):
        super().__init__()
        # CNN
        self.cnn = FeatureExtractorCNN(input_channels, cnn_hidden)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_hidden*2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,
            dropout=0.2,
            bidirectional=False
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=False
        )
        
        # 最终回归层
        self.fc = nn.Linear(lstm_hidden, 1)
        
    def forward(self, x):
        cnn_out = self.cnn(x)
        
        lstm_input = cnn_out.permute(2, 0, 1)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)  
        
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 对 Attention 输出做全局平均池化（保留整体时序信息）
        context = attn_out.mean(dim=0) 

        pred = self.fc(context).squeeze(-1) 
        
        return pred, attn_weights
```