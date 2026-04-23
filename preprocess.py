import pandas as pd
import numpy as np
import os
from loguru import logger
import config
from sklearn.preprocessing import StandardScaler

class TimeSeriesStandardScaler:
    """
    时序数据标准化器：仅用训练集的统计量进行 fit，然后 transform 任意集。
    X 的形状：(N, seq_len, features)
    """
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

class TargetScaler:
    """对一维目标（如 SOC）做标准化"""
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

import sys

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

if __name__ == "__main__":
    process_csv(r"data\vin17.csv", r"data\vin17_processed.csv")
    process_csv(r"data\vin1.csv", r"data\vin1_processed.csv")