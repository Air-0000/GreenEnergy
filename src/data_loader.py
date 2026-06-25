"""
数据加载与预处理模块
从 OPSD 数据集加载风电/光伏/电价数据，并进行预处理
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
class DataLoader:
    """数据加载器"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.wind_data = None
        self.solar_data = None
        self.load_data = None
        self.price_data = None
        self.scaler_wind = MinMaxScaler()
        self.scaler_solar = MinMaxScaler()
    def load_and_preprocess(self):
        """加载并预处理数据"""
        print("加载数据...")
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        # 选择德国数据作为示例（数据最完整）
        print(f"数据集形状: {self.df.shape}")
        print(f"时间范围: {self.df.index.min()} 至 {self.df.index.max()}")
        # 提取所需列
        self.wind_data = self._extract_column_pattern('wind')
        self.solar_data = self._extract_column_pattern('solar')
        self.load_data = self._extract_column_pattern('load')
        self.price_data = self._extract_column_pattern('price')
        # 清理缺失值
        self.wind_data = self.wind_data.dropna()
        self.solar_data = self.solar_data.dropna()
        if self.price_data is not None:
            self.price_data = self.price_data.dropna()
        print(f"风电数据: {len(self.wind_data)} 条")
        print(f"光伏数据: {len(self.solar_data)} 条")
        return self
    def _extract_column_pattern(self, pattern):
        """提取符合模式的列并合并"""
        cols = [c for c in self.df.columns if pattern in c.lower() and 'actual' in c.lower()]
        if not cols:
            # 尝试其他模式
            cols = [c for c in self.df.columns if pattern in c.lower()]
        if not cols:
            print(f"警告: 未找到 {pattern} 相关列")
            return None
        # 合并多个国家/地区的数据（取平均）
        data = self.df[cols].mean(axis=1)
        # 如果全部是NaN，尝试单一列
        if data.isna().all():
            if len(cols) > 0:
                data = self.df[cols[0]]
        return data
    def explore_data(self):
        """探索数据特征"""
        print("\n数据探索...")
        # 基本统计
        print("\n风电功率统计:")
        print(self.wind_data.describe())
        print("\n光伏功率统计:")
        print(self.solar_data.describe())
        # 计算日均和季均
        daily_wind = self.wind_data.resample('D').mean()
        monthly_wind = self.wind_data.resample('ME').mean()
        print(f"\n风电日均功率: {daily_wind.mean():.2f} MW")
        print(f"光伏日均功率: {self.solar_data.resample('D').mean().mean():.2f} MW")
        return self
    def split_data(self, train_ratio=0.8):
        """划分训练集和测试集（风电）"""
        n = len(self.wind_data)
        split_idx = int(n * train_ratio)
        train = self.wind_data.iloc[:split_idx]
        test = self.wind_data.iloc[split_idx:]
        print(f"\n数据划分: 训练集 {len(train)} 条, 测试集 {len(test)} 条")
        return train, test
    def create_sequences(self, data, lookback):
        """创建时间序列的滑动窗口数据集"""
        X, y = [], []
        data_values = data.values
        for i in range(lookback, len(data_values)):
            # 特征：过去 lookback 个时刻的数据
            X.append(data_values[i-lookback:i])
            # 目标：下一个时刻的值
            y.append(data_values[i])
        return np.array(X).reshape(-1, lookback, 1), np.array(y)
    def create_features(self, data):
        """创建额外特征（时间特征、天气特征等）"""
        df_features = pd.DataFrame(index=data.index)
        # 时间特征
        df_features['hour'] = data.index.hour
        df_features['dayofweek'] = data.index.dayofweek
        df_features['month'] = data.index.month
        df_features['dayofyear'] = data.index.dayofyear
        # 周期性编码
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        # 添加滞后特征
        for lag in [1, 2, 3, 6, 12, 24]:
            df_features[f'lag_{lag}'] = data.shift(lag)
        # 滚动统计
        df_features['rolling_mean_6'] = data.rolling(6).mean()
        df_features['rolling_mean_24'] = data.rolling(24).mean()
        df_features['rolling_std_24'] = data.rolling(24).std()
        return df_features.fillna(0)
    def inverse_transform(self, data, scaler_type='wind'):
        """逆变换恢复原始尺度"""
        if scaler_type == 'wind':
            return self.scaler_wind.inverse_transform(data)
        else:
            return self.scaler_solar.inverse_transform(data)
if __name__ == '__main__':
    # 测试数据加载器
    loader = DataLoader('../data/opsd_time_series.csv')
    loader.load_and_preprocess()
    loader.explore_data()
    X, y = loader.create_sequences(loader.wind_data, lookback=24)
    print(f"\n序列数据形状: X={X.shape}, y={y.shape}")
