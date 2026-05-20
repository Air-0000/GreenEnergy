"""
LSTM 预测模型模块
用于风电/光伏功率的多步预测
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class LSTMModel(nn.Module):
    """LSTM 预测模型"""

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # LSTM 前向传播
        out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)
        return out


class LSTMPredictor:
    """LSTM 预测器封装类"""

    def __init__(self, name='LSTM', input_dim=1, hidden_dim=64, num_layers=2, lookback=24):
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lookback = lookback

        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  [{name}] 使用设备: {self.device}")

        # 初始化模型
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)

        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # 训练历史
        self.train_history = []
        self.scaler = None

    def train(self, X, y, epochs=50, batch_size=64, validation_split=0.2):
        """训练模型"""
        # 数据预处理：归一化
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X.reshape(-1, self.input_dim)).reshape(X.shape)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # 划分训练集和验证集
        n = len(X_scaled)
        val_size = int(n * validation_split)
        train_size = n - val_size

        X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_val = y_scaled[:train_size], y_scaled[train_size:]

        # 转换为 Tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # 创建 DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print(f"  [{self.name}] 训练集: {train_size}, 验证集: {val_size}")

        # 训练循环
        self.model.train()
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t).item()

            self.train_history.append({'epoch': epoch, 'train_loss': avg_loss, 'val_loss': val_loss})

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [{self.name}] 早停于第 {epoch+1} 轮，验证损失: {best_val_loss:.6f}")
                    break

            # 每10轮打印一次
            if (epoch + 1) % 10 == 0:
                print(f"  [{self.name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 恢复最佳模型
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        print(f"  [{self.name}] 训练完成，最佳验证损失: {best_val_loss:.6f}")

    def predict(self, X):
        """预测"""
        self.model.eval()

        # 归一化
        X_scaled = self.scaler.transform(X.reshape(-1, self.input_dim)).reshape(X.shape)
        X_t = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy().flatten()

        # 逆归一化
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions

    def evaluate(self, X, y):
        """评估模型"""
        predictions = self.predict(X)

        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)

        # 计算 MAPE（避免除零）
        mask = y != 0
        mape = np.mean(np.abs((y[mask] - predictions[mask]) / y[mask])) * 100 if mask.any() else 0

        print(f"  [{self.name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': predictions,
            'actuals': y
        }

    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'lookback': self.lookback
            }
        }, path)
        print(f"  [{self.name}] 模型已保存至 {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        print(f"  [{self.name}] 模型已加载")

    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        if not self.train_history:
            return

        epochs = [h['epoch'] for h in self.train_history]
        train_losses = [h['train_loss'] for h in self.train_history]
        val_losses = [h['val_loss'] for h in self.train_history]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.name} Training History')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"训练历史图已保存至 {save_path}")
        else:
            plt.show()


if __name__ == '__main__':
    # 测试 LSTM 模型
    print("测试 LSTM 模型...")

    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    t = np.linspace(0, 100, n)
    data = np.sin(t * 0.1) + 0.5 * np.sin(t * 0.3) + np.random.randn(n) * 0.1

    lookback = 24
    X, y = [], []
    for i in range(lookback, n):
        X.append(data[i-lookback:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 训练
    predictor = LSTMPredictor(name='Test', input_dim=1, hidden_dim=32, num_layers=1)
    predictor.train(X, y, epochs=20)

    # 预测
    results = predictor.evaluate(X, y)
    print(f"测试结果: RMSE={results['rmse']:.4f}")