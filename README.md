# 绿电+热电联产+加氢能源 多能互补协同调度系统
---

## 📁 项目结构

```
workspace/GreenEnergy_PDCA/
├── src/                        # 源代码
│   ├── main.py                 # 主程序入口
│   ├── data_loader.py         # 数据加载与预处理
│   ├── lstm_model.py           # LSTM 预测模型
│   ├── dispatcher.py           # 调度引擎
│   ├── llm_analyst.py          # 大模型分析师
│   └── evaluator.py            # 系统评估器
├── data/                       # 数据目录
│   ├── opsd_time_series.csv    # OPSD 欧洲电力数据集 (124MB)
│   └── SOLETE/                 # SOLETE 丹麦风光数据集
├── results/                    # 输出目录
│   ├── wind_lstm.pth           # 风电 LSTM 模型
│   ├── solar_lstm.pth          # 光伏 LSTM 模型
│   ├── dispatch_overview.png    # 调度概览图
│   ├── economics_carbon.png     # 经济碳排图
│   ├── results.json            # 结果 JSON
│   └── llm_analysis.txt        # 大模型分析结果
├── models/                     # 模型存储目录
├── reports/                    # 报告目录
└── requirements.txt            # 依赖包
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd workspace/GreenEnergy_PDCA
pip install -r requirements.txt
```

依赖包括：
- torch >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

### 2. 运行完整流程

```bash
cd workspace/GreenEnergy_PDCA
python3 src/main.py
```

---

## 📋 完整流程说明

### 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    完整运行流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [阶段1] 数据准备                                                │
│  ├── 加载 OPSD 数据集 (50401条, 299列)                           │
│  ├── 提取风电/光伏功率数据                                        │
│  └── 创建时间序列滑动窗口 (lookback=24小时)                       │
│                                                                  │
│  [阶段2] LSTM 模型训练                                          │
│  ├── 风电预测 LSTM (hidden_dim=64, layers=2)                     │
│  ├── 光伏预测 LSTM (hidden_dim=64, layers=2)                    │
│  ├── 早停机制 (patience=10)                                     │
│  └── 保存模型到 results/                                        │
│                                                                  │
│  [阶段3] 预测与调度                                              │
│  ├── 生成168小时预测 (一周数据)                                  │
│  ├── 规则调度: 优先制氢 + 余电热电联产                            │
│  └── 计算经济/环保指标                                           │
│                                                                  │
│  [阶段4] 大模型策略解读                                          │
│  ├── 调用 Claude/GPT API (需配置 key)                           │
│  └── 本地fallback模式 (无API时)                                 │
│                                                                  │
│  [阶段5] 系统评估                                                │
│  ├── 计算能效/经济/环保指标                                       │
│  ├── 生成可视化图表                                              │
│  └── 输出评估报告                                                │
│                                                                  │
│  [阶段6] 生成报告                                                │
│  ├── analysis_report.md (完整分析报告)                          │
│  └── results.json (结构化结果)                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 各模块功能

### 1. data_loader.py - 数据加载器

```python
from data_loader import DataLoader

loader = DataLoader('data/opsd_time_series.csv')
loader.load_and_preprocess()
loader.explore_data()

# 创建序列数据
X, y = loader.create_sequences(loader.wind_data, lookback=24)
# X shape: (samples, 24, 1)
# y shape: (samples,)
```

**数据源**: Open Power System Data (OPSD)
- 时间范围: 2014-12-31 至 2020-09-30
- 采样间隔: 1小时
- 包含: 32个欧洲国家的风电、光伏、负荷、电价数据

### 2. lstm_model.py - LSTM 预测模型

```python
from lstm_model import LSTMPredictor

# 创建预测器
predictor = LSTMPredictor(
    name='Wind',
    input_dim=1,
    hidden_dim=64,
    num_layers=2,
    lookback=24
)

# 训练
predictor.train(X, y, epochs=50)

# 预测
predictions = predictor.predict(X_test)

# 评估
results = predictor.evaluate(X_test, y_test)
# {'rmse': xxx, 'mae': xxx, 'mape': xxx}

# 保存/加载模型
predictor.save('results/wind_lstm.pth')
predictor.load('results/wind_lstm.pth')
```

**模型配置**:
| 参数 | 值 | 说明 |
|------|-----|------|
| input_dim | 1 | 单变量输入 |
| hidden_dim | 64 | 隐藏层维度 |
| num_layers | 2 | LSTM层数 |
| lookback | 24 | 历史窗口(小时) |
| dropout | 0.2 | Dropout比率 |
| optimizer | Adam | 优化器 |
| lr | 0.001 | 学习率 |

### 3. dispatcher.py - 调度引擎

```python
from dispatcher import DispatchEngine

config = {
    'electrolyzer_capacity': 500,    # 电解槽功率 (kW)
    'electrolyzer_efficiency': 0.7,   # 制氢效率
    'chp_capacity': 300,              # 热电联产功率 (kW)
    'chp_heat_ratio': 0.4,            # 热电比
    'h2_storage_capacity': 1000,      # 储氢容量 (kg)
    'electricity_price_low': 0.3,     # 谷时电价
    'electricity_price_high': 0.8,    # 峰时电价
}

dispatcher = DispatchEngine(config)

# 单次调度
result = dispatcher.dispatch(wind_power=300, solar_power=200)
# result: {'h2_produced': 714.29, 'carbon_savings': 400, ...}

# 批量调度
results = dispatcher.run_simulation(power_forecast_df)

# 总结结果
summary = dispatcher.summarize_results(results)
```

**调度规则**:
```
1. 优先使用绿电进行电解制氢
2. 过剩电力送往热电联产
3. 储氢罐作为缓冲调节
4. 峰谷电价联动调节 (8:00-20:00为峰时)
```

### 4. llm_analyst.py - 大模型分析师

```python
from llm_analyst import LLMAnalyst

# 方式1: 使用环境变量
# export OPENAI_API_KEY=sk-xxx 或 export CLAUDE_API_KEY=xxx

# 方式2: 直接传入
analyst = LLMAnalyst(api_key='your-api-key')

# 分析调度场景
analysis = analyst.analyze_scenario(scenario_summary)
```

**无API时**: 自动降级为本地分析模式

### 5. evaluator.py - 系统评估器

```python
from evaluator import SystemEvaluator

evaluator = SystemEvaluator('results')

metrics = evaluator.evaluate(dispatch_results, wind_forecast, solar_forecast)
```

**评估指标**:

| 类别 | 指标 | 说明 |
|------|------|------|
| 能效 | overall_efficiency | 系统综合能效 |
| 能效 | self_use_rate | 绿电自用率 |
| 能效 | curtailment_rate | 弃风弃光率 |
| 生产 | h2_production | 绿氢产量(kg) |
| 经济 | net_economic_benefit | 净经济效益(元) |
| 环保 | carbon_savings | 碳减排量(kg CO2) |

---

## 📊 输出文件说明

### results/dispatch_overview.png
三图合一：
1. 风电/光伏功率预测时序图
2. 功率调度分配堆叠图
3. 储氢水平变化图

### results/economics_carbon.png
1. 每小时经济收益柱状图
2. 碳减排量时序图

### results/results.json
```json
{
  "timestamp": "2026-05-20T...",
  "wind_results": {"rmse": 69.45, "mae": 45.83},
  "solar_results": {"rmse": 30.99, "mae": 18.19},
  "metrics": {
    "overall_efficiency": 1439.53,
    "h2_production": 120000.00,
    "carbon_savings": 106290.51,
    ...
  }
}
```

---

## 🔧 自定义配置

### 修改系统参数

编辑 `src/main.py` 中的 `self.config`:

```python
self.config = {
    'electrolyzer_capacity': 1000,   # 增大电解槽
    'electrolyzer_efficiency': 0.75, # 提升效率
    'chp_capacity': 500,            # 增大热电联产
    # ...
}
```

### 修改模型参数

```python
# main.py 中
self.wind_predictor = LSTMPredictor(
    name='Wind',
    input_dim=1,
    hidden_dim=128,    # 增大隐藏层
    num_layers=3,      # 增加层数
    lookback=48        # 增加历史窗口
)
```

### 添加新的气象特征

修改 `src/data_loader.py` 的 `create_features()` 方法。

---

## ⚠️ 常见问题

### Q: 训练时间太长怎么办？
A: 减少 epochs 或使用更小的数据集测试：
```python
predictor.train(X_small, y_small, epochs=10)
```

### Q: GPU 不被识别？
A: 检查 PyTorch CUDA 安装：
```python
import torch
print(torch.cuda.is_available())
```

### Q: 内存不足？
A: 减小 batch_size：
```python
predictor.train(X, y, epochs=50, batch_size=32)  # 默认64
```

---

## 📈 升级路线

```
大作业(现在)          课程论文(1个月)        毕设(3个月)          发表(6个月)
   │                     │                    │                    │
   ▼                     ▼                    ▼                    ▼
LSTM预测            强化学习(DQN/PPO)      多目标优化           多能流仿真+
规则调度           替代简单调度           NSGA-II             灵敏度分析
大模型解读          智能调度               不确定性处理         投核心期刊
```

---

## 📝 报告结构模板

根据评分标准，报告应包含：

1. **选题正确性 (15分)** - 绿电+热电联产+加氢能源 ✓
2. **方法适当性 (20分)** - LSTM预测+规则调度 ✓
3. **分析与实验设计 (40分)** - 需要详细展开
4. **创新性 (15分)** - 多能互补协同 ✓
5. **格式规范 (10分)** - 参照模板

---

*本指南由 AI 生成 - 2026-05-20*
