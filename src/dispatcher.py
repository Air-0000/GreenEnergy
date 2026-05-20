"""
调度引擎模块
实现绿氢-热电联产协同调度规则
"""

import numpy as np
import pandas as pd
from datetime import datetime, time


class DispatchEngine:
    """协同调度引擎"""

    def __init__(self, config):
        self.config = config
        self.history = []

        # 调度规则参数
        self.electrolyzer_capacity = config.get('electrolyzer_capacity', 500)  # kW
        self.electrolyzer_efficiency = config.get('electrolyzer_efficiency', 0.7)  # kWh/kg H2

        self.chp_capacity = config.get('chp_capacity', 300)  # kW
        self.chp_heat_ratio = config.get('chp_heat_ratio', 0.4)
        self.chp_efficiency = config.get('chp_efficiency', 0.85)

        self.h2_storage_capacity = config.get('h2_storage_capacity', 1000)  # kg
        self.h2_storage_level = config.get('h2_storage_level', 500)  # 当前储量

        self.electricity_price_low = config.get('electricity_price_low', 0.3)
        self.electricity_price_high = config.get('electricity_price_high', 0.8)

    def dispatch(self, wind_power, solar_power, timestamp=None):
        """
        核心调度逻辑

        参数:
            wind_power: 风电预测功率 (kW)
            solar_power: 光伏预测功率 (kW)

        返回:
            dict: 调度结果
        """
        total_green_power = wind_power + solar_power

        # ========== 步骤1: 电解制氢 ==========
        # 优先使用绿电制氢
        electrolyzer_power = min(total_green_power, self.electrolyzer_capacity)
        h2_produced = electrolyzer_power * 1.0 / self.electrolyzer_efficiency  # kg/h

        # 更新储氢量
        new_h2_level = self.h2_storage_level + h2_produced
        h2_stored = max(0, new_h2_level - self.h2_storage_capacity)  # 溢出部分
        self.h2_storage_level = min(new_h2_level, self.h2_storage_capacity)

        # ========== 步骤2: 热电联产 ==========
        # 剩余电力送往热电联产
        remaining_power = max(0, total_green_power - electrolyzer_power)
        chp_power = min(remaining_power, self.chp_capacity)
        chp_heat = chp_power * self.chp_heat_ratio  # kW 热

        # ========== 步骤3: 电量平衡 ==========
        grid_import = max(0, self.chp_capacity - total_green_power)  # 需要从电网购买
        grid_export = max(0, total_green_power - self.electrolyzer_capacity - self.chp_capacity)  # 向电网出售

        # ========== 步骤4: 经济计算 ==========
        # 假设时段（简化：白天峰时，夜间谷时）
        hour = timestamp.hour if timestamp else 12
        is_peak = 8 <= hour <= 20

        electricity_price = self.electricity_price_high if is_peak else self.electricity_price_low

        # 收益计算
        h2_revenue = h2_produced * 30  # 绿氢售价约 30 元/kg
        grid_revenue = grid_export * electricity_price  # 向电网卖电
        grid_cost = grid_import * electricity_price  # 从电网购电

        net_economic_benefit = h2_revenue + grid_revenue - grid_cost

        # ========== 步骤5: 碳减排计算 ==========
        # 假设火力发电碳排放强度 0.8 kg CO2/kWh
        carbon_savings = (electrolyzer_power + chp_power) * 0.8  # kg CO2

        # 记录结果
        result = {
            'timestamp': timestamp or datetime.now(),
            'wind_power': wind_power,
            'solar_power': solar_power,
            'total_green_power': total_green_power,
            'electrolyzer_power': electrolyzer_power,
            'h2_produced': h2_produced,
            'h2_storage_level': self.h2_storage_level,
            'chp_power': chp_power,
            'chp_heat': chp_heat,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'electricity_price': electricity_price,
            'is_peak': is_peak,
            'h2_revenue': h2_revenue,
            'net_economic_benefit': net_economic_benefit,
            'carbon_savings': carbon_savings
        }

        self.history.append(result)
        return result

    def run_simulation(self, power_forecast, timestamps=None):
        """
        运行批量调度仿真

        参数:
            power_forecast: DataFrame 或 dict，包含 wind_power 和 solar_power 列
            timestamps: 时间戳列表

        返回:
            list: 调度结果列表
        """
        results = []

        if isinstance(power_forecast, pd.DataFrame):
            wind_series = power_forecast['wind_power'] if 'wind_power' in power_forecast.columns else power_forecast.iloc[:, 0]
            solar_series = power_forecast['solar_power'] if 'solar_power' in power_forecast.columns else power_forecast.iloc[:, 1]

            for i in range(len(wind_series)):
                ts = timestamps[i] if timestamps else None
                result = self.dispatch(wind_series.iloc[i], solar_series.iloc[i], ts)
                results.append(result)
        else:
            for i, (w, s) in enumerate(zip(power_forecast['wind'], power_forecast['solar'])):
                ts = timestamps[i] if timestamps else None
                result = self.dispatch(w, s, ts)
                results.append(result)

        return results

    def summarize_results(self, results):
        """总结调度结果"""
        if not results:
            return "无调度结果"

        df = pd.DataFrame(results)

        summary = f"""
## 调度结果摘要

### 总体统计
| 指标 | 值 |
|------|-----|
| 总调度时段 | {len(results)} 小时 |
| 风电总量 | {df['wind_power'].sum():.2f} kWh |
| 光伏总量 | {df['solar_power'].sum():.2f} kWh |
| 绿电总量 | {df['total_green_power'].sum():.2f} kWh |

### 制氢统计
| 指标 | 值 |
|------|-----|
| 电解槽利用率 | {(df['electrolyzer_power'].sum() / (len(results) * self.electrolyzer_capacity) * 100):.2f}% |
| 绿氢总产量 | {df['h2_produced'].sum():.2f} kg |
| 平均储氢水平 | {df['h2_storage_level'].mean():.2f} kg |

### 热电联产统计
| 指标 | 值 |
|------|-----|
| CHP 利用率 | {(df['chp_power'].sum() / (len(results) * self.chp_capacity) * 100):.2f}% |
| 产热总量 | {df['chp_heat'].sum():.2f} kWh |

### 经济性分析
| 指标 | 值 |
|------|-----|
| 绿氢收益 | {df['h2_revenue'].sum():.2f} 元 |
| 净经济效益 | {df['net_economic_benefit'].sum():.2f} 元 |
| 峰时调度次数 | {df['is_peak'].sum()} 次 |
| 谷时调度次数 | {len(results) - df['is_peak'].sum()} 次 |

### 环保性分析
| 指标 | 值 |
|------|-----|
| 碳减排总量 | {df['carbon_savings'].sum():.2f} kg CO2 |
| 平均每时段碳减排 | {df['carbon_savings'].mean():.2f} kg CO2 |

### 能效评估
- 绿电自用率: {(df['electrolyzer_power'].sum() + df['chp_power'].sum()) / df['total_green_power'].sum() * 100:.2f}%
- 弃风弃光率: {(df['grid_export'].sum() / df['total_green_power'].sum() * 100):.2f}%
"""
        return summary

    def get_statistics(self, results):
        """获取统计指标"""
        if not results:
            return {}

        df = pd.DataFrame(results)

        stats = {
            'total_hours': len(results),
            'wind_total': df['wind_power'].sum(),
            'solar_total': df['solar_power'].sum(),
            'green_power_total': df['total_green_power'].sum(),
            'h2_production': df['h2_produced'].sum(),
            'chp_heat_total': df['chp_heat'].sum(),
            'economic_benefit': df['net_economic_benefit'].sum(),
            'carbon_savings': df['carbon_savings'].sum(),
            'curtailment_rate': df['grid_export'].sum() / df['total_green_power'].sum() * 100
        }

        return stats


if __name__ == '__main__':
    # 测试调度引擎
    config = {
        'electrolyzer_capacity': 500,
        'electrolyzer_efficiency': 0.7,
        'chp_capacity': 300,
        'chp_heat_ratio': 0.4,
        'h2_storage_capacity': 1000,
        'h2_storage_level': 500
    }

    dispatcher = DispatchEngine(config)

    # 模拟数据
    np.random.seed(42)
    n = 24  # 一天
    wind = np.random.uniform(100, 400, n)
    solar = np.abs(np.sin(np.linspace(0, np.pi, n)) * 300) + 50

    results = []
    for i in range(n):
        from datetime import datetime
        ts = datetime(2024, 1, 1, i)
        result = dispatcher.dispatch(wind[i], solar[i], ts)
        results.append(result)

    summary = dispatcher.summarize_results(results)
    print(summary)