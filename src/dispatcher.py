"""
调度引擎模块
实现绿氢-余热回收协同调度规则
CHP定位：电解槽余热回收系统，不消耗额外电能
"""
import numpy as np
import pandas as pd
from datetime import datetime, time
class DispatchEngine:
    """协同调度引擎"""
    def __init__(self, config):
        self.config = config
        self.history = []
        # 电解槽参数
        self.electrolyzer_capacity = config.get('electrolyzer_capacity', 500)  # kW
        self.electrolyzer_efficiency = config.get('electrolyzer_efficiency', 50)  # kWh/kg H2
        # 余热回收参数（CHP为电解槽废热回收，不耗电）
        self.waste_heat_ratio = config.get('waste_heat_ratio', 0.30)  # 废热占输入电能比例
        self.recovery_efficiency = config.get('recovery_efficiency', 0.75)  # 余热回收效率
        self.chp_capacity = config.get('chp_capacity', 150)  # 余热回收系统额定热功率 kW
        # 储氢参数
        self.h2_storage_capacity = config.get('h2_storage_capacity', 1000)  # kg
        self.h2_storage_level = config.get('h2_storage_level', 500)  # 当前储量
        # 电价参数
        self.electricity_price_low = config.get('electricity_price_low', 0.3)
        self.electricity_price_high = config.get('electricity_price_high', 0.8)
    def dispatch(self, wind_power, solar_power, timestamp=None):
        """
        核心调度逻辑（并网场景：电网作为备用电源，保证电解槽满负荷运行）
        优先级：
            1. 优先使用绿电制氢
            2. 绿电不足时，从电网购电补充，保证电解槽连续稳定运行
            3. 绿电富余时，多余电量全部上网售电
            4. 电解槽运行余热全部回收用于供热
        参数:
            wind_power: 风电预测功率 (kW)
            solar_power: 光伏预测功率 (kW)
        返回:
            dict: 调度结果
        """
        total_green_power = wind_power + solar_power
        # ========== 步骤1: 电解制氢（优先绿电，不足补网电，满负荷运行） ==========
        # 绿电用于制氢的部分
        green_for_h2 = min(total_green_power, self.electrolyzer_capacity)
        # 电网补电：风光不足时，从电网购电，保证电解槽满负荷运行
        grid_import = max(0, self.electrolyzer_capacity - total_green_power)
        # 电解槽实际运行总功率 = 绿电 + 网电 = 额定功率（满负荷）
        electrolyzer_power = green_for_h2 + grid_import
        # 制氢量 = 电解槽总功率 / 制氢电耗
        h2_produced = electrolyzer_power / self.electrolyzer_efficiency  # kg/h
        # 更新储氢量
        new_h2_level = self.h2_storage_level + h2_produced
        self.h2_storage_level = min(new_h2_level, self.h2_storage_capacity)
        # ========== 步骤2: 余热回收（CHP不消耗额外电能） ==========
        # 回收电解槽废热，chp_power=0 表示不耗电
        chp_power = 0
        chp_heat_theoretical = electrolyzer_power * self.waste_heat_ratio * self.recovery_efficiency
        chp_heat = min(chp_heat_theoretical, self.chp_capacity)
        # ========== 步骤3: 电量平衡 ==========
        # 富余绿电 = 总绿电 - 制氢用绿电
        surplus_green = max(0, total_green_power - green_for_h2)
        # 上网电量 = 富余绿电（并网场景无上网限制，全部可上网）
        grid_export = surplus_green
        # 弃电量 = 0（并网场景全部富余电量均可上网）
        curtailment = 0
        # ========== 步骤4: 经济计算 ==========
        hour = timestamp.hour if timestamp else 12
        is_peak = 8 <= hour <= 20
        electricity_price = self.electricity_price_high if is_peak else self.electricity_price_low
        h2_revenue = h2_produced * 30  # 绿氢售价约 30 元/kg
        grid_export_revenue = grid_export * electricity_price  # 上网售电收益
        grid_import_cost = grid_import * electricity_price  # 购电成本
        net_economic_benefit = h2_revenue + grid_export_revenue - grid_import_cost
        # ========== 步骤5: 碳减排计算 ==========
        # 仅绿电制氢和绿电对应的余热产生减排，网电部分不计入
        green_h2 = green_for_h2 / self.electrolyzer_efficiency  # 绿电生产的氢气
        green_heat = green_for_h2 * self.waste_heat_ratio * self.recovery_efficiency  # 绿电对应的余热
        # 碳减排因子：氢 10 kgCO2/kg（替代灰氢），热 0.11 kgCO2/kWh（替代燃煤供热）
        carbon_savings = green_h2 * 10 + green_heat * 0.11
        # 记录结果
        result = {
            'timestamp': timestamp or datetime.now(),
            'wind_power': wind_power,
            'solar_power': solar_power,
            'total_green_power': total_green_power,
            'electrolyzer_power': electrolyzer_power,
            'green_for_h2': green_for_h2,
            'h2_produced': h2_produced,
            'h2_storage_level': self.h2_storage_level,
            'chp_power': chp_power,
            'chp_heat': chp_heat,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'curtailment': curtailment,
            'electricity_price': electricity_price,
            'is_peak': is_peak,
            'h2_revenue': h2_revenue,
            'grid_export_revenue': grid_export_revenue,
            'grid_import_cost': grid_import_cost,
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
### 余热回收统计
| 指标 | 值 |
|------|-----|
| 回收热量总量 | {df['chp_heat'].sum():.2f} kWh |
| 余热回收系统利用率 | {(df['chp_heat'].sum() / (len(results) * self.chp_capacity) * 100):.2f}% |
### 电网交互统计
| 指标 | 值 |
|------|-----|
| 购电总量 | {df['grid_import'].sum():.2f} kWh |
| 售电总量 | {df['grid_export'].sum():.2f} kWh |
| 净购电量 | {df['grid_import'].sum() - df['grid_export'].sum():.2f} kWh |
### 经济性分析
| 指标 | 值 |
|------|-----|
| 绿氢收益 | {df['h2_revenue'].sum():.2f} 元 |
| 上网售电收益 | {df['grid_export_revenue'].sum():.2f} 元 |
| 购电成本 | {df['grid_import_cost'].sum():.2f} 元 |
| 毛经济效益 | {df['net_economic_benefit'].sum():.2f} 元 |
| 峰时调度次数 | {df['is_peak'].sum()} 次 |
| 谷时调度次数 | {len(results) - df['is_peak'].sum()} 次 |
### 环保性分析
| 指标 | 值 |
|------|-----|
| 碳减排总量 | {df['carbon_savings'].sum():.2f} kg CO2 |
| 平均每时段碳减排 | {df['carbon_savings'].mean():.2f} kg CO2 |
### 能效评估
- 绿电自用率: {(df['green_for_h2'].sum() / df['total_green_power'].sum() * 100):.2f}%
- 绿电上网率: {(df['grid_export'].sum() / df['total_green_power'].sum() * 100):.2f}%
- 弃风弃光率: {(df['curtailment'].sum() / df['total_green_power'].sum() * 100):.2f}%
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
            'grid_import_total': df['grid_import'].sum(),
            'grid_export_total': df['grid_export'].sum(),
            'economic_benefit': df['net_economic_benefit'].sum(),
            'carbon_savings': df['carbon_savings'].sum(),
            'self_use_rate': df['green_for_h2'].sum() / df['total_green_power'].sum() * 100 if df['total_green_power'].sum() > 0 else 0,
            'export_rate': df['grid_export'].sum() / df['total_green_power'].sum() * 100 if df['total_green_power'].sum() > 0 else 0,
            'curtailment_rate': df['curtailment'].sum() / df['total_green_power'].sum() * 100 if df['total_green_power'].sum() > 0 else 0
        }
        return stats
if __name__ == '__main__':
    # 测试调度引擎
    config = {
        'electrolyzer_capacity': 500,
        'electrolyzer_efficiency': 50,
        'waste_heat_ratio': 0.30,
        'recovery_efficiency': 0.75,
        'chp_capacity': 150,
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
