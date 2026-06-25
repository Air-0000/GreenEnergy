"""
系统评估器模块
评估多能互补系统的整体性能
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
class SystemEvaluator:
    """系统性能评估器"""
    def __init__(self, output_dir='results',
                 electrolyzer_rated=500,
                 chp_rated=300,
                 h2_storage_rated=1000):
        self.output_dir = output_dir
        self.electrolyzer_rated = electrolyzer_rated
        self.chp_rated = chp_rated
        self.h2_storage_rated = h2_storage_rated
        os.makedirs(output_dir, exist_ok=True)
    def evaluate(self, dispatch_results, wind_forecast, solar_forecast):
        """
        评估系统整体性能（并网场景：多余电量优先上网，超出部分弃电）
        """
        if not dispatch_results:
            return self._default_metrics()
        df = pd.DataFrame(dispatch_results)
        # ========== 1. 能效与电量平衡指标 ==========
        green_total = df['total_green_power'].sum()
        # 绿电自用：仅制氢消耗的绿电部分（green_for_h2）
        # 注意：electrolyzer_power 包含电网补电，不是纯绿电
        if 'green_for_h2' in df.columns:
            green_used = df['green_for_h2'].sum()
        else:
            green_used = df['electrolyzer_power'].sum()
        # 电解槽总耗电 = 绿电制氢 + 电网补电 = 系统总输入电能
        electrolyzer_total_power = df['electrolyzer_power'].sum()
        # 上网售电量
        grid_export_sum = df['grid_export'].sum() if 'grid_export' in df.columns else 0
        # 弃电量 = 总绿电 - 自用 - 上网（并网场景下，无法消纳且无法上网的电量）
        curtailment_sum = df['curtailment'].sum() if 'curtailment' in df.columns else max(green_total - green_used - grid_export_sum, 0)
        # 电网购电量
        grid_import_sum = df['grid_import'].sum() if 'grid_import' in df.columns else 0
        # 比率计算
        self_use_rate = green_used / green_total if green_total > 0 else 0
        export_rate = grid_export_sum / green_total if green_total > 0 else 0
        curtailment_rate = curtailment_sum / green_total if green_total > 0 else 0
        # 系统综合能效 = 有效能量产出 / 电解槽实际消耗总电能
        # 有效能量产出 = 氢气化学能 + 回收余热
        # 电解槽总耗电 = 绿电 + 网电
        h2_energy = df['h2_produced'].sum() * 33.3  # 氢气低位发热量 33.3 kWh/kg
        chp_heat_energy = df['chp_heat'].sum()
        total_output = h2_energy + chp_heat_energy
        total_input = electrolyzer_total_power  # 电解槽总耗电就是系统总输入
        overall_efficiency = total_output / total_input if total_input > 0 else 0
        # ========== 2. 经济指标（全成本核算） ==========
        # 主营收益
        h2_revenue = df['h2_revenue'].sum() if 'h2_revenue' in df.columns else df['h2_produced'].sum() * 30
        heat_price = 0.3  # 供热价格 元/kWh
        heat_revenue = chp_heat_energy * heat_price
        # 电网侧收支
        elec_price = df['electricity_price'].iloc[0] if 'electricity_price' in df.columns else 0.5
        grid_import_cost = grid_import_sum * elec_price
        grid_export_revenue = grid_export_sum * elec_price
        # 运行成本：运维、水耗、耗材，按氢收益10%估算
        operation_cost = h2_revenue * 0.1
        # 设备折旧：按产氢量分摊，单位折旧0.02元/kg
        depreciation_cost = df['h2_produced'].sum() * 0.02
        # 全口径收支
        total_revenue = h2_revenue + heat_revenue + grid_export_revenue
        total_cost = grid_import_cost + operation_cost + depreciation_cost
        net_benefit = total_revenue - total_cost
        # ========== 3. 环保指标 ==========
        # 碳减排：绿氢替代灰氢 + 余热替代燃煤供热
        # 注意：仅绿电部分产生减排，网电部分不计入
        h2_carbon_factor = 10  # kgCO2/kg H2（灰氢排放因子）
        heat_carbon_factor = 0.11  # kgCO2/kWh（燃煤供热排放因子）
        if 'green_for_h2' in df.columns:
            green_h2_total = df['green_for_h2'].sum() / 50  # 绿电生产的氢气
            green_heat_total = df['green_for_h2'].sum() * 0.3 * 0.75  # 绿电对应的余热
        else:
            green_h2_total = df['h2_produced'].sum()
            green_heat_total = chp_heat_energy
        h2_carbon_saving = green_h2_total * h2_carbon_factor
        heat_carbon_saving = green_heat_total * heat_carbon_factor
        carbon_savings = h2_carbon_saving + heat_carbon_saving
        # ========== 4. 设备利用率 ==========
        electrolyzer_util = df['electrolyzer_power'].mean() / self.electrolyzer_rated * 100
        chp_util = df['chp_power'].mean() / self.chp_rated * 100 if 'chp_power' in df.columns and self.chp_rated > 0 else 0
        h2_storage_mean = df['h2_storage_level'].mean() if 'h2_storage_level' in df.columns else 0
        h2_utilization = h2_storage_mean / self.h2_storage_rated * 100
        # ========== 5. 封装结果 ==========
        metrics = {
            'overall_efficiency': overall_efficiency * 100,
            'self_use_rate': self_use_rate * 100,
            'export_rate': export_rate * 100,
            'curtailment_rate': curtailment_rate * 100,
            'h2_production': df['h2_produced'].sum(),
            'chp_heat_total': chp_heat_energy,
            'green_power_total': green_total,
            'green_power_used': green_used,
            'grid_import_total': grid_import_sum,
            'grid_export_total': grid_export_sum,
            'total_revenue': total_revenue,
            'h2_revenue': h2_revenue,
            'heat_revenue': heat_revenue,
            'grid_export_revenue': grid_export_revenue,
            'total_cost': total_cost,
            'grid_import_cost': grid_import_cost,
            'operation_cost': operation_cost,
            'depreciation_cost': depreciation_cost,
            'net_economic_benefit': net_benefit,
            'carbon_savings': carbon_savings,
            'h2_utilization': h2_utilization,
            'electrolyzer_utilization': electrolyzer_util,
            'chp_utilization': chp_util
        }
        self._print_metrics(metrics)
        self._generate_plots(df, wind_forecast, solar_forecast)
        return metrics
    def _print_metrics(self, metrics):
        """打印评估指标"""
        print("\n" + "=" * 50)
        print("系统性能评估报告")
        print("=" * 50)
        print(f"【能效指标】")
        print(f"  系统综合能效: {metrics['overall_efficiency']:.2f}%")
        print(f"  绿电自用率: {metrics['self_use_rate']:.2f}%")
        print(f"  绿电上网率: {metrics['export_rate']:.2f}%")
        print(f"  弃风弃光率: {metrics['curtailment_rate']:.2f}%")
        print(f"\n【生产指标】")
        print(f"  绿氢总产量: {metrics['h2_production']:.2f} kg")
        print(f"  热电联产产热: {metrics['chp_heat_total']:.2f} kWh")
        print(f"  电解槽利用率: {metrics['electrolyzer_utilization']:.2f}%")
        print(f"\n【电量平衡】")
        print(f"  总绿电量: {metrics['green_power_total']:.2f} kWh")
        print(f"  绿电自用: {metrics['green_power_used']:.2f} kWh")
        print(f"  电网购电: {metrics['grid_import_total']:.2f} kWh")
        print(f"  上网售电: {metrics['grid_export_total']:.2f} kWh")
        print(f"\n【经济指标】")
        print(f"  总收益: {metrics['total_revenue']:.2f} 元")
        print(f"    绿氢收益: {metrics['h2_revenue']:.2f} 元")
        print(f"    供热收益: {metrics['heat_revenue']:.2f} 元")
        print(f"    上网售电收益: {metrics['grid_export_revenue']:.2f} 元")
        print(f"  总成本: {metrics['total_cost']:.2f} 元")
        print(f"    购电成本: {metrics['grid_import_cost']:.2f} 元")
        print(f"    运行运维成本: {metrics['operation_cost']:.2f} 元")
        print(f"    设备折旧成本: {metrics['depreciation_cost']:.2f} 元")
        print(f"  净经济效益: {metrics['net_economic_benefit']:.2f} 元")
        print(f"\n【环保指标】")
        print(f"  碳减排量: {metrics['carbon_savings']:.2f} kg CO2")
        print("=" * 50)
    def _generate_plots(self, df, wind_forecast, solar_forecast):
        """生成可视化图表"""
        wind_arr = np.array(wind_forecast)
        solar_arr = np.array(solar_forecast)
        n = min(len(wind_arr), len(solar_arr), len(df))
        # 图1: 功率调度时序图
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        # 子图1: 风光预测
        ax1 = axes[0]
        hours = range(n)
        ax1.plot(hours, wind_arr[:n], label='Wind', alpha=0.8)
        ax1.plot(hours, solar_arr[:n], label='Solar', alpha=0.8)
        ax1.plot(hours, wind_arr[:n] + solar_arr[:n], label='Total Green', linewidth=2)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Power (kW)')
        ax1.set_title('Wind & Solar Power Forecast')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 子图2: 调度功率分配
        ax2 = axes[1]
        electrolyzer = df['electrolyzer_power'].values[:n] if 'electrolyzer_power' in df.columns else np.zeros(n)
        chp = df['chp_power'].values[:n] if 'chp_power' in df.columns else np.zeros(n)
        grid_exp = df['grid_export'].values[:n] if 'grid_export' in df.columns else np.zeros(n)
        ax2.stackplot(hours, electrolyzer, chp, grid_exp,
                      labels=['Electrolyzer', 'CHP', 'Grid Export'],
                      alpha=0.8)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Power (kW)')
        ax2.set_title('Power Dispatch Allocation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # 子图3: 储氢水平变化
        ax3 = axes[2]
        h2_storage = df['h2_storage_level'].values[:n] if 'h2_storage_level' in df.columns else np.zeros(n)
        ax3.fill_between(hours, 0, h2_storage, alpha=0.5, label='H2 Storage')
        ax3.axhline(y=self.h2_storage_rated, color='r', linestyle='--', label='Max Capacity')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('H2 (kg)')
        ax3.set_title('Hydrogen Storage Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dispatch_overview.png'), dpi=150)
        print(f"dispatch_overview.png saved")
        # 图2: 经济与碳排放
        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
        # 经济收益
        ax_econ = axes2[0]
        h2_rev = df['h2_revenue'].values[:n] if 'h2_revenue' in df.columns else np.zeros(n)
        net_ben = df['net_economic_benefit'].values[:n] if 'net_economic_benefit' in df.columns else np.zeros(n)
        ax_econ.bar(hours, h2_rev, label='H2 Revenue', alpha=0.7)
        ax_econ.bar(hours, np.maximum(net_ben - h2_rev, 0), bottom=h2_rev, label='Other Benefit', alpha=0.7)
        ax_econ.set_xlabel('Time (hours)')
        ax_econ.set_ylabel('Revenue (Yuan)')
        ax_econ.set_title('Economic Benefit per Hour')
        ax_econ.legend()
        ax_econ.grid(True, alpha=0.3)
        # 碳减排
        ax_carbon = axes2[1]
        carbon = df['carbon_savings'].values[:n] if 'carbon_savings' in df.columns else np.zeros(n)
        ax_carbon.fill_between(hours, 0, carbon, alpha=0.5, color='green')
        ax_carbon.set_xlabel('Time (hours)')
        ax_carbon.set_ylabel('Carbon Savings (kg CO2)')
        ax_carbon.set_title('Carbon Emission Reduction')
        ax_carbon.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'economics_carbon.png'), dpi=150)
        print(f"economics_carbon.png saved")
        plt.close('all')
    def _default_metrics(self):
        """默认指标（当无数据时）"""
        return {
            'overall_efficiency': 0,
            'self_use_rate': 0,
            'export_rate': 0,
            'curtailment_rate': 0,
            'h2_production': 0,
            'chp_heat_total': 0,
            'green_power_total': 0,
            'green_power_used': 0,
            'grid_import_total': 0,
            'grid_export_total': 0,
            'total_revenue': 0,
            'h2_revenue': 0,
            'heat_revenue': 0,
            'grid_export_revenue': 0,
            'total_cost': 0,
            'grid_import_cost': 0,
            'operation_cost': 0,
            'depreciation_cost': 0,
            'net_economic_benefit': 0,
            'carbon_savings': 0,
            'h2_utilization': 0,
            'electrolyzer_utilization': 0,
            'chp_utilization': 0
        }
    def compare_scenarios(self, scenario1_results, scenario2_results, scenario_names=None):
        """对比两个场景的性能"""
        metrics1 = self.evaluate(scenario1_results, [], [])
        metrics2 = self.evaluate(scenario2_results, [], [])
        if scenario_names is None:
            scenario_names = ['Scenario 1', 'Scenario 2']
        comparison = pd.DataFrame({
            'Metric': list(metrics1.keys()),
            scenario_names[0]: list(metrics1.values()),
            scenario_names[1]: list(metrics2.values())
        })
        comparison['Difference'] = comparison[scenario_names[1]] - comparison[scenario_names[0]]
        comparison['% Change'] = (comparison['Difference'] / comparison[scenario_names[0]] * 100).round(2)
        print("\n场景对比分析:")
        print(comparison.to_string(index=False))
        return comparison
    def generate_report(self, metrics, dispatch_results):
        """生成评估报告"""
        report = f"""
# 系统性能评估报告
## 1. 系统场景说明
本系统为**并网型风光制氢热电联产系统**，电网作为备用调节电源：
- 风光出力充足时，优先使用绿电制氢，富余电量上网售电；
- 风光出力不足时，从电网购电补充，保证电解槽连续稳定运行；
- 电解槽运行余热全部回收用于供热，实现能量梯级利用。
## 2. 能效指标
| 指标 | 值 | 说明 |
|------|-----|------|
| 系统综合能效 | {metrics['overall_efficiency']:.2f}% | 氢能+热能 / 电解槽消耗总电能 |
| 绿电自用率 | {metrics['self_use_rate']:.2f}% | 制氢消纳绿电占总绿电比例 |
| 绿电上网率 | {metrics['export_rate']:.2f}% | 富余绿电上网售电比例 |
| 弃风弃光率 | {metrics['curtailment_rate']:.2f}% | 无法消纳且无法上网的废弃电量比例 |
## 3. 生产指标
| 指标 | 值 |
|------|-----|
| 绿氢总产量 | {metrics['h2_production']:.2f} kg |
| 热电联产产热 | {metrics['chp_heat_total']:.2f} kWh |
| 电解槽利用率 | {metrics['electrolyzer_utilization']:.2f}% |
| 热电联产利用率 | {metrics['chp_utilization']:.2f}% |
## 4. 经济指标
| 指标 | 值 |
|------|-----|
| 总收益 | {metrics['total_revenue']:.2f} 元 |
| 绿氢收益 | {metrics['h2_revenue']:.2f} 元 |
| 供热收益 | {metrics['heat_revenue']:.2f} 元 |
| 上网售电收益 | {metrics['grid_export_revenue']:.2f} 元 |
| 总成本 | {metrics['total_cost']:.2f} 元 |
| 购电成本 | {metrics['grid_import_cost']:.2f} 元 |
| 运行运维成本 | {metrics['operation_cost']:.2f} 元 |
| 设备折旧成本 | {metrics['depreciation_cost']:.2f} 元 |
| 净经济效益 | {metrics['net_economic_benefit']:.2f} 元 |
## 5. 环保指标
| 指标 | 值 | 说明 |
|------|-----|------|
| 碳减排量 | {metrics['carbon_savings']:.2f} kg CO2 | 绿氢替代灰氢+余热替代燃煤供热 |
## 6. 结论
该多能互补系统通过"绿电制氢+热电联产"的协同调度，实现了：
- **{metrics['self_use_rate']:.1f}%** 的绿电自用率
- 系统综合能效达 **{metrics['overall_efficiency']:.1f}%**
- 累计碳减排 **{metrics['carbon_savings']:.1f} kg CO2**
## 7. 改进建议
1. 扩容电解槽容量，提升绿电消纳比例，降低弃电率
2. 引入预测不确定性处理，优化调度鲁棒性
3. 升级为强化学习调度，自适应优化多目标策略
4. 拓展供热用户，提升余热回收的经济价值
"""
        with open(os.path.join(self.output_dir, 'evaluation_report.md'), 'w') as f:
            f.write(report)
        print(f"✓ 评估报告已保存至 {self.output_dir}/evaluation_report.md")
        return report
if __name__ == '__main__':
    # 测试评估器
    evaluator = SystemEvaluator()
    # 模拟数据
    np.random.seed(42)
    n = 168  # 一周
    dispatch_results = []
    for i in range(n):
        result = {
            'electrolyzer_power': np.random.uniform(300, 500),
            'green_for_h2': np.random.uniform(200, 500),
            'chp_power': 0,
            'total_green_power': np.random.uniform(400, 800),
            'h2_produced': np.random.uniform(6, 10),
            'chp_heat': np.random.uniform(50, 150),
            'grid_import': np.random.uniform(0, 200),
            'grid_export': np.random.uniform(0, 300),
            'curtailment': 0,
            'h2_storage_level': np.random.uniform(200, 800),
            'h2_revenue': np.random.uniform(180, 300),
            'net_economic_benefit': np.random.uniform(100, 250),
            'carbon_savings': np.random.uniform(10, 30),
            'electricity_price': 0.5
        }
        dispatch_results.append(result)
    metrics = evaluator.evaluate(dispatch_results, [], [])
    print(f"测试完成，系统能效: {metrics['overall_efficiency']:.2f}%")
