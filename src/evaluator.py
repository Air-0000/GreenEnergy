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

    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, dispatch_results, wind_forecast, solar_forecast):
        """
        评估系统整体性能

        参数:
            dispatch_results: 调度结果列表
            wind_forecast: 风电预测值
            solar_forecast: 光伏预测值

        返回:
            dict: 评估指标
        """
        if not dispatch_results:
            return self._default_metrics()

        df = pd.DataFrame(dispatch_results)

        # ========== 1. 能效指标 ==========
        # 绿电自用率
        green_used = df['electrolyzer_power'].sum() + df['chp_power'].sum()
        green_total = df['total_green_power'].sum()
        self_use_rate = green_used / green_total if green_total > 0 else 0

        # 弃风弃光率
        curtailment = df['grid_export'].sum() / green_total if green_total > 0 else 0

        # 系统综合能效（制氢+热电联产效率加权）
        h2_energy = df['h2_produced'].sum() * 33.3  # kWh/kg H2 (下界)
        chp_energy = df['chp_power'].sum()  # kWh
        total_useful = h2_energy + chp_energy
        overall_efficiency = total_useful / green_total if green_total > 0 else 0

        # ========== 2. 经济指标 ==========
        h2_revenue = df['h2_revenue'].sum()
        net_benefit = df['net_economic_benefit'].sum()
        grid_import_cost = (df.get('grid_import', pd.Series([0]*len(df))) * df.get('electricity_price', pd.Series([0.5]*len(df)))).sum()
        grid_export_revenue = (df.get('grid_export', pd.Series([0]*len(df))) * df.get('electricity_price', pd.Series([0.5]*len(df)))).sum()

        # ========== 3. 环保指标 ==========
        carbon_savings = df['carbon_savings'].sum()

        # ========== 4. 系统稳定性 ==========
        h2_storage_std = df['h2_storage_level'].std()
        h2_utilization = df['h2_produced'].mean() / 500 * 100  # 相对于额定容量的利用率

        # ========== 5. 评估结果 ==========
        metrics = {
            'overall_efficiency': overall_efficiency * 100,  # 百分比
            'self_use_rate': self_use_rate * 100,
            'curtailment_rate': curtailment * 100,
            'h2_production': df['h2_produced'].sum(),
            'chp_heat_total': df['chp_heat'].sum(),
            'h2_revenue': h2_revenue,
            'net_economic_benefit': net_benefit,
            'carbon_savings': carbon_savings,
            'grid_import_cost': grid_import_cost,
            'grid_export_revenue': grid_export_revenue,
            'h2_utilization': h2_utilization,
            'electrolyzer_utilization': df['electrolyzer_power'].mean() / 500 * 100,
            'chp_utilization': df['chp_power'].mean() / 300 * 100
        }

        # 打印评估结果
        self._print_metrics(metrics)

        # 生成可视化
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
        print(f"  弃风弃光率: {metrics['curtailment_rate']:.2f}%")
        print(f"\n【生产指标】")
        print(f"  绿氢总产量: {metrics['h2_production']:.2f} kg")
        print(f"  热电联产产热: {metrics['chp_heat_total']:.2f} kWh")
        print(f"  电解槽利用率: {metrics['electrolyzer_utilization']:.2f}%")
        print(f"\n【经济指标】")
        print(f"  绿氢收益: {metrics['h2_revenue']:.2f} 元")
        print(f"  净经济效益: {metrics['net_economic_benefit']:.2f} 元")
        print(f"\n【环保指标】")
        print(f"  碳减排量: {metrics['carbon_savings']:.2f} kg CO2")
        print("=" * 50)

    def _generate_plots(self, df, wind_forecast, solar_forecast):
        """生成可视化图表"""
        # 确保数组格式正确
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
        ax3.axhline(y=1000, color='r', linestyle='--', label='Max Capacity')
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
        ax_econ.bar(hours, net_ben - h2_rev, bottom=h2_rev, label='Grid Benefit', alpha=0.7)
        ax_econ.set_xlabel('Time (hours)')
        ax_econ.set_ylabel('Revenue')
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
            'curtailment_rate': 0,
            'h2_production': 0,
            'chp_heat_total': 0,
            'h2_revenue': 0,
            'net_economic_benefit': 0,
            'carbon_savings': 0
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

## 1. 能效指标
| 指标 | 值 |
|------|-----|
| 系统综合能效 | {metrics['overall_efficiency']:.2f}% |
| 绿电自用率 | {metrics['self_use_rate']:.2f}% |
| 弃风弃光率 | {metrics['curtailment_rate']:.2f}% |

## 2. 生产指标
| 指标 | 值 |
|------|-----|
| 绿氢总产量 | {metrics['h2_production']:.2f} kg |
| 热电联产产热 | {metrics['chp_heat_total']:.2f} kWh |
| 电解槽利用率 | {metrics['electrolyzer_utilization']:.2f}% |
| 热电联产利用率 | {metrics['chp_utilization']:.2f}% |

## 3. 经济指标
| 指标 | 值 |
|------|-----|
| 绿氢收益 | {metrics['h2_revenue']:.2f} 元 |
| 净经济效益 | {metrics['net_economic_benefit']:.2f} 元 |
| 上网收益 | {metrics['grid_export_revenue']:.2f} 元 |
| 购电成本 | {metrics['grid_import_cost']:.2f} 元 |

## 4. 环保指标
| 指标 | 值 |
|------|-----|
| 碳减排量 | {metrics['carbon_savings']:.2f} kg CO2 |

## 5. 结论
该多能互补系统通过"绿电制氢+热电联产"的协同调度，实现了：
- **{metrics['self_use_rate']:.1f}%** 的绿电自用率
- 相比传统弃风弃光方案，减少了 **{metrics['curtailment_rate']:.1f}%** 的能源浪费
- 碳减排达 **{metrics['carbon_savings']:.1f} kg CO2**

## 6. 改进建议
1. 引入预测不确定性处理，降低弃风弃光率
2. 升级为强化学习调度，自适应优化调度策略
3. 引入多目标优化，平衡经济与环保目标
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
            'chp_power': np.random.uniform(100, 300),
            'total_green_power': np.random.uniform(400, 800),
            'h2_produced': np.random.uniform(5, 15),
            'chp_heat': np.random.uniform(40, 120),
            'h2_storage_level': np.random.uniform(400, 800),
            'grid_export': np.random.uniform(0, 100),
            'h2_revenue': np.random.uniform(150, 450),
            'net_economic_benefit': np.random.uniform(50, 300),
            'carbon_savings': np.random.uniform(320, 640)
        }
        dispatch_results.append(result)

    metrics = evaluator.evaluate(dispatch_results, np.random.randn(n), np.random.randn(n))
    print("\n评估完成！")