"""
大模型策略分析师模块
使用 GPT/Claude API 解读调度策略
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')


class LLMAnalyst:
    """大模型策略分析师"""

    def __init__(self, api_key=None, model='gpt-4o-mini'):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model

        if not self.api_key:
            print("警告: 未设置 API key，大模型分析功能不可用")
            self.available = False
        else:
            self.available = True
            print("✓ 大模型分析师已初始化")

    def analyze_scenario(self, scenario_summary, analysis_type='dispatch_strategy'):
        """
        分析调度场景

        参数:
            scenario_summary: 调度结果摘要（字符串）
            analysis_type: 分析类型
                - 'dispatch_strategy': 调度策略分析
                - 'efficiency_evaluation': 能效评估
                - 'optimization_suggestions': 优化建议

        返回:
            str: 分析结果
        """
        if not self.available:
            return self._fallback_analysis(scenario_summary)

        system_prompt = """你是一位能源系统优化专家，擅长分析绿电+热电联产+加氢能源多能互补系统的调度策略。
请从以下维度进行分析：
1. 调度逻辑是否合理
2. 能效提升空间
3. 潜在风险与改进建议
4. 与强化学习/多目标优化的结合方向

请用简洁专业的语言输出分析结果，突出关键发现和建议。"""

        user_prompt = f"""## 调度场景摘要

{scenario_summary}

请根据上述数据，分析该多能互补系统的调度策略，给出：
1. 调度策略评价（2-3句话）
2. 能效改进建议（3-5条）
3. 升级为智能调度的方向（强化学习/多目标优化）
"""

        try:
            # 尝试使用 Claude API
            response = self._call_claude(user_prompt, system_prompt)
            return response
        except Exception as e:
            # 降级为本地分析
            print(f"API 调用失败: {e}，使用本地分析")
            return self._fallback_analysis(scenario_summary)

    def _call_claude(self, user_prompt, system_prompt):
        """调用 Claude API"""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            return response.content[0].text
        except ImportError:
            # 如果没有 anthropic 包，尝试 OpenAI
            return self._call_openai(user_prompt, system_prompt)

    def _call_openai(self, user_prompt, system_prompt):
        """调用 OpenAI API"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"API 调用失败: {response.status_code}")

    def _fallback_analysis(self, scenario_summary):
        """本地降级分析（当 API 不可用时）"""
        return """
## 调度策略本地分析

### 策略评价
本系统采用"优先制氢、余电热电联产"的协同调度策略，逻辑清晰、简单有效。
通过绿电直接制氢，可最大化清洁能源利用率；热电联产作为调峰手段，
实现了热-电-氢的多能互补。峰谷电价联动机制有助于优化经济效益。

### 能效改进建议
1. **动态调节电解槽功率**: 根据储氢罐液位动态调整制氢功率，避免溢出损失
2. **引入余热回收**: 热电联产余热可用于建筑供暖或海水淡化，提升系统综合能效
3. **预测不确定性处理**: 当前规则调度未考虑预测误差，可引入随机规划或鲁棒优化
4. **多目标优化**: 同时考虑经济效益、碳减排、能效等多个目标

### 升级方向
1. **短期（课程论文）**: 升级为强化学习调度（DQN/PPO），学习最优调度策略
2. **中期（毕设）**: 引入 NSGA-II 多目标优化，处理风光波动和电价随机
3. **长期（发表论文）**: 建立热-电-氢多能流耦合模型，进行灵敏度分析

### 技术对比
| 方法 | 优点 | 缺点 |
|------|------|------|
| 规则调度 | 简单、可解释 | 无法处理复杂场景 |
| 强化学习 | 自适应、泛化强 | 需要大量训练数据 |
| 多目标优化 | 帕累托最优 | 计算复杂度高 |
"""

    def generate_dispatch_report(self, dispatch_results, wind_forecast, solar_forecast):
        """生成完整的调度分析报告"""
        # 构建场景摘要
        import pandas as pd

        if dispatch_results:
            df = pd.DataFrame(dispatch_results)

            scenario = f"""
## 调度场景数据

### 输入数据
- 风电预测: {len(wind_forecast)} 个点，平均 {sum(wind_forecast)/len(wind_forecast):.2f} kW
- 光伏预测: {len(solar_forecast)} 个点，平均 {sum(solar_forecast)/len(solar_forecast):.2f} kW

### 关键指标
- 绿氢总产量: {df['h2_produced'].sum():.2f} kg
- 碳减排量: {df['carbon_savings'].sum():.2f} kg CO2
- 净经济效益: {df['net_economic_benefit'].sum():.2f} 元
- 电解槽平均利用率: {df['electrolyzer_power'].mean() / 500 * 100:.2f}%
- 热电联产平均利用率: {df['chp_power'].mean() / 300 * 100:.2f}%
"""
        else:
            scenario = "## 无调度结果数据"

        # 调用分析
        return self.analyze_scenario(scenario)

    def explain_dispatch_rule(self, wind_power, solar_power, dispatch_result):
        """解释单个调度决策"""
        prompt = f"""## 调度决策

输入条件：
- 风电功率: {wind_power:.2f} kW
- 光伏功率: {solar_power:.2f} kW
- 总绿电: {wind_power + solar_power:.2f} kW

调度结果：
- 电解槽功率: {dispatch_result['electrolyzer_power']:.2f} kW
- 制氢量: {dispatch_result['h2_produced']:.2f} kg/h
- 热电联产: {dispatch_result['chp_power']:.2f} kW
- 产热量: {dispatch_result['chp_heat']:.2f} kW

请解释这个调度决策的逻辑和依据。"""

        return self.analyze_scenario(prompt, analysis_type='dispatch_explanation')


if __name__ == '__main__':
    # 测试大模型分析师
    analyst = LLMAnalyst()

    test_summary = """
### 调度统计
- 总时段: 168 小时
- 风电总量: 45632 kWh
- 光伏总量: 23456 kWh
- 绿氢产量: 892 kg
- 碳减排: 5542 kg CO2
    """

    result = analyst.analyze_scenario(test_summary)
    print(result)