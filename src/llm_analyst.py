"""
大模型分析师模块
调用大模型对调度策略进行解读和优化建议
"""
import os
import json
from datetime import datetime
class LLMAnalyst:
    """大模型策略分析师"""
    def __init__(self, api_key=None, model='gpt-4', base_url=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.base_url = base_url
        self.history = []
        if not self.api_key:
            print("警告: 未提供 API key，大模型功能将不可用")
    def analyze_scenario(self, scenario_data):
        """
        分析调度场景，给出策略解读和优化建议
        参数:
            scenario_data: 调度场景摘要或数据
        返回:
            str: 大模型分析结果
        """
        if not self.api_key:
            return "（API key 未配置，跳过大模型分析）"
        prompt = self._build_analysis_prompt(scenario_data)
        try:
            response = self._call_llm(prompt)
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'response': response
            })
            return response
        except Exception as e:
            print(f"大模型调用失败: {e}")
            return f"（大模型调用失败: {str(e)}）"
    def _build_analysis_prompt(self, scenario_data):
        """构建分析提示词"""
        prompt = f"""
你是一位资深的能源系统调度专家，擅长分析风光制氢热电联产多能互补系统的运行策略。
请分析以下调度场景数据，并给出专业的策略解读和优化建议：
{scenario_data}
请从以下几个维度进行分析：
## 1. 调度策略解读
- 当前调度规则的合理性分析
- 制氢和热电联产的协同效果评估
- 储能（储氢）的利用效率
## 2. 经济性分析
- 成本结构分析
- 收益来源评估
- 峰谷电价套利效果
## 3. 能效与环保评估
- 系统综合能效分析
- 绿电消纳比例评估
- 碳减排效果分析
## 4. 优化建议
- 短期优化（参数调整）
- 中期优化（策略升级）
- 长期优化（系统扩容）
## 5. 风险提示
- 预测误差对调度的影响
- 设备安全运行风险
- 市场价格波动风险
请用专业但易懂的语言回答，控制在 800 字以内。
"""
        return prompt
    def _call_llm(self, prompt):
        """调用大模型 API"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位资深的能源系统调度专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except ImportError:
            # 如果没有 openai 库，尝试使用其他方式
            return self._fallback_analysis(prompt)
    def _fallback_analysis(self, prompt):
        """降级分析（当 API 不可用时）"""
        return """
## 调度策略分析（离线模式）
### 1. 调度策略解读
当前系统采用"优先制氢、余热回收、富余上网"的规则调度策略，逻辑清晰且符合工程实际。
制氢与热电联产的协同效果良好，通过余热回收提升了系统综合能效。
### 2. 经济性分析
- 主要收益来源：绿氢销售（占比最高）
- 主要成本：购电成本（风光不足时从电网购电）
- 建议：优化峰谷电价调度策略，谷时多制氢、峰时少制氢
### 3. 能效与环保评估
- 系统综合能效处于合理范围
- 绿电消纳比例有待提升
- 碳减排效果显著，具有良好的环保效益
### 4. 优化建议
**短期**：调整电解槽运行策略，提升绿电自用比例
**中期**：引入强化学习调度算法，自适应优化多目标
**长期**：扩容电解槽和储氢容量，提升系统规模效应
### 5. 风险提示
- 风光预测误差可能导致调度偏差，建议引入不确定性处理
- 电解槽频繁启停会影响设备寿命，需优化启停策略
- 电价和氢价波动会影响经济效益，建议做敏感性分析
（注：此为离线模式的模板分析，配置 API key 后可获得更精准的个性化分析）
"""
    def generate_report(self, metrics, dispatch_summary):
        """生成完整的分析报告"""
        prompt = f"""
请基于以下系统运行数据，生成一份专业的多能互补系统分析报告：
【系统指标】
{json.dumps(metrics, indent=2, ensure_ascii=False)}
【调度摘要】
{dispatch_summary}
报告要求：
1. 结构清晰，分章节论述
2. 数据驱动，有理有据
3. 给出具体可行的优化建议
4. 字数控制在 1500 字左右
"""
        return self.analyze_scenario(prompt)
    def compare_scenarios(self, scenario1, scenario2, names=None):
        """对比两个场景的优劣"""
        if names is None:
            names = ['场景A', '场景B']
        prompt = f"""
请对比分析以下两个调度场景的优劣：
【{names[0]}】
{scenario1}
【{names[1]}】
{scenario2}
请从能效、经济性、环保性三个维度进行对比，并给出推荐方案。
"""
        return self.analyze_scenario(prompt)
if __name__ == '__main__':
    # 测试大模型分析师
    analyst = LLMAnalyst()
    test_data = """
    调度时段: 168小时（一周）
    风电总量: 50000 kWh
    光伏总量: 30000 kWh
    绿氢产量: 1200 kg
    热电联产产热: 8000 kWh
    系统综合能效: 78.5%
    净经济效益: 25000 元
    碳减排量: 12000 kg CO2
    """
    result = analyst.analyze_scenario(test_data)
    print(result)
