"""
Evaluator Agent — 评估官。
职责：逐轮评分 + 最终评估报告。
特权能力：能看到原始简历和修改后简历，对比检测夸大。

使用 evaluator_system.j2 模板作为 system message，提供完整的：
- 5 维评分标准（每个维度 1-5 的详细描述）
- 教练视角指南
- 对比检测指南
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from mosaic.core.agent import EVALUATOR_POLICY, BaseAgent
from mosaic.core.events import Event, EventType
from mosaic.core.memory import WorkingMemory
from mosaic.llm.client import LLMClient

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# 完整 fallback — 与 evaluator_system.j2 内容对齐
_EVALUATOR_SYSTEM_FALLBACK = """\
你是一位资深的**职业发展教练型评估官**。你的任务不仅是评分，更是帮助候选人成长。

## 你的特殊权限

你同时拥有候选人的**原始简历**和**面试所用简历**。这使你能够：
1. 评估候选人的真实水平与简历呈现之间的差距
2. 发现候选人的知识盲区和成长空间
3. 给出针对性的改进建议和学习路线

## 评分维度

### 技术能力 (1-5)
- 5: 技术深度和广度远超岗位要求，能深入原理
- 4: 技术扎实，能胜任岗位，理解设计权衡
- 3: 基本达标，部分领域需要深入
- 2: 明显不足，停留在使用层面
- 1: 严重不达标

### 沟通表达 (1-5)
- 5: 表达清晰有条理，善于用类比和实例，有结构化思维
- 4: 沟通流畅，能清楚传达想法
- 3: 基本能沟通，偶尔表达模糊
- 2: 沟通困难，逻辑不清
- 1: 无法有效沟通

### 项目经验 (1-5)
- 5: 深度参与核心项目，有显著技术贡献，能讲清架构
- 4: 有扎实的项目经验，能说清技术细节
- 3: 有相关经验但深度不够
- 2: 项目经验较少或理解表面
- 1: 几乎没有相关项目经验

### 团队协作 (1-5)
- 5: 展现出优秀的技术领导力和跨团队协作能力
- 4: 能良好地融入团队，有效沟通
- 3: 基本的团队协作能力
- 2: 协作能力有待提升
- 1: 可能存在团队协作问题

### 学习潜力 (1-5)
- 5: 学习能力强，有明确的技术成长路径，关注前沿
- 4: 愿意学习，有一定的自驱力
- 3: 需要指导但有学习意愿
- 2: 学习意愿不强
- 1: 缺乏学习动力

## 教练视角

在评估的同时，请特别关注：
- **亮点**: 候选人在哪些方面表现出色，值得保持
- **改进空间**: 哪些回答可以更好，具体如何改进
- **知识盲区**: 候选人在哪些技术领域存在明显短板
- **参考答案**: 对于回答不够好的问题，给出参考答案的要点

## 对比检测

特别关注以下信号：
- 简历中的技术深度 vs 面试中的实际表现
- 候选人是否能深入解释简历中提到的技术
- 被追问时是否能给出具体的细节和例子
- 技术广度 vs 技术深度的平衡
"""

TURN_EVALUATION_PROMPT = """\
请评估以下面试交互轮次，并给出改进建议。

## 轮次信息
- 第 {round_number} 轮（共 {total_rounds} 轮）
- 面试官问题: {question}
- 候选人回答: {answer}

{exaggeration_check}

## 评估要求
请根据 system message 中的评分标准，从以下 4 个维度打分（1-5分）：
1. **回答质量**: 回答是否有深度、有条理（参照技术能力标准）
2. **技术准确性**: 技术描述是否正确（参照技术能力标准）
3. **沟通表达**: 表达是否清晰、专业（参照沟通表达标准）
4. **真实可信度**: 回答是否可信、有细节支撑（参照对比检测指南）

## 覆盖维度判断
请判断本轮问答主要覆盖了以下哪些面试维度（可多选）：
- "项目经历深挖": 深入讨论了具体项目的技术细节、架构、个人贡献
- "技术纵深考察": 考察了核心技术原理、设计权衡、底层机制
- "系统设计思维": 涉及系统架构、扩展性、高可用等设计问题
- "团队协作与沟通": 涉及团队合作、跨部门沟通、领导力等
- "技术视野与学习力": 涉及技术趋势、学习方法、职业发展

请以 JSON 格式返回：
{{
  "scores": {{
    "answer_quality": 0,
    "technical_accuracy": 0,
    "communication": 0,
    "credibility": 0
  }},
  "covered_dimensions": ["本轮覆盖的维度名称"],
  "comment": "简要评语",
  "highlights": "本轮回答的亮点",
  "improvement_suggestions": "具体的改进建议：候选人如何能回答得更好",
  "reference_points": ["优秀回答应该包含的要点1", "要点2", "要点3"],
  "knowledge_gaps": "发现的知识盲区（如有）",
  "exaggeration_detected": false,
  "exaggeration_details": ""
}}
"""

FINAL_EVALUATION_PROMPT = """\
请基于整场面试生成最终评估报告，重点给出成长建议。

## 候选人信息
{resume_info}

## 面试过程摘要
{interview_summary}

## 各轮评分
{turn_scores}

## 跨轮趋势分析
{trend_analysis}

## 语义记忆（抽取的事实）
{fact_table}

## 检测到的矛盾
{contradictions}

## 评估要求
请根据 system message 中的 5 维评分标准（技术能力/沟通表达/项目经验/团队协作/学习潜力），
结合教练视角和对比检测指南，生成结构化评估报告。

特别注意：
- 结合**趋势分析**中的表现变化情况，评估候选人的心态稳定性和深度潜力
- 如果候选人在后半段明显提升/下滑，在评语中说明原因
- 弱势维度应在 learning_roadmap 中给出具体提升路径

请严格使用以下 JSON 格式返回：
{{
  "overall_score": 7,
  "dimension_scores": {{
    "技术能力": 4,
    "沟通表达": 3,
    "项目经验": 4,
    "团队协作": 3,
    "学习潜力": 4
  }},
  "strengths": ["优势1", "优势2", "优势3"],
  "improvements": ["改进建议1", "改进建议2", "改进建议3"],
  "trend_summary": "面试表现趋势总结（进步/稳定/下滑及原因）",
  "consistency_assessment": "一致性评估（是否发现矛盾/知识盲区）",
  "recommendation": "强烈推荐/推荐/待定/不推荐",
  "detailed_comments": "200-300字的详细评语",
  "coaching_notes": "整体教练建议：候选人最需要提升的3个方面及具体方法",
  "learning_roadmap": [
    {{
      "area": "技术领域名称",
      "current_level": "当前水平描述",
      "target_level": "目标水平描述",
      "resources": ["推荐资源1", "推荐资源2"],
      "timeline": "建议学习周期"
    }}
  ]
}}

注意：overall_score 为 1-10 整数，dimension_scores 各项为 1-5 整数。
learning_roadmap 应包含 2-4 个最需要提升的技术领域。
"""


class EvaluatorAgent(BaseAgent):
    """
    评估官 Agent。

    特权：能看到原始简历 + 修改后简历 → 对比检测夸大。
    使用 evaluator_system.j2 作为 system message，提供完整评分标准。

    事件订阅：
    - CONTRADICTION_FOUND → 记录为扣分项

    事件发布：
    - TURN_EVALUATED — 单轮评分
    """

    def __init__(self, llm: LLMClient, working_memory: WorkingMemory) -> None:
        super().__init__(name="evaluator", memory_policy=EVALUATOR_POLICY)
        self.llm = llm
        self.working_memory = working_memory
        self._turn_scores: list[dict[str, Any]] = []
        self._detected_contradictions: list[dict[str, Any]] = []
        self._resume_diff_summary: str = ""  # 简历差异摘要（一次生成，多次复用）
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _get_system_prompt(self) -> str:
        """加载评估官的 system prompt（含完整评分标准）"""
        try:
            template = self._jinja_env.get_template("evaluator_system.j2")
            return template.render()
        except Exception:
            logger.warning("Failed to load evaluator_system.j2, using fallback")
            return _EVALUATOR_SYSTEM_FALLBACK

    def _register_subscriptions(self) -> None:
        self._subscribe(EventType.CONTRADICTION_FOUND, self._on_contradiction)

    async def _on_contradiction(self, event: Event) -> None:
        """记录矛盾作为扣分依据"""
        self._detected_contradictions.append(event.data)

    async def generate_resume_diff_summary(self) -> str:
        """生成简历差异摘要 — 一次性调 LLM 对比原始和优化简历的核心差异。

        在简历修改完成后调用一次，后续每轮评估复用这份摘要，
        替代每次截断原文的粗暴做法。这样既保留了完整信息，又节省 token。

        Returns:
            结构化的差异摘要文本
        """
        privileged = self.working_memory.privileged.get_visible(self.memory_policy)
        original = str(privileged.get("original_resume", ""))
        modified = str(privileged.get("modified_resume", ""))

        if not original or not modified:
            self._resume_diff_summary = ""
            return ""

        prompt = (
            "你是一个简历对比分析专家。请对比以下两份简历，生成结构化的差异摘要。\n\n"
            f"## 原始简历\n{original}\n\n"
            f"## 优化后简历\n{modified}\n\n"
            "请按以下格式输出差异摘要（控制在 500 字以内）：\n\n"
            "### 基本事实（未变更）\n"
            "- 列出教育背景、公司名、职位、时间线等不应改变的事实\n\n"
            "### 技术深度扩展\n"
            "- 列出优化版本中新增或深化的技术描述（原文→优化后）\n\n"
            "### 成果量化变化\n"
            "- 列出数字、指标、成果方面的变化\n\n"
            "### 夸大风险点\n"
            "- 列出候选人最可能在面试中无法自圆其说的扩展内容\n"
        )

        try:
            summary = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            self._resume_diff_summary = summary.strip()
            logger.info(
                f"Resume diff summary generated: {len(self._resume_diff_summary)} chars"
            )
            return self._resume_diff_summary
        except Exception as e:
            logger.error(f"Resume diff summary generation failed: {e}")
            self._resume_diff_summary = ""
            return ""

    def _get_exaggeration_context(self) -> str:
        """获取夸大检测上下文 — 优先用差异摘要，否则 fallback 到截断原文。"""
        if self._resume_diff_summary:
            return (
                "\n## 简历差异摘要（用于夸大检测）\n"
                f"{self._resume_diff_summary}\n\n"
                "请特别关注候选人的回答是否涉及「夸大风险点」中的内容，"
                "以及回答深度是否与「技术深度扩展」匹配。"
            )

        # Fallback：仍用截断（但不应走到这里）
        privileged = self.working_memory.privileged.get_visible(self.memory_policy)
        if "original_resume" in privileged and "modified_resume" in privileged:
            return (
                "\n## 简历对比（用于夸大检测）\n"
                f"**原始简历摘要**: {str(privileged['original_resume'])[:800]}\n"
                f"**面试用简历**: {str(privileged['modified_resume'])[:800]}\n"
                "请特别关注候选人的回答是否超出了原始简历的范围。"
            )
        return ""

    async def evaluate_turn(
        self,
        question: str,
        answer: str,
        round_number: int,
        total_rounds: int,
    ) -> dict[str, Any]:
        """评估单轮面试"""

        exaggeration_check = self._get_exaggeration_context()

        user_prompt = TURN_EVALUATION_PROMPT.format(
            round_number=round_number,
            total_rounds=total_rounds,
            question=question,
            answer=answer,
            exaggeration_check=exaggeration_check,
        )

        try:
            response = await self.llm.chat_json(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            evaluation = json.loads(response)
            evaluation["round_number"] = round_number
            self._turn_scores.append(evaluation)

            await self._emit(Event(
                type=EventType.TURN_EVALUATED,
                source=self.name,
                data=evaluation,
            ))

            return evaluation
        except Exception as e:
            logger.error(f"Turn evaluation failed: {e}")
            return {"error": str(e), "round_number": round_number}

    def _compute_trend_analysis(self) -> str:
        """计算跨轮趋势分析 — 纯数据计算，不调 LLM。

        分析维度：
        1. 总体趋势（前半 vs 后半平均分）
        2. 各维度平均分 + 最弱维度
        3. 分数变化趋势（逐轮移动平均）
        4. 稳定性（标准差）
        """
        if not self._turn_scores:
            return "无评分数据，无法分析趋势。"

        # 提取各维度逐轮分数
        dims = ["answer_quality", "technical_accuracy", "communication", "credibility"]
        dim_labels = {
            "answer_quality": "回答质量",
            "technical_accuracy": "技术准确性",
            "communication": "沟通表达",
            "credibility": "真实可信度",
        }

        dim_scores: dict[str, list[float]] = {d: [] for d in dims}
        round_avgs: list[float] = []

        for ts in self._turn_scores:
            scores = ts.get("scores", {})
            vals = []
            for d in dims:
                v = scores.get(d)
                if isinstance(v, (int, float)):
                    dim_scores[d].append(float(v))
                    vals.append(float(v))
            if vals:
                round_avgs.append(sum(vals) / len(vals))

        if not round_avgs:
            return "评分数据不完整，无法分析趋势。"

        lines = []

        # 1. 总体趋势
        overall_avg = sum(round_avgs) / len(round_avgs)
        lines.append(f"**总体平均分**: {overall_avg:.2f}/5")

        if len(round_avgs) >= 4:
            mid = len(round_avgs) // 2
            first_half = sum(round_avgs[:mid]) / mid
            second_half = sum(round_avgs[mid:]) / (len(round_avgs) - mid)
            diff = second_half - first_half
            if diff > 0.3:
                trend = f"📈 明显上升（前半 {first_half:.1f} → 后半 {second_half:.1f}，+{diff:.1f}）"
            elif diff < -0.3:
                trend = f"📉 明显下滑（前半 {first_half:.1f} → 后半 {second_half:.1f}，{diff:.1f}）"
            else:
                trend = f"➡️ 基本稳定（前半 {first_half:.1f} → 后半 {second_half:.1f}）"
            lines.append(f"**表现趋势**: {trend}")
        elif len(round_avgs) >= 2:
            if round_avgs[-1] > round_avgs[0] + 0.5:
                lines.append("**表现趋势**: 📈 呈上升趋势")
            elif round_avgs[-1] < round_avgs[0] - 0.5:
                lines.append("**表现趋势**: 📉 呈下滑趋势")
            else:
                lines.append("**表现趋势**: ➡️ 基本稳定")

        # 2. 各维度分析
        lines.append("\n**各维度平均分:**")
        weakest_dim = ""
        weakest_score = 6.0
        for d in dims:
            if dim_scores[d]:
                avg = sum(dim_scores[d]) / len(dim_scores[d])
                lines.append(f"  - {dim_labels[d]}: {avg:.2f}")
                if avg < weakest_score:
                    weakest_score = avg
                    weakest_dim = dim_labels[d]

        if weakest_dim:
            lines.append(f"\n**最薄弱维度**: {weakest_dim}（{weakest_score:.2f}）")

        # 3. 稳定性
        if len(round_avgs) >= 3:
            mean = sum(round_avgs) / len(round_avgs)
            variance = sum((x - mean) ** 2 for x in round_avgs) / len(round_avgs)
            std = variance ** 0.5
            if std < 0.3:
                stability = "表现稳定，波动小"
            elif std < 0.6:
                stability = "表现有一定波动"
            else:
                stability = "表现波动较大，稳定性不足"
            lines.append(f"**稳定性**: {stability}（标准差 {std:.2f}）")

        # 4. 逐轮分数一览
        lines.append(f"\n**逐轮平均分**: {' → '.join(f'{s:.1f}' for s in round_avgs)}")

        return "\n".join(lines)

    async def generate_final_report(self) -> dict[str, Any]:
        """生成最终评估报告"""

        # 简历信息：优先用差异摘要，更精准也更省 token
        if self._resume_diff_summary:
            resume_info = (
                "## 简历差异摘要\n"
                f"{self._resume_diff_summary}\n"
            )
        else:
            privileged = self.working_memory.privileged.get_visible(self.memory_policy)
            resume_info = ""
            if "original_resume" in privileged:
                resume_info += f"**原始简历**: {str(privileged['original_resume'])[:800]}\n"
            if "modified_resume" in privileged:
                resume_info += f"**面试用简历**: {str(privileged['modified_resume'])[:800]}\n"

        # 各轮评分汇总
        turn_scores_text = ""
        for ts in self._turn_scores:
            scores = ts.get("scores", {})
            turn_scores_text += (
                f"第{ts.get('round_number', '?')}轮: "
                f"质量={scores.get('answer_quality', '?')}, "
                f"技术={scores.get('technical_accuracy', '?')}, "
                f"沟通={scores.get('communication', '?')}, "
                f"可信={scores.get('credibility', '?')}"
                f" | {ts.get('comment', '')}\n"
            )

        # 跨轮趋势分析
        trend_analysis = self._compute_trend_analysis()

        # 矛盾记录
        contradictions_text = ""
        if self._detected_contradictions:
            for c in self._detected_contradictions:
                contradictions_text += (
                    f"- [第{c.get('round_number', '?')}轮] "
                    f"{c.get('description', '')}\n"
                )
        else:
            contradictions_text = "未检测到明显矛盾。"

        user_prompt = FINAL_EVALUATION_PROMPT.format(
            resume_info=resume_info or "无简历信息",
            interview_summary=self.working_memory.conversation_summary or "无摘要",
            turn_scores=turn_scores_text or "无评分记录",
            trend_analysis=trend_analysis,
            fact_table=self.working_memory.semantic.format_fact_table(),
            contradictions=contradictions_text,
        )

        try:
            response = await self.llm.chat_json(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            report = json.loads(response)
            return report
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            return {"error": str(e)}

    async def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Agent 接口"""
        action = context.get("action")
        if action == "evaluate_turn":
            return await self.evaluate_turn(
                question=context.get("question", ""),
                answer=context.get("answer", ""),
                round_number=context.get("round_number", 0),
                total_rounds=context.get("total_rounds", 10),
            )
        elif action == "final_report":
            return await self.generate_final_report()
        return {"error": "Unknown action"}
