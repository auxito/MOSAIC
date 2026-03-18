"""
Evaluator Agent — 评估官。
职责：逐轮评分 + 最终评估报告。
特权能力：能看到原始简历和修改后简历，对比检测夸大。
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

TURN_EVALUATION_PROMPT = """\
你是一位面试评估专家。请评估以下面试交互轮次。

## 轮次信息
- 第 {round_number} 轮（共 {total_rounds} 轮）
- 面试官问题: {question}
- 候选人回答: {answer}

## 评估维度
请从以下维度打分（1-5分），并给出简要评语：

1. **回答质量** (1-5): 回答是否有深度、有条理
2. **技术准确性** (1-5): 技术描述是否正确
3. **沟通表达** (1-5): 表达是否清晰、专业
4. **真实可信度** (1-5): 回答是否可信、有细节支撑

{exaggeration_check}

请以 JSON 格式返回：
{{
  "scores": {{
    "answer_quality": 0,
    "technical_accuracy": 0,
    "communication": 0,
    "credibility": 0
  }},
  "comment": "简要评语",
  "exaggeration_detected": false,
  "exaggeration_details": ""
}}
"""

FINAL_EVALUATION_PROMPT = """\
你是一位资深面试评估专家。请基于整场面试生成最终评估报告。

## 候选人信息
{resume_info}

## 面试过程摘要
{interview_summary}

## 各轮评分
{turn_scores}

## 语义记忆（抽取的事实）
{fact_table}

## 检测到的矛盾
{contradictions}

请生成一份结构化的评估报告，包含：
1. **综合评分** (1-10)
2. **各维度评分** (技术能力/沟通表达/项目经验/团队协作/学习潜力，各1-5)
3. **优势总结** (3-5 条)
4. **改进建议** (3-5 条)
5. **一致性评估** (是否发现矛盾/夸大)
6. **录用建议** (强烈推荐/推荐/待定/不推荐)
7. **详细评语** (200-300字)

请以 JSON 格式返回。
"""


class EvaluatorAgent(BaseAgent):
    """
    评估官 Agent。

    特权：能看到原始简历 + 修改后简历 → 对比检测夸大。

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
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _register_subscriptions(self) -> None:
        self._subscribe(EventType.CONTRADICTION_FOUND, self._on_contradiction)

    async def _on_contradiction(self, event: Event) -> None:
        """记录矛盾作为扣分依据"""
        self._detected_contradictions.append(event.data)

    async def evaluate_turn(
        self,
        question: str,
        answer: str,
        round_number: int,
        total_rounds: int,
    ) -> dict[str, Any]:
        """评估单轮面试"""

        # 夸大检测：对比原始简历和修改后简历
        exaggeration_check = ""
        privileged = self.working_memory.privileged.get_visible(self.memory_policy)
        if "original_resume" in privileged and "modified_resume" in privileged:
            exaggeration_check = (
                "\n## 简历对比（用于夸大检测）\n"
                f"**原始简历摘要**: {str(privileged['original_resume'])[:500]}\n"
                f"**面试用简历**: {str(privileged['modified_resume'])[:500]}\n"
                "请特别关注候选人的回答是否超出了原始简历的范围。"
            )

        prompt = TURN_EVALUATION_PROMPT.format(
            round_number=round_number,
            total_rounds=total_rounds,
            question=question,
            answer=answer,
            exaggeration_check=exaggeration_check,
        )

        try:
            response = await self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
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

    async def generate_final_report(self) -> dict[str, Any]:
        """生成最终评估报告"""

        privileged = self.working_memory.privileged.get_visible(self.memory_policy)

        # 简历信息
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

        prompt = FINAL_EVALUATION_PROMPT.format(
            resume_info=resume_info or "无简历信息",
            interview_summary=self.working_memory.conversation_summary or "无摘要",
            turn_scores=turn_scores_text or "无评分记录",
            fact_table=self.working_memory.semantic.format_fact_table(),
            contradictions=contradictions_text,
        )

        try:
            response = await self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
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
