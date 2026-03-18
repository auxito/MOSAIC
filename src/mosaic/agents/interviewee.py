"""
Interviewee Agent — 面试者（AI模式）。
职责：基于简历人设一致地回答问题。
核心能力：回答前检查语义记忆的一致性。
支持 AI/真人切换。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from jinja2 import Environment, FileSystemLoader

from mosaic.core.agent import INTERVIEWEE_POLICY, BaseAgent
from mosaic.core.events import Event, EventType
from mosaic.core.memory import WorkingMemory
from mosaic.llm.client import LLMClient

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@runtime_checkable
class Participant(Protocol):
    """面试者槽位接口 — AI 和真人实现同一协议"""

    async def respond(self, question: str, context: dict[str, Any]) -> str: ...


INTERVIEWEE_SYSTEM_FALLBACK = """\
你是一位正在参加面试的候选人。请基于你的简历和经历来回答面试官的问题。

## 关键要求

1. **保持一致性**: 你的回答必须与简历内容一致，也必须与之前的回答一致
2. **自然真实**: 像真人一样回答，可以有思考停顿，不必面面俱到
3. **适度谦虚**: 承认不足，但也展示学习能力
4. **注意细节**: 如果提到数字（团队大小、性能提升等），保持前后一致

## 一致性检查

在回答前，请检查以下已知事实，确保不与之前的说法矛盾：
{fact_table}

如果面试官问到你不确定的领域，诚实说明你的了解程度。
"""


class IntervieweeAgent(BaseAgent):
    """
    面试者 Agent。

    核心设计：
    - 委托给 Participant（AI 或真人）
    - AI 模式下：自动检查一致性后生成回答
    - 真人模式下：将问题传给用户，回答仍享有记忆系统支持

    事件发布：
    - ANSWER_GIVEN — 回答完成
    """

    def __init__(
        self,
        llm: LLMClient,
        working_memory: WorkingMemory,
        participant: Participant | None = None,
    ) -> None:
        super().__init__(name="interviewee", memory_policy=INTERVIEWEE_POLICY)
        self.llm = llm
        self.working_memory = working_memory
        self._participant = participant
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def set_participant(self, participant: Participant) -> None:
        """切换参与者（AI ↔ 真人）"""
        self._participant = participant

    async def answer(
        self,
        question: str,
        current_round: int,
    ) -> str:
        """回答面试问题"""

        context = {
            "question": question,
            "current_round": current_round,
            "fact_table": self.working_memory.semantic.format_fact_table(),
        }

        if self._participant:
            # 委托给 Participant（可能是 AI 或真人）
            answer = await self._participant.respond(question, context)
        else:
            # 默认 AI 回答
            answer = await self._ai_respond(question, current_round)

        await self._emit(Event(
            type=EventType.ANSWER_GIVEN,
            source=self.name,
            data={
                "question": question,
                "answer": answer,
                "round_number": current_round,
            },
        ))

        return answer

    async def _ai_respond(self, question: str, current_round: int) -> str:
        """AI 模式生成回答"""
        fact_table = self.working_memory.semantic.format_fact_table()

        try:
            template = self._jinja_env.get_template("interviewee_system.j2")
            system_prompt = template.render(fact_table=fact_table)
        except Exception:
            system_prompt = INTERVIEWEE_SYSTEM_FALLBACK.format(
                fact_table=fact_table
            )

        # 面试者视角：面试官是 assistant（提问方），自己是 user→assistant
        messages = self.working_memory.compose_for_agent(
            system_prompt=system_prompt,
            policy=self.memory_policy,
            current_input=question,
            role_map={"interviewer": "user", "interviewee": "assistant"},
        )

        answer = await self.llm.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )

        return answer

    async def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Agent 接口"""
        answer = await self.answer(
            question=context.get("question", ""),
            current_round=context.get("current_round", 1),
        )
        return {"answer": answer}
