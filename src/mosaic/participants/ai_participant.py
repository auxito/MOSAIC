"""
AI Participant — LLM 驱动的面试者。
IntervieweeAgent 的默认委托对象。
"""

from __future__ import annotations

from typing import Any

from mosaic.core.memory import WorkingMemory
from mosaic.core.agent import INTERVIEWEE_POLICY
from mosaic.llm.client import LLMClient


class AIParticipant:
    """
    AI 面试参与者。
    使用 LLM 基于简历人设生成回答。
    """

    def __init__(self, llm: LLMClient, working_memory: WorkingMemory) -> None:
        self.llm = llm
        self.working_memory = working_memory

    async def respond(self, question: str, context: dict[str, Any]) -> str:
        """基于简历和记忆生成回答"""
        fact_table = context.get("fact_table", "")

        system_prompt = (
            "你是一位正在参加面试的候选人。请基于简历内容回答问题。\n"
            "保持与之前回答的一致性。自然真实地回答。\n"
        )
        if fact_table:
            system_prompt += f"\n## 已知事实（确保一致性）\n{fact_table}"

        messages = self.working_memory.compose_for_agent(
            system_prompt=system_prompt,
            policy=INTERVIEWEE_POLICY,
            current_input=question,
            role_map={"interviewer": "user", "interviewee": "assistant"},
        )

        return await self.llm.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )
