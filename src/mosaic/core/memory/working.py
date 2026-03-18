"""
工作记忆 — 每次 LLM 调用时动态组合的上下文。
封装四层记忆的协作逻辑。
"""

from __future__ import annotations

from typing import Any

from mosaic.core.agent import MemoryPolicy
from mosaic.core.memory.context_manager import ContextWindowManager
from mosaic.core.memory.episodic import EpisodicMemory
from mosaic.core.memory.privileged import PrivilegedMemory
from mosaic.core.memory.semantic import SemanticMemory


class WorkingMemory:
    """
    工作记忆：组合四层记忆为 LLM 可用的上下文。

    = 系统提示 + 特权记忆子集 + 语义事实表
      + 历史轮次摘要 + 最近K轮原文 + 当前输入
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        privileged: PrivilegedMemory,
        context_manager: ContextWindowManager,
    ) -> None:
        self.episodic = episodic
        self.semantic = semantic
        self.privileged = privileged
        self.context_manager = context_manager
        self._conversation_summary: str = ""
        self._last_summary_round: int = 0

    @property
    def conversation_summary(self) -> str:
        return self._conversation_summary

    @conversation_summary.setter
    def conversation_summary(self, value: str) -> None:
        self._conversation_summary = value

    @property
    def last_summary_round(self) -> int:
        return self._last_summary_round

    @last_summary_round.setter
    def last_summary_round(self, value: int) -> None:
        self._last_summary_round = value

    def compose_for_agent(
        self,
        system_prompt: str,
        policy: MemoryPolicy,
        current_input: str | None = None,
        role_map: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        """
        为某个 Agent 组合上下文。

        Args:
            system_prompt: Agent 的系统提示
            policy: 该 Agent 的记忆策略
            current_input: 当前轮的输入
            role_map: 对话角色映射 {"interviewer": "assistant", "interviewee": "user"}
        """
        # 1. 特权信息（按策略过滤）
        privileged_info = self.privileged.get_visible(policy)

        # 2. 语义事实表
        fact_table = self.semantic.format_fact_table()

        # 3. 最近K轮原文
        recent = self.episodic.get_last_n_rounds(
            self.context_manager.verbatim_window
        )
        if role_map:
            recent_msgs = [
                {"role": role_map.get(t.role, t.role), "content": t.content}
                for t in recent
            ]
        else:
            recent_msgs = self.episodic.to_messages(recent)

        # 4. 组合
        return self.context_manager.compose_messages(
            system_prompt=system_prompt,
            privileged_info=privileged_info,
            fact_table=fact_table,
            conversation_summary=self._conversation_summary,
            recent_turns=recent_msgs,
            current_input=current_input,
        )

    def needs_summary(self) -> bool:
        """是否需要生成新摘要"""
        return self.context_manager.should_summarize(
            self.episodic.total_rounds, self._last_summary_round
        )
