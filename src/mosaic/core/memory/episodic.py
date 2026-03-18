"""
情景记忆 — 只追加的完整对话日志。
用于回放、评估、渐进式摘要的源数据。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DialogueTurn:
    """单轮对话"""

    round_number: int
    role: str  # "interviewer" | "interviewee"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """
    情景记忆：不可变的对话历史。
    只追加，不修改，不截断。
    """

    def __init__(self) -> None:
        self._turns: list[DialogueTurn] = []

    def append(self, turn: DialogueTurn) -> None:
        """追加一轮对话"""
        self._turns.append(turn)

    @property
    def turns(self) -> list[DialogueTurn]:
        """只读访问"""
        return list(self._turns)

    @property
    def total_rounds(self) -> int:
        """总轮数（按 round_number 去重）"""
        if not self._turns:
            return 0
        return max(t.round_number for t in self._turns)

    def get_range(self, start: int, end: int | None = None) -> list[DialogueTurn]:
        """获取指定轮次范围的对话"""
        if end is None:
            end = self.total_rounds + 1
        return [t for t in self._turns if start <= t.round_number < end]

    def get_last_n_rounds(self, n: int) -> list[DialogueTurn]:
        """获取最近 n 轮的对话（按 round_number）"""
        if not self._turns:
            return []
        max_round = self.total_rounds
        start = max(1, max_round - n + 1)
        return self.get_range(start, max_round + 1)

    def to_messages(self, turns: list[DialogueTurn] | None = None) -> list[dict[str, str]]:
        """转换为 LLM message 格式"""
        turns = turns or self._turns
        role_map = {"interviewer": "assistant", "interviewee": "user"}
        return [
            {"role": role_map.get(t.role, t.role), "content": t.content}
            for t in turns
        ]

    def clear(self) -> None:
        """重置（测试用）"""
        self._turns.clear()
