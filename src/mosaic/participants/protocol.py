"""
Participant Protocol — 面试者槽位的抽象接口。
AI 和真人实现同一接口，IntervieweeAgent 不感知差异。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Participant(Protocol):
    """面试参与者协议"""

    async def respond(self, question: str, context: dict[str, Any]) -> str:
        """
        对面试问题给出回答。

        Args:
            question: 面试官的问题
            context: 附加上下文（事实表、当前轮次等）

        Returns:
            回答文本
        """
        ...
