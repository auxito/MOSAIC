"""
事件系统 — 事件驱动架构的核心
Agent 之间通过事件总线通信，不直接耦合。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """所有事件类型"""

    # 工作流控制
    PHASE_TRANSITION = auto()
    WORKFLOW_COMPLETE = auto()

    # 简历相关
    RESUME_PARSED = auto()
    RESUME_MODIFIED = auto()

    # 面试过程
    QUESTION_POSED = auto()
    ANSWER_GIVEN = auto()
    INTERVIEW_COMPLETE = auto()

    # 记忆系统
    FACT_EXTRACTED = auto()
    CONTRADICTION_FOUND = auto()
    SUMMARY_GENERATED = auto()

    # 评估
    TURN_EVALUATED = auto()


@dataclass
class Event:
    """不可变事件对象"""

    type: EventType
    source: str  # 发送方 Agent 名称
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"Event({self.type.name}, src={self.source}, id={self.event_id[:8]})"


# 事件处理器类型
EventHandler = Any  # Callable[[Event], Awaitable[None]] — 避免循环导入用 Any


class EventBus:
    """
    发布-订阅事件总线。
    Agent 订阅感兴趣的事件类型，Orchestrator 发布事件后自动路由。
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._history: list[Event] = []

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """订阅某类事件"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """取消订阅"""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    async def publish(self, event: Event) -> None:
        """发布事件 — 按订阅顺序异步调用所有处理器"""
        self._history.append(event)
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            await handler(event)

    @property
    def history(self) -> list[Event]:
        """只读事件历史"""
        return list(self._history)

    def clear(self) -> None:
        """重置（测试用）"""
        self._handlers.clear()
        self._history.clear()
