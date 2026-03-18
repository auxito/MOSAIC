"""
Agent 抽象基类 + 记忆访问策略。
所有 Agent 继承 BaseAgent，通过 MemoryPolicy 控制信息可见性。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mosaic.core.events import Event, EventBus, EventType


@dataclass
class MemoryPolicy:
    """
    控制某个 Agent 能看到哪些特权信息。

    can_see_original_resume: 能否看到原始简历（只有 Evaluator 能）
    can_see_modified_resume: 能否看到修改后简历
    can_see_jd: 能否看到 JD
    can_see_evaluation: 能否看到评分标准/评分结果
    """

    can_see_original_resume: bool = False
    can_see_modified_resume: bool = False
    can_see_jd: bool = False
    can_see_evaluation: bool = False


# 预定义策略
INTERVIEWER_POLICY = MemoryPolicy(
    can_see_modified_resume=True, can_see_jd=True
)
INTERVIEWEE_POLICY = MemoryPolicy(
    can_see_modified_resume=True
)
EVALUATOR_POLICY = MemoryPolicy(
    can_see_original_resume=True,
    can_see_modified_resume=True,
    can_see_jd=True,
    can_see_evaluation=True,
)
RESUME_MODIFIER_POLICY = MemoryPolicy(
    can_see_original_resume=True, can_see_jd=True
)
MEMORY_MANAGER_POLICY = MemoryPolicy()  # 不需要特权信息，只处理对话


class BaseAgent(ABC):
    """
    所有 Agent 的基类。
    - name: 唯一标识
    - memory_policy: 信息可见性策略
    - event_bus: 事件总线引用（由 Orchestrator 注入）
    """

    def __init__(
        self,
        name: str,
        memory_policy: MemoryPolicy | None = None,
    ) -> None:
        self.name = name
        self.memory_policy = memory_policy or MemoryPolicy()
        self._event_bus: EventBus | None = None
        self.logger = logging.getLogger(f"agent.{name}")

    def bind_event_bus(self, bus: EventBus) -> None:
        """Orchestrator 调用，注入事件总线并注册订阅"""
        self._event_bus = bus
        self._register_subscriptions()

    def _register_subscriptions(self) -> None:
        """子类覆写以订阅感兴趣的事件"""
        pass

    def _subscribe(self, event_type: EventType, handler: Any) -> None:
        """便捷方法：订阅事件"""
        if self._event_bus:
            self._event_bus.subscribe(event_type, handler)

    async def _emit(self, event: Event) -> None:
        """便捷方法：发布事件"""
        if self._event_bus:
            await self._event_bus.publish(event)

    @abstractmethod
    async def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Agent 的核心处理方法。
        接收上下文，返回结果。具体语义由子类定义。
        """
        ...


@dataclass
class AgentState:
    """
    全局共享状态 — Orchestrator 持有，各阶段传递。
    """

    # 简历数据
    original_resume: dict[str, Any] = field(default_factory=dict)
    modified_resumes: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_resume_style: str = ""
    selected_resume: dict[str, Any] = field(default_factory=dict)

    # JD
    job_description: str = ""

    # 面试配置
    interview_rounds: int = 10
    current_round: int = 0

    # 对话历史（episodic memory 的源）
    conversation: list[dict[str, str]] = field(default_factory=list)

    # 语义记忆中抽取的事实
    semantic_facts: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)

    # 渐进式摘要
    conversation_summary: str = ""

    # 评估结果
    turn_evaluations: list[dict[str, Any]] = field(default_factory=list)
    final_evaluation: dict[str, Any] = field(default_factory=dict)

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)
