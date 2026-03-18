"""
Orchestrator — 状态机 + 事件总线。
驱动整个面试工作流的核心引擎。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mosaic.core.agent import AgentState, BaseAgent
from mosaic.core.events import Event, EventBus, EventType
from mosaic.core.memory import (
    ContextWindowManager,
    EpisodicMemory,
    PrivilegedMemory,
    SemanticMemory,
    WorkingMemory,
)
from mosaic.core.workflow import Phase, WorkflowConfig

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    面试模拟系统的核心编排器。

    职责：
    1. 管理工作流状态机（Phase 转换）
    2. 维护事件总线（Agent 间通信）
    3. 持有四层记忆系统
    4. 按阶段调度 Agent
    """

    def __init__(self, workflow: WorkflowConfig) -> None:
        self.workflow = workflow
        self.state = AgentState(interview_rounds=workflow.interview_rounds)
        self.current_phase: Phase = workflow.phases[0]

        # 事件总线
        self.event_bus = EventBus()

        # 四层记忆
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.privileged = PrivilegedMemory()
        self.context_manager = ContextWindowManager()
        self.working_memory = WorkingMemory(
            episodic=self.episodic,
            semantic=self.semantic,
            privileged=self.privileged,
            context_manager=self.context_manager,
        )

        # Agent 注册表
        self._agents: dict[str, BaseAgent] = {}

        # 阶段处理器注册表
        self._phase_handlers: dict[Phase, Any] = {}

        # 运行时状态
        self._start_time: float = 0
        self._phase_log: list[dict[str, Any]] = []

    def register_agent(self, agent: BaseAgent) -> None:
        """注册 Agent 并绑定事件总线"""
        self._agents[agent.name] = agent
        agent.bind_event_bus(self.event_bus)
        logger.info(f"Registered agent: {agent.name}")

    def register_phase_handler(self, phase: Phase, handler: Any) -> None:
        """注册阶段处理器"""
        self._phase_handlers[phase] = handler

    def get_agent(self, name: str) -> BaseAgent:
        """获取已注册的 Agent"""
        if name not in self._agents:
            raise KeyError(f"Agent not found: {name}")
        return self._agents[name]

    async def run(self) -> AgentState:
        """运行整个工作流"""
        self._start_time = time.time()
        logger.info(f"Starting workflow: {self.workflow.name}")

        while self.current_phase != Phase.COMPLETE:
            await self._execute_phase(self.current_phase)
            next_phase = self.workflow.next_phase(self.current_phase)
            if next_phase is None:
                break
            await self._transition_to(next_phase)

        elapsed = time.time() - self._start_time
        logger.info(f"Workflow complete in {elapsed:.1f}s")
        self.state.metadata["elapsed_time"] = elapsed
        self.state.metadata["phase_log"] = self._phase_log

        await self.event_bus.publish(
            Event(type=EventType.WORKFLOW_COMPLETE, source="orchestrator")
        )
        return self.state

    async def _execute_phase(self, phase: Phase) -> None:
        """执行单个阶段"""
        phase_start = time.time()
        logger.info(f"=== Phase: {phase.name} ===")

        handler = self._phase_handlers.get(phase)
        if handler:
            await handler(self)
        else:
            logger.warning(f"No handler for phase {phase.name}, skipping")

        elapsed = time.time() - phase_start
        self._phase_log.append({
            "phase": phase.name,
            "elapsed": elapsed,
        })

    async def _transition_to(self, next_phase: Phase) -> None:
        """状态转换"""
        prev = self.current_phase
        self.current_phase = next_phase
        logger.info(f"Phase transition: {prev.name} → {next_phase.name}")

        await self.event_bus.publish(
            Event(
                type=EventType.PHASE_TRANSITION,
                source="orchestrator",
                data={"from": prev.name, "to": next_phase.name},
            )
        )
