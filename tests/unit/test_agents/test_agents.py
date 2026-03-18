"""
Agent 单元测试。
测试事件系统、Agent 基类、工作流配置。
"""

from __future__ import annotations

import asyncio

import pytest

from mosaic.core.agent import AgentState, BaseAgent, MemoryPolicy
from mosaic.core.events import Event, EventBus, EventType
from mosaic.core.workflow import (
    FULL_INTERVIEW,
    RESUME_ONLY,
    Phase,
    WorkflowConfig,
)


# ============ EventBus ============


class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.QUESTION_POSED, handler)

        event = Event(
            type=EventType.QUESTION_POSED,
            source="test",
            data={"question": "你好"},
        )
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].data["question"] == "你好"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self) -> None:
        bus = EventBus()
        counts = {"a": 0, "b": 0}

        async def handler_a(event: Event) -> None:
            counts["a"] += 1

        async def handler_b(event: Event) -> None:
            counts["b"] += 1

        bus.subscribe(EventType.CONTRADICTION_FOUND, handler_a)
        bus.subscribe(EventType.CONTRADICTION_FOUND, handler_b)

        await bus.publish(Event(
            type=EventType.CONTRADICTION_FOUND, source="test"
        ))

        assert counts["a"] == 1
        assert counts["b"] == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        bus = EventBus()
        count = 0

        async def handler(event: Event) -> None:
            nonlocal count
            count += 1

        bus.subscribe(EventType.FACT_EXTRACTED, handler)
        await bus.publish(Event(type=EventType.FACT_EXTRACTED, source="test"))
        assert count == 1

        bus.unsubscribe(EventType.FACT_EXTRACTED, handler)
        await bus.publish(Event(type=EventType.FACT_EXTRACTED, source="test"))
        assert count == 1  # 不再增加

    @pytest.mark.asyncio
    async def test_no_handler(self) -> None:
        """发布无人订阅的事件不应报错"""
        bus = EventBus()
        await bus.publish(Event(type=EventType.WORKFLOW_COMPLETE, source="test"))

    @pytest.mark.asyncio
    async def test_event_history(self) -> None:
        bus = EventBus()
        await bus.publish(Event(type=EventType.PHASE_TRANSITION, source="a"))
        await bus.publish(Event(type=EventType.QUESTION_POSED, source="b"))

        assert len(bus.history) == 2
        assert bus.history[0].source == "a"


# ============ Workflow ============


class TestWorkflow:
    def test_phase_sequence(self) -> None:
        wf = FULL_INTERVIEW
        assert wf.phases[0] == Phase.INIT
        assert wf.phases[-1] == Phase.COMPLETE

    def test_next_phase(self) -> None:
        wf = FULL_INTERVIEW
        assert wf.next_phase(Phase.INIT) == Phase.RESUME_INPUT
        assert wf.next_phase(Phase.COMPLETE) is None

    def test_has_phase(self) -> None:
        assert FULL_INTERVIEW.has_phase(Phase.INTERVIEW_LOOP)
        assert not RESUME_ONLY.has_phase(Phase.INTERVIEW_LOOP)

    def test_custom_workflow(self) -> None:
        wf = WorkflowConfig(
            name="custom",
            phases=[Phase.INIT, Phase.INTERVIEW_LOOP, Phase.COMPLETE],
            interview_rounds=5,
        )
        assert wf.interview_rounds == 5
        assert wf.next_phase(Phase.INIT) == Phase.INTERVIEW_LOOP


# ============ AgentState ============


class TestAgentState:
    def test_default_state(self) -> None:
        state = AgentState()
        assert state.current_round == 0
        assert state.conversation == []
        assert state.semantic_facts == []

    def test_state_mutation(self) -> None:
        state = AgentState(interview_rounds=5)
        state.current_round = 3
        state.conversation.append({"role": "interviewer", "content": "Q"})
        assert state.current_round == 3
        assert len(state.conversation) == 1


# ============ MemoryPolicy ============


class TestMemoryPolicy:
    def test_default_policy(self) -> None:
        policy = MemoryPolicy()
        assert not policy.can_see_original_resume
        assert not policy.can_see_modified_resume
        assert not policy.can_see_jd

    def test_evaluator_policy(self) -> None:
        from mosaic.core.agent import EVALUATOR_POLICY
        assert EVALUATOR_POLICY.can_see_original_resume
        assert EVALUATOR_POLICY.can_see_modified_resume
        assert EVALUATOR_POLICY.can_see_jd
        assert EVALUATOR_POLICY.can_see_evaluation
