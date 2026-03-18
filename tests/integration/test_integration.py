"""
集成测试 — 端到端工作流测试（不调用真实 LLM）。
"""

from __future__ import annotations

import pytest

from mosaic.core.agent import AgentState, EVALUATOR_POLICY, INTERVIEWER_POLICY, INTERVIEWEE_POLICY
from mosaic.core.events import EventBus, Event, EventType
from mosaic.core.memory import (
    ContextWindowManager,
    DialogueTurn,
    EpisodicMemory,
    PrivilegedMemory,
    SemanticFact,
    SemanticMemory,
    WorkingMemory,
    Confidence,
    Contradiction,
)
from mosaic.core.workflow import FULL_INTERVIEW, Phase
from mosaic.resume.schema import ResumeData
from mosaic.resume.structured_input import create_sample_resume


class TestEndToEndMemory:
    """模拟20轮对话，验证记忆系统"""

    def _build_memory_system(self) -> WorkingMemory:
        return WorkingMemory(
            episodic=EpisodicMemory(),
            semantic=SemanticMemory(),
            privileged=PrivilegedMemory(),
            context_manager=ContextWindowManager(
                token_budget=12000, verbatim_window=6, summary_batch=5
            ),
        )

    def test_20_round_simulation(self) -> None:
        """模拟20轮对话，验证记忆系统各层协作"""
        wm = self._build_memory_system()

        # 设置特权信息
        wm.privileged.set("original_resume", "原始简历：3年Python")
        wm.privileged.set("modified_resume", "修改简历：5年Python，带过20人团队")
        wm.privileged.set("job_description", "高级后端工程师")

        # 模拟20轮对话
        for i in range(1, 21):
            wm.episodic.append(DialogueTurn(
                round_number=i, role="interviewer", content=f"面试问题{i}"
            ))
            wm.episodic.append(DialogueTurn(
                round_number=i, role="interviewee", content=f"回答{i}"
            ))

        assert wm.episodic.total_rounds == 20

        # 添加语义事实
        wm.semantic.add_fact(SemanticFact(
            content="3年Python经验", category="skill",
            round_number=2, confidence=Confidence.HIGH,
        ))
        wm.semantic.add_fact(SemanticFact(
            content="5年Python经验", category="skill",
            round_number=12, confidence=Confidence.HIGH,
        ))

        # 添加矛盾
        wm.semantic.add_contradiction(Contradiction(
            fact_a=wm.semantic.facts[0],
            fact_b=wm.semantic.facts[1],
            description="Python经验年限前后矛盾",
            severity="high",
            detected_at_round=12,
        ))

        # 验证各视角
        interviewer_view = wm.compose_for_agent(
            system_prompt="你是面试官",
            policy=INTERVIEWER_POLICY,
        )
        # 面试官应看到修改简历但不看到原始简历
        system_content = interviewer_view[0]["content"]
        assert "修改简历" in system_content or "modified_resume" in system_content

        interviewee_view = wm.compose_for_agent(
            system_prompt="你是候选人",
            policy=INTERVIEWEE_POLICY,
        )

        evaluator_view = wm.compose_for_agent(
            system_prompt="你是评估官",
            policy=EVALUATOR_POLICY,
        )
        # 评估官应看到两份简历
        eval_content = evaluator_view[0]["content"]
        assert "original_resume" in eval_content or "原始" in eval_content

        # 验证事实表
        fact_table = wm.semantic.format_fact_table()
        assert "矛盾" in fact_table
        assert "Python" in fact_table

    def test_context_window_respects_budget(self) -> None:
        """验证上下文窗口不超出预算"""
        wm = self._build_memory_system()

        # 添加大量对话
        for i in range(1, 31):
            wm.episodic.append(DialogueTurn(
                round_number=i, role="interviewer",
                content=f"{'很长的问题内容 ' * 20}第{i}轮",
            ))
            wm.episodic.append(DialogueTurn(
                round_number=i, role="interviewee",
                content=f"{'很长的回答内容 ' * 20}第{i}轮",
            ))

        wm.conversation_summary = "这是一段面试摘要 " * 10

        messages = wm.compose_for_agent(
            system_prompt="系统提示",
            policy=INTERVIEWER_POLICY,
        )

        # 不应该包含所有30轮（应该被裁剪）
        non_system = [m for m in messages if m["role"] != "system"]
        assert len(non_system) < 60  # 30轮*2 = 60，应该远小于此


class TestResumeSchema:
    def test_sample_resume(self) -> None:
        resume = create_sample_resume()
        assert resume.name == "张三"
        assert len(resume.work_experience) == 2
        assert len(resume.skills) > 0

    def test_resume_to_text(self) -> None:
        resume = create_sample_resume()
        text = resume.to_text()
        assert "张三" in text
        assert "字节跳动" in text
        assert "Python" in text

    def test_resume_roundtrip(self) -> None:
        """序列化往返测试"""
        resume = create_sample_resume()
        data = resume.model_dump()
        restored = ResumeData(**data)
        assert restored.name == resume.name
        assert len(restored.work_experience) == len(resume.work_experience)


class TestEventBusIntegration:
    """事件总线集成测试：矛盾发现 → 面试官追问"""

    @pytest.mark.asyncio
    async def test_contradiction_triggers_interviewer(self) -> None:
        bus = EventBus()
        interviewer_alerts: list[dict] = []

        async def interviewer_on_contradiction(event: Event) -> None:
            interviewer_alerts.append(event.data)

        evaluator_notes: list[dict] = []

        async def evaluator_on_contradiction(event: Event) -> None:
            evaluator_notes.append(event.data)

        # 面试官和评估官都订阅矛盾事件
        bus.subscribe(EventType.CONTRADICTION_FOUND, interviewer_on_contradiction)
        bus.subscribe(EventType.CONTRADICTION_FOUND, evaluator_on_contradiction)

        # MemoryManager 发现矛盾
        await bus.publish(Event(
            type=EventType.CONTRADICTION_FOUND,
            source="memory_manager",
            data={
                "round_number": 8,
                "description": "团队规模矛盾：第3轮说5人，第8轮说20人",
                "severity": "high",
            },
        ))

        # 两者都应收到
        assert len(interviewer_alerts) == 1
        assert interviewer_alerts[0]["severity"] == "high"
        assert len(evaluator_notes) == 1
