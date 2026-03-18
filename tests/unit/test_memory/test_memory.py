"""
记忆系统单元测试 — 重点测试。
覆盖：情景记忆、语义记忆、特权记忆、工作记忆、上下文窗口管理。
"""

from __future__ import annotations

import pytest

from mosaic.core.agent import (
    EVALUATOR_POLICY,
    INTERVIEWEE_POLICY,
    INTERVIEWER_POLICY,
    MemoryPolicy,
)
from mosaic.core.memory.context_manager import ContextWindowManager, count_tokens
from mosaic.core.memory.episodic import DialogueTurn, EpisodicMemory
from mosaic.core.memory.privileged import PrivilegedMemory
from mosaic.core.memory.semantic import (
    Confidence,
    Contradiction,
    SemanticFact,
    SemanticMemory,
)
from mosaic.core.memory.working import WorkingMemory


# ============ EpisodicMemory ============


class TestEpisodicMemory:
    def test_append_and_access(self) -> None:
        mem = EpisodicMemory()
        mem.append(DialogueTurn(round_number=1, role="interviewer", content="问题1"))
        mem.append(DialogueTurn(round_number=1, role="interviewee", content="回答1"))

        assert len(mem.turns) == 2
        assert mem.total_rounds == 1

    def test_get_last_n_rounds(self) -> None:
        mem = EpisodicMemory()
        for i in range(1, 6):
            mem.append(DialogueTurn(round_number=i, role="interviewer", content=f"Q{i}"))
            mem.append(DialogueTurn(round_number=i, role="interviewee", content=f"A{i}"))

        last_3 = mem.get_last_n_rounds(3)
        rounds = {t.round_number for t in last_3}
        assert rounds == {3, 4, 5}

    def test_get_range(self) -> None:
        mem = EpisodicMemory()
        for i in range(1, 11):
            mem.append(DialogueTurn(round_number=i, role="interviewer", content=f"Q{i}"))

        result = mem.get_range(3, 6)
        assert len(result) == 3
        assert all(3 <= t.round_number < 6 for t in result)

    def test_to_messages(self) -> None:
        mem = EpisodicMemory()
        mem.append(DialogueTurn(round_number=1, role="interviewer", content="Hi"))
        mem.append(DialogueTurn(round_number=1, role="interviewee", content="Hello"))

        msgs = mem.to_messages()
        assert msgs[0] == {"role": "assistant", "content": "Hi"}
        assert msgs[1] == {"role": "user", "content": "Hello"}

    def test_empty(self) -> None:
        mem = EpisodicMemory()
        assert mem.total_rounds == 0
        assert mem.get_last_n_rounds(5) == []

    def test_immutability(self) -> None:
        """确保外部无法修改内部列表"""
        mem = EpisodicMemory()
        mem.append(DialogueTurn(round_number=1, role="interviewer", content="Q"))
        turns = mem.turns
        turns.clear()
        assert len(mem.turns) == 1  # 内部不受影响


# ============ SemanticMemory ============


class TestSemanticMemory:
    def test_add_and_query_facts(self) -> None:
        mem = SemanticMemory()
        mem.add_fact(SemanticFact(
            content="3年Python经验", category="skill", round_number=2
        ))
        mem.add_fact(SemanticFact(
            content="带过5人团队", category="team", round_number=5
        ))

        assert len(mem.facts) == 2
        assert len(mem.get_facts_by_category("skill")) == 1
        assert len(mem.get_facts_by_round(5)) == 1

    def test_contradiction_recording(self) -> None:
        mem = SemanticMemory()
        fact_a = SemanticFact(content="3年Python", category="skill", round_number=2)
        fact_b = SemanticFact(content="5年Python", category="skill", round_number=8)
        mem.add_fact(fact_a)
        mem.add_fact(fact_b)

        contradiction = Contradiction(
            fact_a=fact_a,
            fact_b=fact_b,
            description="Python经验年限矛盾：第2轮说3年，第8轮说5年",
            severity="high",
            detected_at_round=8,
        )
        mem.add_contradiction(contradiction)

        assert len(mem.contradictions) == 1
        assert "矛盾" in mem.contradictions[0].description

    def test_format_fact_table(self) -> None:
        mem = SemanticMemory()
        mem.add_fact(SemanticFact(
            content="Redis经验", category="skill", round_number=3,
            confidence=Confidence.HIGH,
        ))
        mem.add_fact(SemanticFact(
            content="管理10人", category="team", round_number=5,
            confidence=Confidence.LOW,
        ))

        table = mem.format_fact_table()
        assert "已知事实" in table
        assert "Redis" in table
        assert "skill" in table

    def test_empty_fact_table(self) -> None:
        mem = SemanticMemory()
        assert "暂无" in mem.format_fact_table()

    def test_fact_table_with_contradictions(self) -> None:
        mem = SemanticMemory()
        fact_a = SemanticFact(content="A", category="skill", round_number=1)
        fact_b = SemanticFact(content="B", category="skill", round_number=5)
        mem.add_fact(fact_a)
        mem.add_fact(fact_b)
        mem.add_contradiction(Contradiction(
            fact_a=fact_a, fact_b=fact_b,
            description="矛盾内容", detected_at_round=5,
        ))

        table = mem.format_fact_table()
        assert "矛盾" in table


# ============ PrivilegedMemory ============


class TestPrivilegedMemory:
    def test_information_asymmetry(self) -> None:
        """核心测试：信息不对称"""
        mem = PrivilegedMemory()
        mem.set("original_resume", "原始简历内容")
        mem.set("modified_resume", "修改后简历内容")
        mem.set("job_description", "JD内容")

        # 面试官：能看修改后简历+JD，不能看原始简历
        interviewer_view = mem.get_visible(INTERVIEWER_POLICY)
        assert "modified_resume" in interviewer_view
        assert "job_description" in interviewer_view
        assert "original_resume" not in interviewer_view

        # 面试者：只能看修改后简历
        interviewee_view = mem.get_visible(INTERVIEWEE_POLICY)
        assert "modified_resume" in interviewee_view
        assert "job_description" not in interviewee_view
        assert "original_resume" not in interviewee_view

        # 评估官：全都能看
        evaluator_view = mem.get_visible(EVALUATOR_POLICY)
        assert "original_resume" in evaluator_view
        assert "modified_resume" in evaluator_view
        assert "job_description" in evaluator_view

    def test_empty_fields_not_returned(self) -> None:
        mem = PrivilegedMemory()
        # 什么都没设置
        view = mem.get_visible(EVALUATOR_POLICY)
        assert len(view) == 0

    def test_invalid_key(self) -> None:
        mem = PrivilegedMemory()
        with pytest.raises(KeyError):
            mem.set("invalid_key", "value")


# ============ ContextWindowManager ============


class TestContextWindowManager:
    def test_compose_within_budget(self) -> None:
        mgr = ContextWindowManager(token_budget=12000, verbatim_window=6)
        messages = mgr.compose_messages(
            system_prompt="你是面试官",
            privileged_info={},
            fact_table="",
            conversation_summary="",
            recent_turns=[
                {"role": "assistant", "content": "问题1"},
                {"role": "user", "content": "回答1"},
            ],
        )

        assert messages[0]["role"] == "system"
        assert len(messages) >= 3  # system + 2 turns

    def test_trimming_when_over_budget(self) -> None:
        mgr = ContextWindowManager(token_budget=200, verbatim_window=6)

        # 创建大量对话
        turns = [
            {"role": "user" if i % 2 else "assistant", "content": f"很长的内容 " * 50}
            for i in range(20)
        ]

        messages = mgr.compose_messages(
            system_prompt="系统",
            privileged_info={},
            fact_table="",
            conversation_summary="摘要内容",
            recent_turns=turns,
        )

        # 应该被裁剪
        assert len(messages) < len(turns) + 2

    def test_should_summarize(self) -> None:
        mgr = ContextWindowManager(summary_batch=5)
        assert mgr.should_summarize(total_rounds=5, last_summary_round=0)
        assert not mgr.should_summarize(total_rounds=3, last_summary_round=0)
        assert mgr.should_summarize(total_rounds=10, last_summary_round=5)

    def test_count_tokens(self) -> None:
        """token 计数应该返回正数"""
        result = count_tokens("你好世界 hello world")
        assert result > 0


# ============ WorkingMemory ============


class TestWorkingMemory:
    def _make_working_memory(self) -> WorkingMemory:
        return WorkingMemory(
            episodic=EpisodicMemory(),
            semantic=SemanticMemory(),
            privileged=PrivilegedMemory(),
            context_manager=ContextWindowManager(),
        )

    def test_compose_for_agent(self) -> None:
        wm = self._make_working_memory()
        wm.episodic.append(
            DialogueTurn(round_number=1, role="interviewer", content="Hi")
        )
        wm.privileged.set("modified_resume", "简历内容")

        messages = wm.compose_for_agent(
            system_prompt="你是面试官",
            policy=INTERVIEWER_POLICY,
        )

        # 系统消息应包含简历
        system_msg = messages[0]["content"]
        assert "简历" in system_msg

    def test_needs_summary(self) -> None:
        wm = self._make_working_memory()
        assert not wm.needs_summary()  # 没有对话

        # 添加足够轮次
        for i in range(1, 6):
            wm.episodic.append(
                DialogueTurn(round_number=i, role="interviewer", content=f"Q{i}")
            )
        assert wm.needs_summary()  # 5轮了，该摘要了

    def test_summary_update(self) -> None:
        wm = self._make_working_memory()
        wm.conversation_summary = "面试进行了5轮"
        wm.last_summary_round = 5
        assert wm.conversation_summary == "面试进行了5轮"
        assert wm.last_summary_round == 5
