"""
Interviewer Agent — 面试官。
职责：自适应提问 + 追问 + 收尾。
订阅矛盾事件，发现矛盾时改变策略追问。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from mosaic.core.agent import INTERVIEWER_POLICY, BaseAgent
from mosaic.core.events import Event, EventType
from mosaic.core.memory import WorkingMemory
from mosaic.llm.client import LLMClient

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

INTERVIEWER_SYSTEM_FALLBACK = """\
你是一位经验丰富的技术面试官，正在进行一场模拟面试。

## 你的面试策略

1. **开场**: 自我介绍，简单寒暄，让候选人放松
2. **项目深挖**: 针对简历中的项目经历，追问技术细节和个人贡献
3. **技术考察**: 基于 JD 要求，考察核心技术能力
4. **行为面试**: 了解团队协作、解决问题的方式
5. **收尾**: 让候选人提问，总结面试

## 提问原则

- 由浅入深，先开放后具体
- 对模糊回答进行追问（"能具体说说吗？""给个例子？"）
- 关注数字和细节（"团队多大？""提升了多少？"）
- 每次只问一个问题，等待回答后再继续
- 如果发现矛盾，用温和的方式追问核实

## 提问覆盖追踪

你需要覆盖以下维度，用 [已覆盖] 标记已提问的维度：
- 项目经历深挖
- 技术能力考察
- 系统设计思维
- 团队协作能力
- 学习成长能力

当前是第 {current_round} 轮，共 {total_rounds} 轮。
{coverage_status}
{contradiction_alert}
"""


class InterviewerAgent(BaseAgent):
    """
    面试官 Agent。

    事件订阅：
    - CONTRADICTION_FOUND → 标记矛盾，下次提问时追问

    事件发布：
    - QUESTION_POSED — 提出新问题
    """

    COVERAGE_DIMENSIONS = [
        "项目经历深挖",
        "技术能力考察",
        "系统设计思维",
        "团队协作能力",
        "学习成长能力",
    ]

    def __init__(self, llm: LLMClient, working_memory: WorkingMemory) -> None:
        super().__init__(name="interviewer", memory_policy=INTERVIEWER_POLICY)
        self.llm = llm
        self.working_memory = working_memory
        self._coverage: dict[str, bool] = {
            dim: False for dim in self.COVERAGE_DIMENSIONS
        }
        self._pending_contradictions: list[dict[str, Any]] = []
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _register_subscriptions(self) -> None:
        self._subscribe(EventType.CONTRADICTION_FOUND, self._on_contradiction)

    async def _on_contradiction(self, event: Event) -> None:
        """收到矛盾通知 → 记下来，下次提问时追问"""
        self._pending_contradictions.append(event.data)
        logger.info(
            f"Interviewer noted contradiction: {event.data.get('description')}"
        )

    async def ask(
        self,
        current_round: int,
        total_rounds: int,
        last_answer: str | None = None,
    ) -> str:
        """生成面试问题"""

        # 构建矛盾提醒
        contradiction_alert = ""
        if self._pending_contradictions:
            alerts = []
            for c in self._pending_contradictions:
                alerts.append(f"⚠️ 候选人可能存在矛盾: {c.get('description', '')}")
            contradiction_alert = (
                "\n## 矛盾提醒\n"
                + "\n".join(alerts)
                + "\n请在接下来的提问中，用温和但直接的方式追问这些矛盾点。"
            )
            # 消费掉
            self._pending_contradictions.clear()

        # 覆盖度状态
        coverage_lines = []
        for dim, covered in self._coverage.items():
            status = "[已覆盖]" if covered else "[待覆盖]"
            coverage_lines.append(f"  {status} {dim}")
        coverage_status = "\n".join(coverage_lines)

        # 构建系统提示
        try:
            template = self._jinja_env.get_template("interviewer_system.j2")
            system_prompt = template.render(
                current_round=current_round,
                total_rounds=total_rounds,
                coverage_status=coverage_status,
                contradiction_alert=contradiction_alert,
            )
        except Exception:
            system_prompt = INTERVIEWER_SYSTEM_FALLBACK.format(
                current_round=current_round,
                total_rounds=total_rounds,
                coverage_status=coverage_status,
                contradiction_alert=contradiction_alert,
            )

        # 面试官视角：自己是 assistant，候选人是 user
        messages = self.working_memory.compose_for_agent(
            system_prompt=system_prompt,
            policy=self.memory_policy,
            current_input=last_answer,
            role_map={"interviewer": "assistant", "interviewee": "user"},
        )

        question = await self.llm.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )

        await self._emit(Event(
            type=EventType.QUESTION_POSED,
            source=self.name,
            data={"question": question, "round_number": current_round},
        ))

        return question

    def mark_coverage(self, dimension: str) -> None:
        """标记某维度已覆盖"""
        if dimension in self._coverage:
            self._coverage[dimension] = True

    async def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Agent 接口"""
        question = await self.ask(
            current_round=context.get("current_round", 1),
            total_rounds=context.get("total_rounds", 10),
            last_answer=context.get("last_answer"),
        )
        return {"question": question}
