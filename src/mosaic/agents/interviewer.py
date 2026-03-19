"""
Interviewer Agent — 领域专家面试官。
职责：自适应深度提问 + 技术深挖 + 追问策略动态调整。
订阅矛盾事件和评估事件，根据候选人表现调整策略。
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from mosaic.core.agent import INTERVIEWER_POLICY, BaseAgent
from mosaic.core.events import Event, EventType
from mosaic.core.memory import WorkingMemory
from mosaic.llm.client import LLMClient

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class InterviewStrategy(Enum):
    """面试追问策略 — 根据候选人表现自适应调整"""
    DEEP_DIVE = "deep_dive"   # 候选人答得好 → 继续深入更高难度
    GUIDED = "guided"         # 答得一般 → 给提示引导
    PIVOT = "pivot"           # 答不上来 → 温和换话题


STRATEGY_LABELS = {
    InterviewStrategy.DEEP_DIVE: "🟢 深入模式 — 候选人表现优秀，继续深入更高难度",
    InterviewStrategy.GUIDED: "🟡 引导模式 — 候选人表现一般，给提示引导",
    InterviewStrategy.PIVOT: "🔴 转向模式 — 候选人有困难，温和换话题",
}


INTERVIEWER_SYSTEM_FALLBACK = """\
你是目标岗位领域的技术大佬，正在作为面试官进行一场深度技术面试。

## 你的身份

你是该技术领域的顶级专家，能从候选人简历中的关键词出发，追问到技术纵深。

## JD 驱动提问原则

你必须让面试紧密围绕目标岗位的 JD 要求展开：
1. 从 JD 中提取 3-5 个核心技能要求作为考察主线
2. 每个问题都应至少关联一个 JD 核心要求
3. 项目深挖时，引导候选人往 JD 相关方向展开
4. 如果候选人的简历经历与 JD 要求有差距，通过提问探测其迁移能力

## 自适应追问策略

当前策略: {current_strategy}

🟢 深入模式: 答得好 → 沿同一话题继续深入更高难度
🟡 引导模式: 答得一般 → 换个角度或给提示
🔴 转向模式: 答不上来 → 换一个新话题

## 提问覆盖追踪

你需要覆盖以下维度：
{coverage_status}

当前是第 {current_round} 轮，共 {total_rounds} 轮。
{contradiction_alert}

## 输出规则（必须严格遵守）

1. **只输出一个问题，不要输出任何其他内容**
2. **禁止评价候选人的上一轮回答**（不要说"回答得很好"、"不错"之类的话）
3. **禁止总结或复述候选人说过的内容**
4. 直接提出下一个问题，简洁有力，不超过 3 句话
5. 问题要具体，指向明确的技术点
"""


class InterviewerAgent(BaseAgent):
    """
    领域专家面试官 Agent。

    事件订阅：
    - CONTRADICTION_FOUND → 标记矛盾，下次提问时追问
    - TURN_EVALUATED → 根据评分调整追问策略

    事件发布：
    - QUESTION_POSED — 提出新问题
    """

    COVERAGE_DIMENSIONS = [
        "项目经历深挖",
        "技术纵深考察",
        "系统设计思维",
        "团队协作与沟通",
        "技术视野与学习力",
    ]

    def __init__(self, llm: LLMClient, working_memory: WorkingMemory) -> None:
        super().__init__(name="interviewer", memory_policy=INTERVIEWER_POLICY)
        self.llm = llm
        self.working_memory = working_memory
        self._coverage: dict[str, bool] = {
            dim: False for dim in self.COVERAGE_DIMENSIONS
        }
        self._pending_contradictions: list[dict[str, Any]] = []
        self._current_strategy: InterviewStrategy = InterviewStrategy.GUIDED
        self._score_history: list[float] = []  # 历史各轮平均分
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _register_subscriptions(self) -> None:
        self._subscribe(EventType.CONTRADICTION_FOUND, self._on_contradiction)
        self._subscribe(EventType.TURN_EVALUATED, self._on_turn_evaluated)

    async def _on_contradiction(self, event: Event) -> None:
        """收到矛盾通知 → 记下来，下次提问时追问"""
        self._pending_contradictions.append(event.data)
        logger.info(
            f"Interviewer noted contradiction: {event.data.get('description')}"
        )

    async def _on_turn_evaluated(self, event: Event) -> None:
        """收到评估结果 → 综合最近 N 轮趋势调整追问策略。

        策略：滑动窗口加权平均（最近 3 轮，越近权重越高）。
        避免单轮偶然低分直接切换到 PIVOT 的问题。
        """
        scores = event.data.get("scores", {})
        if not scores:
            return

        score_values = [v for v in scores.values() if isinstance(v, (int, float))]
        if not score_values:
            return

        current_avg = sum(score_values) / len(score_values)
        self._score_history.append(current_avg)

        # 滑动窗口加权平均：最近 3 轮，权重 1:2:3（越近越重）
        window = self._score_history[-3:]
        weights = list(range(1, len(window) + 1))  # [1], [1,2], [1,2,3]
        weighted_avg = sum(s * w for s, w in zip(window, weights)) / sum(weights)

        if weighted_avg >= 3.8:
            self._current_strategy = InterviewStrategy.DEEP_DIVE
        elif weighted_avg >= 2.5:
            self._current_strategy = InterviewStrategy.GUIDED
        else:
            self._current_strategy = InterviewStrategy.PIVOT

        logger.info(
            f"Interview strategy adjusted to {self._current_strategy.value} "
            f"(current={current_avg:.1f}, weighted_avg={weighted_avg:.1f}, "
            f"window={[f'{s:.1f}' for s in window]})"
        )

    @property
    def current_strategy(self) -> InterviewStrategy:
        """当前追问策略"""
        return self._current_strategy

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

        # 当前策略标签
        strategy_label = STRATEGY_LABELS.get(
            self._current_strategy,
            str(self._current_strategy.value),
        )

        # 构建系统提示
        try:
            template = self._jinja_env.get_template("interviewer_system.j2")
            system_prompt = template.render(
                current_round=current_round,
                total_rounds=total_rounds,
                coverage_status=coverage_status,
                contradiction_alert=contradiction_alert,
                current_strategy=strategy_label,
            )
        except Exception:
            system_prompt = INTERVIEWER_SYSTEM_FALLBACK.format(
                current_round=current_round,
                total_rounds=total_rounds,
                coverage_status=coverage_status,
                contradiction_alert=contradiction_alert,
                current_strategy=strategy_label,
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
            max_tokens=200,
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
