"""
工作流定义 — Phase 枚举 + WorkflowConfig。
不同模式 = 不同阶段序列，不需要改 Agent 代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Phase(Enum):
    """面试工作流阶段"""

    INIT = auto()
    RESUME_INPUT = auto()
    RESUME_MODIFY = auto()
    INTERVIEW_LOOP = auto()
    EVALUATION = auto()
    REPORT = auto()
    COMPLETE = auto()


@dataclass
class WorkflowConfig:
    """工作流配置 — 定义阶段序列和参数"""

    name: str
    phases: list[Phase]
    description: str = ""
    interview_rounds: int = 10
    resume_styles: list[str] = field(
        default_factory=lambda: ["rigorous", "embellished", "wild"]
    )
    extra: dict = field(default_factory=dict)

    def has_phase(self, phase: Phase) -> bool:
        return phase in self.phases

    def next_phase(self, current: Phase) -> Phase | None:
        """获取下一个阶段，如果是最后阶段返回 None"""
        try:
            idx = self.phases.index(current)
            if idx + 1 < len(self.phases):
                return self.phases[idx + 1]
        except ValueError:
            pass
        return None


# ============ 预设工作流 ============

RESUME_ONLY = WorkflowConfig(
    name="resume_only",
    description="仅简历修改，不进行面试",
    phases=[
        Phase.INIT,
        Phase.RESUME_INPUT,
        Phase.RESUME_MODIFY,
        Phase.REPORT,
        Phase.COMPLETE,
    ],
    interview_rounds=0,
)

FULL_INTERVIEW = WorkflowConfig(
    name="full_interview",
    description="完整流程：简历修改 → 模拟面试 → 评估报告",
    phases=[
        Phase.INIT,
        Phase.RESUME_INPUT,
        Phase.RESUME_MODIFY,
        Phase.INTERVIEW_LOOP,
        Phase.EVALUATION,
        Phase.REPORT,
        Phase.COMPLETE,
    ],
    interview_rounds=10,
)

HUMAN_PRACTICE = WorkflowConfig(
    name="human_practice",
    description="真人练习模式：跳过简历修改，直接面试",
    phases=[
        Phase.INIT,
        Phase.RESUME_INPUT,
        Phase.INTERVIEW_LOOP,
        Phase.EVALUATION,
        Phase.REPORT,
        Phase.COMPLETE,
    ],
    interview_rounds=10,
)

EVAL_ONLY = WorkflowConfig(
    name="eval_only",
    description="仅评估模式：面试 + 评估，不修改简历",
    phases=[
        Phase.INIT,
        Phase.RESUME_INPUT,
        Phase.INTERVIEW_LOOP,
        Phase.EVALUATION,
        Phase.REPORT,
        Phase.COMPLETE,
    ],
    interview_rounds=10,
)

# 工作流注册表
WORKFLOWS: dict[str, WorkflowConfig] = {
    "resume_only": RESUME_ONLY,
    "full_interview": FULL_INTERVIEW,
    "human_practice": HUMAN_PRACTICE,
    "eval_only": EVAL_ONLY,
}
