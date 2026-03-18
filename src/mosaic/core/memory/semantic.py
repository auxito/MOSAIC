"""
语义记忆 — 从对话中抽取的结构化事实 + 矛盾检测。
解决人设一致性问题：面试者回答前检查已有事实。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Confidence(Enum):
    """事实置信度"""

    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()


@dataclass
class SemanticFact:
    """从对话中抽取的单条事实"""

    content: str  # 事实内容，如 "声称有3年Python经验"
    category: str  # 分类：skill/experience/project/education/team/other
    round_number: int  # 来源轮次
    confidence: Confidence = Confidence.MEDIUM
    timestamp: float = field(default_factory=time.time)
    fact_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fact_id:
            self.fact_id = f"fact_{self.round_number}_{id(self) % 10000:04d}"


@dataclass
class Contradiction:
    """检测到的矛盾"""

    fact_a: SemanticFact
    fact_b: SemanticFact
    description: str  # 矛盾描述
    severity: str = "medium"  # low / medium / high
    detected_at_round: int = 0
    timestamp: float = field(default_factory=time.time)


class SemanticMemory:
    """
    语义记忆存储。
    - 存储结构化事实
    - 提供按类别/轮次查询
    - 矛盾记录
    """

    def __init__(self) -> None:
        self._facts: list[SemanticFact] = []
        self._contradictions: list[Contradiction] = []

    def add_fact(self, fact: SemanticFact) -> None:
        """添加一条事实"""
        self._facts.append(fact)

    def add_facts(self, facts: list[SemanticFact]) -> None:
        """批量添加事实"""
        self._facts.extend(facts)

    def add_contradiction(self, contradiction: Contradiction) -> None:
        """记录矛盾"""
        self._contradictions.append(contradiction)

    @property
    def facts(self) -> list[SemanticFact]:
        return list(self._facts)

    @property
    def contradictions(self) -> list[Contradiction]:
        return list(self._contradictions)

    def get_facts_by_category(self, category: str) -> list[SemanticFact]:
        """按类别查询事实"""
        return [f for f in self._facts if f.category == category]

    def get_facts_by_round(self, round_number: int) -> list[SemanticFact]:
        """获取某一轮抽取的事实"""
        return [f for f in self._facts if f.round_number == round_number]

    def format_fact_table(self) -> str:
        """格式化为简洁的事实表（用于注入 LLM 上下文）"""
        if not self._facts:
            return "暂无抽取的事实。"

        lines = ["## 已知事实"]
        categories: dict[str, list[SemanticFact]] = {}
        for f in self._facts:
            categories.setdefault(f.category, []).append(f)

        for cat, facts in categories.items():
            lines.append(f"\n### {cat}")
            for f in facts:
                conf = f.confidence.name.lower()
                lines.append(f"- [{conf}] (第{f.round_number}轮) {f.content}")

        if self._contradictions:
            lines.append("\n### ⚠️ 矛盾")
            for c in self._contradictions:
                lines.append(
                    f"- [第{c.detected_at_round}轮检出] {c.description}"
                )

        return "\n".join(lines)

    def clear(self) -> None:
        """重置"""
        self._facts.clear()
        self._contradictions.clear()
