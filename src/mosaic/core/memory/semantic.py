"""
语义记忆 — 从对话中抽取的结构化事实 + 矛盾检测。
解决人设一致性问题：面试者回答前检查已有事实。

compact() 机制：
- 去重：内容相似度高的事实只保留置信度最高的
- 升级：同一事实被多次提及 → 置信度升为 HIGH
- 上限：超过 MAX_FACTS 时，丢弃低置信度旧事实
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class Confidence(Enum):
    """事实置信度"""

    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()


# 置信度排序：HIGH > MEDIUM > LOW
_CONFIDENCE_RANK = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}


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
    mention_count: int = 1  # 被提及次数（compact 时累计）

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
    - compact() 自动去重压缩
    """

    MAX_FACTS = 30  # 事实表上限（超过后压缩）
    SIMILARITY_THRESHOLD = 0.6  # 内容重叠度阈值（用于去重）

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

    def compact(self) -> int:
        """压缩事实表 — 去重、合并、控制上限。

        策略：
        1. 同类别内，内容高度重叠的事实合并（保留置信度高的，累计提及次数）
        2. 多次提及的事实置信度升为 HIGH
        3. 超过上限时，按优先级排序丢弃末尾：
           - 优先保留：HIGH > MEDIUM > LOW
           - 同级内保留：最近提及的 > 最早提及的

        Returns:
            压缩掉的事实数量
        """
        before = len(self._facts)
        if before <= self.MAX_FACTS // 2:
            return 0  # 还远没到上限，不需要压缩

        # 按类别分组
        by_category: dict[str, list[SemanticFact]] = {}
        for f in self._facts:
            by_category.setdefault(f.category, []).append(f)

        merged: list[SemanticFact] = []
        for cat, facts in by_category.items():
            cat_merged = self._merge_similar(facts)
            merged.extend(cat_merged)

        # 多次提及 → 升级置信度
        for f in merged:
            if f.mention_count >= 2 and f.confidence != Confidence.HIGH:
                f.confidence = Confidence.HIGH

        # 超过上限时裁剪
        if len(merged) > self.MAX_FACTS:
            merged.sort(
                key=lambda f: (
                    _CONFIDENCE_RANK.get(f.confidence, 0),
                    f.round_number,
                ),
                reverse=True,
            )
            merged = merged[: self.MAX_FACTS]

        self._facts = merged
        removed = before - len(self._facts)
        if removed > 0:
            logger.info(f"Semantic memory compacted: {before} → {len(self._facts)} facts (-{removed})")
        return removed

    @staticmethod
    def _merge_similar(facts: list[SemanticFact]) -> list[SemanticFact]:
        """同类别内合并高度相似的事实。

        使用字符级 Jaccard 相似度（基于字符 bigram），
        不依赖外部库，对中英文混合内容都有效。
        """
        if len(facts) <= 1:
            return list(facts)

        def bigram_set(text: str) -> set[str]:
            t = text.strip().lower()
            return {t[i : i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else {t}

        def similarity(a: str, b: str) -> float:
            sa, sb = bigram_set(a), bigram_set(b)
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        # 贪心合并
        used = [False] * len(facts)
        result: list[SemanticFact] = []

        for i, fi in enumerate(facts):
            if used[i]:
                continue
            # fi 为基准，找所有与它相似的
            best = fi
            count = fi.mention_count
            for j in range(i + 1, len(facts)):
                if used[j]:
                    continue
                if similarity(fi.content, facts[j].content) >= SemanticMemory.SIMILARITY_THRESHOLD:
                    used[j] = True
                    count += facts[j].mention_count
                    # 保留置信度更高的
                    if _CONFIDENCE_RANK.get(facts[j].confidence, 0) > _CONFIDENCE_RANK.get(best.confidence, 0):
                        best = facts[j]
                    # 保留轮次更近的
                    elif facts[j].round_number > best.round_number:
                        best = facts[j]

            best.mention_count = count
            result.append(best)

        return result

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
                # 多次提及的事实加标记，让 LLM 更重视
                repeat = f" (×{f.mention_count})" if f.mention_count > 1 else ""
                lines.append(f"- [{conf}]{repeat} (第{f.round_number}轮) {f.content}")

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
