"""
MemoryManager Agent — 基础设施 Agent。
职责：事实抽取、矛盾检测、渐进式摘要。
不参与面试对话，在幕后维护记忆系统。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mosaic.core.agent import BaseAgent, MemoryPolicy
from mosaic.core.events import Event, EventType
from mosaic.core.memory import (
    Confidence,
    Contradiction,
    SemanticFact,
    SemanticMemory,
    WorkingMemory,
)
from mosaic.llm.client import LLMClient

logger = logging.getLogger(__name__)

FACT_EXTRACTION_PROMPT = """\
你是一个信息抽取专家。从以下面试对话轮次中，抽取候选人声称的事实。

对话内容：
问题: {question}
回答: {answer}

当前轮次: 第{round_number}轮

请以 JSON 格式返回抽取的事实列表：
{{
  "facts": [
    {{
      "content": "声称的事实内容",
      "category": "skill|experience|project|education|team|achievement|other",
      "confidence": "high|medium|low"
    }}
  ]
}}

注意：
- 只抽取候选人明确声称的事实，不要推断
- category 必须是指定的类别之一
- 对模糊的说法标记低置信度
"""

CONTRADICTION_CHECK_PROMPT = """\
你是一个逻辑一致性检查专家。检查新事实是否与已有事实存在矛盾。

已有事实：
{existing_facts}

新事实：
{new_facts}

请以 JSON 格式返回检测结果：
{{
  "contradictions": [
    {{
      "new_fact": "矛盾的新事实内容",
      "existing_fact": "矛盾的已有事实内容",
      "description": "矛盾描述",
      "severity": "low|medium|high"
    }}
  ]
}}

如果没有矛盾，返回空列表：{{"contradictions": []}}
注意：只标记真正的逻辑矛盾，不要标记补充信息。
"""

SUMMARY_PROMPT = """\
你是一个对话摘要专家。请对以下面试对话进行渐进式摘要。

{previous_summary}

新增对话：
{new_turns}

当前总轮次: {total_rounds}

## 摘要要求（按优先级）

### 必须保留的信息（绝不能丢弃）
1. **所有具体数字**：年限、百分比、QPS、延迟、团队人数、项目周期等
2. **所有时间节点**：工作起止时间、项目时间线
3. **技术栈和工具名称**：完整保留，不要用"等技术"省略
4. **候选人的关键判断和决策**：为什么选这个方案、权衡了什么

### 应保留的信息
5. 项目中的具体贡献和角色
6. 技术方案的核心思路（不需要完整细节）
7. 候选人表现出的亮点或弱点

### 可以压缩的信息
8. 面试官的提问细节（保留话题方向即可）
9. 候选人的语气词、重复表述
10. 已经在事实表中记录过的基本事实（避免重复）

## 格式要求
- 按话题/维度组织，不要按轮次流水账
- 字数上限: {max_length} 字
- 如果有之前的摘要，将新信息整合进去，而不是追加
"""


class MemoryManagerAgent(BaseAgent):
    """
    记忆管理 Agent。

    事件订阅：
    - ANSWER_GIVEN → 抽取事实 + 检测矛盾
    - 定期生成渐进式摘要

    事件发布：
    - FACT_EXTRACTED — 新事实
    - CONTRADICTION_FOUND — 发现矛盾
    - SUMMARY_GENERATED — 摘要更新
    """

    def __init__(self, llm: LLMClient, working_memory: WorkingMemory) -> None:
        super().__init__(name="memory_manager", memory_policy=MemoryPolicy())
        self.llm = llm
        self.working_memory = working_memory

    def _register_subscriptions(self) -> None:
        self._subscribe(EventType.ANSWER_GIVEN, self._on_answer_given)

    async def _on_answer_given(self, event: Event) -> None:
        """收到回答后：抽取事实 → 检测矛盾 → 可能生成摘要"""
        question = event.data.get("question", "")
        answer = event.data.get("answer", "")
        round_number = event.data.get("round_number", 0)

        if not answer:
            return

        # 1. 抽取事实
        new_facts = await self._extract_facts(question, answer, round_number)

        if new_facts:
            # 2. 检测矛盾
            await self._check_contradictions(new_facts, round_number)

            # 3. 存入语义记忆
            self.working_memory.semantic.add_facts(new_facts)

            # 3.5 压缩事实表（去重 + 合并 + 上限控制）
            self.working_memory.semantic.compact()

            await self._emit(Event(
                type=EventType.FACT_EXTRACTED,
                source=self.name,
                data={
                    "round_number": round_number,
                    "facts": [f.content for f in new_facts],
                },
            ))

        # 4. 检查是否需要摘要
        if self.working_memory.needs_summary():
            await self._generate_summary()

    async def _extract_facts(
        self, question: str, answer: str, round_number: int
    ) -> list[SemanticFact]:
        """从回答中抽取结构化事实"""
        prompt = FACT_EXTRACTION_PROMPT.format(
            question=question, answer=answer, round_number=round_number
        )

        try:
            response = await self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            data = json.loads(response)
            facts = []
            for item in data.get("facts", []):
                conf_map = {
                    "high": Confidence.HIGH,
                    "medium": Confidence.MEDIUM,
                    "low": Confidence.LOW,
                }
                facts.append(SemanticFact(
                    content=item["content"],
                    category=item.get("category", "other"),
                    round_number=round_number,
                    confidence=conf_map.get(
                        item.get("confidence", "medium"), Confidence.MEDIUM
                    ),
                ))
            logger.info(f"Extracted {len(facts)} facts from round {round_number}")
            return facts
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    async def _check_contradictions(
        self, new_facts: list[SemanticFact], round_number: int
    ) -> None:
        """检查新事实与已有事实的矛盾"""
        existing = self.working_memory.semantic.facts
        if not existing:
            return

        existing_text = "\n".join(
            f"- (第{f.round_number}轮, {f.category}) {f.content}" for f in existing
        )
        new_text = "\n".join(f"- {f.content}" for f in new_facts)

        prompt = CONTRADICTION_CHECK_PROMPT.format(
            existing_facts=existing_text, new_facts=new_text
        )

        try:
            response = await self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            data = json.loads(response)
            for item in data.get("contradictions", []):
                contradiction = Contradiction(
                    fact_a=SemanticFact(
                        content=item["existing_fact"],
                        category="unknown",
                        round_number=0,
                    ),
                    fact_b=SemanticFact(
                        content=item["new_fact"],
                        category="unknown",
                        round_number=round_number,
                    ),
                    description=item["description"],
                    severity=item.get("severity", "medium"),
                    detected_at_round=round_number,
                )
                self.working_memory.semantic.add_contradiction(contradiction)

                logger.warning(
                    f"Contradiction found at round {round_number}: "
                    f"{item['description']}"
                )

                await self._emit(Event(
                    type=EventType.CONTRADICTION_FOUND,
                    source=self.name,
                    data={
                        "round_number": round_number,
                        "description": item["description"],
                        "severity": item.get("severity", "medium"),
                    },
                ))
        except Exception as e:
            logger.error(f"Contradiction check failed: {e}")

    async def _generate_summary(self) -> None:
        """生成渐进式摘要 — 动态字数上限，优先保留关键数据。"""
        episodic = self.working_memory.episodic
        last_round = self.working_memory.last_summary_round
        new_turns = episodic.get_range(last_round + 1)

        if not new_turns:
            return

        new_turns_text = "\n".join(
            f"[第{t.round_number}轮 {t.role}]: {t.content}" for t in new_turns
        )

        prev = self.working_memory.conversation_summary
        previous_section = f"之前的摘要：\n{prev}" if prev else "（这是第一段摘要）"

        # 动态字数上限：随轮次增长，但有上限
        total_rounds = episodic.total_rounds
        max_length = min(300 + total_rounds * 40, 800)

        prompt = SUMMARY_PROMPT.format(
            previous_summary=previous_section,
            new_turns=new_turns_text,
            total_rounds=total_rounds,
            max_length=max_length,
        )

        try:
            summary = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            self.working_memory.conversation_summary = summary
            self.working_memory.last_summary_round = episodic.total_rounds

            logger.info(
                f"Summary updated, covering rounds 1-{episodic.total_rounds}"
            )

            await self._emit(Event(
                type=EventType.SUMMARY_GENERATED,
                source=self.name,
                data={"summary": summary},
            ))
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")

    async def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """手动触发模式（非事件驱动时使用）"""
        action = context.get("action")
        if action == "summarize":
            await self._generate_summary()
        return {"status": "ok"}
