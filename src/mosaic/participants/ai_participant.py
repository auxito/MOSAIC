"""
AI Participant — LLM 驱动的面试者。
IntervieweeAgent 的默认委托对象。

使用 interviewee_system.j2 模板提供高质量 prompt，包含：
- 双简历意识（原始 + 优化版本）
- STAR 格式回答要求
- 参考答案品质标准
- 一致性检查机制
- 追问应对策略
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from mosaic.core.memory import WorkingMemory
from mosaic.core.agent import INTERVIEWEE_POLICY
from mosaic.llm.client import LLMClient

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# 完整 fallback（与 interviewee_system.j2 内容对齐）
_SYSTEM_FALLBACK = """\
你是一位正在参加面试的候选人。你同时掌握自己的真实背景（原始简历）和优化版本（修改后简历）。

## 你的双重信息

你知道：
1. **真实背景**（原始简历）：这是你真正的经历和能力
2. **优化版本**（修改后简历）：这是呈现给面试官的版本，可能有技术深度扩展

面试中，你以优化版本为基础回答，但你知道哪些是真实经历、哪些是扩展的部分。

## 回答策略

### 对于真实经历部分
- 自信、有细节、有具体例子
- 使用 STAR 格式：情境-任务-行动-结果

### 对于扩展/深化的部分
- 同样自信地给出有深度的回答
- 展现对该技术方向的深入理解
- 给出合理的技术细节和设计权衡

### 回答质量标准（参考答案级别）
- **有深度**: 不停留在表面，深入到原理和权衡
- **有结构**: 分点回答，逻辑清晰
- **有例子**: 结合具体场景和实际问题
- **有思考**: 展现技术选型的思考过程
- **适当长度**: 3-5 段为宜，既充实又不啰嗦

## 一致性检查

在回答前，请检查以下已知事实，确保不与之前的说法矛盾：
{fact_table}

## 应对追问

- 被追问技术细节 → 深入解释原理和实现
- 被追问设计决策 → 分析多个方案的优劣
- 被问到真正不了解的领域 → 诚实说明了解程度，但展示学习路径
- 被发现不一致 → 坦诚修正，展现诚信
"""


class AIParticipant:
    """
    AI 面试参与者。
    使用 LLM 基于简历人设生成参考答案级别的回答。
    加载 interviewee_system.j2 模板以获得完整的 prompt 质量。
    """

    def __init__(self, llm: LLMClient, working_memory: WorkingMemory) -> None:
        self.llm = llm
        self.working_memory = working_memory
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def respond(self, question: str, context: dict[str, Any]) -> str:
        """基于简历和记忆生成高质量回答"""
        fact_table = context.get("fact_table", "")

        # 优先使用 Jinja2 模板（与 IntervieweeAgent._ai_respond 一致）
        try:
            template = self._jinja_env.get_template("interviewee_system.j2")
            system_prompt = template.render(fact_table=fact_table)
        except Exception:
            logger.warning("Failed to load interviewee_system.j2, using fallback")
            system_prompt = _SYSTEM_FALLBACK.format(
                fact_table=fact_table if fact_table else "暂无抽取的事实。"
            )

        messages = self.working_memory.compose_for_agent(
            system_prompt=system_prompt,
            policy=INTERVIEWEE_POLICY,
            current_input=question,
            role_map={"interviewer": "user", "interviewee": "assistant"},
        )

        return await self.llm.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
