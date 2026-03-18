"""
上下文窗口管理器 — 渐进式摘要 + token 预算。
解决长对话超出上下文窗口的问题。
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import tiktoken

    _encoder = tiktoken.encoding_for_model("gpt-4o")

    def count_tokens(text: str) -> int:
        return len(_encoder.encode(text))

except ImportError:
    # fallback: 粗略估算（1 token ≈ 1.5 中文字符 / 4 英文字符）
    def count_tokens(text: str) -> int:
        # 简单混合估算
        cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        en_chars = len(text) - cn_chars
        return int(cn_chars / 1.5 + en_chars / 4)


logger = logging.getLogger(__name__)


class ContextWindowManager:
    """
    管理 LLM 调用时的上下文窗口组合。

    优先级（从不裁剪到最先裁剪）：
    1. 系统提示 + 人设 → 永不裁剪
    2. 特权记忆（简历/JD）→ 永不裁剪
    3. 语义事实表 → 永不裁剪（本身紧凑）
    4. 最近K轮原文 → 先缩小K再裁剪
    5. 渐进式摘要 → 最可压缩
    """

    def __init__(
        self,
        token_budget: int = 12000,
        verbatim_window: int = 6,
        summary_batch: int = 5,
    ) -> None:
        self.token_budget = token_budget
        self.verbatim_window = verbatim_window
        self.summary_batch = summary_batch

    def compose_messages(
        self,
        system_prompt: str,
        privileged_info: dict[str, Any],
        fact_table: str,
        conversation_summary: str,
        recent_turns: list[dict[str, str]],
        current_input: str | None = None,
    ) -> list[dict[str, str]]:
        """
        组合最终发送给 LLM 的 messages 列表。
        自动管理 token 预算。
        """
        messages: list[dict[str, str]] = []

        # 1. 系统提示（含特权信息和事实表）
        system_parts = [system_prompt]

        if privileged_info:
            for key, value in privileged_info.items():
                if isinstance(value, dict):
                    # 结构化数据格式化
                    formatted = self._format_dict(key, value)
                else:
                    formatted = f"\n## {key}\n{value}"
                system_parts.append(formatted)

        if fact_table:
            system_parts.append(f"\n{fact_table}")

        system_content = "\n".join(system_parts)
        messages.append({"role": "system", "content": system_content})

        # 计算已用 tokens
        used = count_tokens(system_content)
        if current_input:
            used += count_tokens(current_input)

        remaining = self.token_budget - used

        # 2. 加入摘要（如果有）
        summary_tokens = 0
        if conversation_summary:
            summary_msg = f"[对话摘要]\n{conversation_summary}"
            summary_tokens = count_tokens(summary_msg)

        # 3. 计算能放多少轮原文
        recent_text = "\n".join(t["content"] for t in recent_turns)
        recent_tokens = count_tokens(recent_text)

        if summary_tokens + recent_tokens <= remaining:
            # 全放得下
            if conversation_summary:
                messages.append(
                    {"role": "system", "content": f"[对话摘要]\n{conversation_summary}"}
                )
            messages.extend(recent_turns)
        elif recent_tokens <= remaining:
            # 只放原文，不放摘要
            messages.extend(recent_turns)
        else:
            # 需要裁剪：先放摘要，再尽量多放近期原文
            if conversation_summary:
                messages.append(
                    {"role": "system", "content": f"[对话摘要]\n{conversation_summary}"}
                )
                remaining -= summary_tokens

            # 从最近的开始，尽量多放
            fitted: list[dict[str, str]] = []
            for turn in reversed(recent_turns):
                turn_tokens = count_tokens(turn["content"])
                if remaining >= turn_tokens:
                    fitted.insert(0, turn)
                    remaining -= turn_tokens
                else:
                    break
            messages.extend(fitted)
            logger.info(
                f"Context trimmed: {len(recent_turns)} → {len(fitted)} turns"
            )

        # 4. 当前输入
        if current_input:
            messages.append({"role": "user", "content": current_input})

        return messages

    def should_summarize(self, total_rounds: int, last_summary_round: int) -> bool:
        """是否应该生成新摘要"""
        rounds_since = total_rounds - last_summary_round
        return rounds_since >= self.summary_batch

    def _format_dict(self, title: str, data: dict) -> str:
        """格式化字典为可读文本"""
        lines = [f"\n## {title}"]
        for k, v in data.items():
            if isinstance(v, list):
                lines.append(f"**{k}**:")
                for item in v:
                    lines.append(f"  - {item}")
            elif isinstance(v, dict):
                lines.append(f"**{k}**:")
                for sk, sv in v.items():
                    lines.append(f"  - {sk}: {sv}")
            else:
                lines.append(f"**{k}**: {v}")
        return "\n".join(lines)
