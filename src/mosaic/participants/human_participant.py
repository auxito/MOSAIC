"""
Human Participant — 真人面试者。
通过终端获取用户输入，同样享有完整的记忆系统支持。
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


class HumanParticipant:
    """
    真人面试参与者。
    从终端获取用户输入，并展示辅助信息。
    """

    def __init__(self, show_hints: bool = False) -> None:
        """
        Args:
            show_hints: 是否显示辅助提示（事实表、矛盾提醒等）
        """
        self.show_hints = show_hints

    async def respond(self, question: str, context: dict[str, Any]) -> str:
        """获取真人回答"""

        # 显示面试官问题
        console.print()
        console.print(Panel(
            Markdown(question),
            title="[bold blue]面试官提问[/bold blue]",
            border_style="blue",
        ))

        # 可选：显示辅助提示
        if self.show_hints:
            fact_table = context.get("fact_table", "")
            if fact_table and fact_table != "暂无抽取的事实。":
                console.print(Panel(
                    fact_table,
                    title="[dim]辅助信息：已知事实[/dim]",
                    border_style="dim",
                ))

        # 获取用户输入
        console.print("\n[bold green]你的回答[/bold green]（输入完成后按两次回车提交）:")
        lines: list[str] = []
        empty_count = 0
        while True:
            try:
                line = input()
                if line == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                    lines.append("")
                else:
                    empty_count = 0
                    lines.append(line)
            except EOFError:
                break

        answer = "\n".join(lines).strip()
        if not answer:
            answer = "（跳过此问题）"

        return answer
