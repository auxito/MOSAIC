"""
全局配置
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class Config:
    """运行时配置"""

    # LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # 上下文窗口
    token_budget: int = 12000
    verbatim_window: int = 6
    summary_batch: int = 5

    # 面试
    default_rounds: int = 10

    # 输出
    report_dir: str = "reports"

    @classmethod
    def from_env(cls) -> Config:
        """从环境变量加载"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            token_budget=int(os.getenv("TOKEN_BUDGET", "12000")),
            verbatim_window=int(os.getenv("VERBATIM_WINDOW", "6")),
            summary_batch=int(os.getenv("SUMMARY_BATCH", "5")),
            default_rounds=int(os.getenv("DEFAULT_ROUNDS", "10")),
            report_dir=os.getenv("REPORT_DIR", "reports"),
        )
