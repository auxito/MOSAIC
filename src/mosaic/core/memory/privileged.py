"""
特权记忆 — 按 MemoryPolicy 控制每个 Agent 的信息可见范围。
核心功能：信息不对称。
"""

from __future__ import annotations

from typing import Any

from mosaic.core.agent import MemoryPolicy


class PrivilegedMemory:
    """
    特权信息存储 + 按策略过滤。

    存储：
    - original_resume: 原始简历
    - modified_resume: 修改后简历
    - job_description: JD
    - evaluation_criteria: 评分标准

    每个 Agent 调用 get_visible() 时，只返回其 MemoryPolicy 允许的内容。
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {
            "original_resume": None,
            "modified_resume": None,
            "job_description": None,
            "evaluation_criteria": None,
        }

    def set(self, key: str, value: Any) -> None:
        """设置特权信息"""
        if key not in self._store:
            raise KeyError(f"Unknown privileged key: {key}")
        self._store[key] = value

    def get_visible(self, policy: MemoryPolicy) -> dict[str, Any]:
        """按策略返回可见的特权信息"""
        visible: dict[str, Any] = {}

        if policy.can_see_original_resume and self._store["original_resume"]:
            visible["original_resume"] = self._store["original_resume"]

        if policy.can_see_modified_resume and self._store["modified_resume"]:
            visible["modified_resume"] = self._store["modified_resume"]

        if policy.can_see_jd and self._store["job_description"]:
            visible["job_description"] = self._store["job_description"]

        if policy.can_see_evaluation and self._store["evaluation_criteria"]:
            visible["evaluation_criteria"] = self._store["evaluation_criteria"]

        return visible

    def get_raw(self, key: str) -> Any:
        """直接访问（仅 Orchestrator 使用）"""
        return self._store.get(key)

    def clear(self) -> None:
        """重置"""
        for key in self._store:
            self._store[key] = None
