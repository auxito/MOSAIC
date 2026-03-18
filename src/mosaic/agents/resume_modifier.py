"""
向后兼容模块 — 已迁移至 career_coach.py。
ResumeModifierAgent 现为 CareerCoachAgent 的别名。
"""

from mosaic.agents.career_coach import CareerCoachAgent

# 向后兼容别名
ResumeModifierAgent = CareerCoachAgent

__all__ = ["ResumeModifierAgent", "CareerCoachAgent"]
