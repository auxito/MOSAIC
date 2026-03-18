"""
真人练习工作流 — 跳过简历修改，真人直接面试。
"""

from __future__ import annotations

from mosaic.core.workflow import Phase
from mosaic.workflows.full_interview import (
    handle_evaluation,
    handle_init,
    handle_interview_loop,
    handle_report,
    handle_resume_input,
)

PHASE_HANDLERS = {
    Phase.INIT: handle_init,
    Phase.RESUME_INPUT: handle_resume_input,
    Phase.INTERVIEW_LOOP: handle_interview_loop,
    Phase.EVALUATION: handle_evaluation,
    Phase.REPORT: handle_report,
}
