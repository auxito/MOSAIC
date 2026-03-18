"""
仅简历修改工作流。
"""

from __future__ import annotations

from mosaic.core.workflow import Phase
from mosaic.workflows.full_interview import (
    handle_init,
    handle_report,
    handle_resume_input,
    handle_resume_modify,
)

PHASE_HANDLERS = {
    Phase.INIT: handle_init,
    Phase.RESUME_INPUT: handle_resume_input,
    Phase.RESUME_MODIFY: handle_resume_modify,
    Phase.REPORT: handle_report,
}
