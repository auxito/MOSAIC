"""
ResumeParser Agent — 简历解析。
v1: 文本解析。架构预留 PDF/DOCX 扩展。
"""

from __future__ import annotations

import logging
from typing import Any

from mosaic.core.agent import BaseAgent, MemoryPolicy
from mosaic.core.events import Event, EventType
from mosaic.resume.file_parser import FileParser
from mosaic.resume.schema import ResumeData
from mosaic.resume.structured_input import resume_from_text

logger = logging.getLogger(__name__)


class ResumeParserAgent(BaseAgent):
    """
    简历解析 Agent。

    v1 支持：
    - ResumeData 对象直接传入
    - 纯文本输入
    - 文本文件（.txt / .md）

    v2 预留：
    - PDF / DOCX 解析

    事件发布：RESUME_PARSED
    """

    def __init__(self) -> None:
        super().__init__(name="resume_parser", memory_policy=MemoryPolicy())
        self._file_parser = FileParser()

    async def parse(
        self,
        resume_input: str | dict | ResumeData,
        input_type: str = "auto",
    ) -> ResumeData:
        """
        解析简历输入。

        Args:
            resume_input: 简历内容（文本/dict/ResumeData/文件路径）
            input_type: "auto" | "text" | "dict" | "file"
        """
        if isinstance(resume_input, ResumeData):
            result = resume_input
        elif isinstance(resume_input, dict):
            result = ResumeData(**resume_input)
        elif isinstance(resume_input, str):
            if input_type == "file" or (
                input_type == "auto" and len(resume_input) < 500 and "." in resume_input
            ):
                # 尝试作为文件路径
                try:
                    result = self._file_parser.parse(resume_input)
                except (FileNotFoundError, NotImplementedError) as e:
                    logger.warning(f"File parse failed: {e}, treating as text")
                    result = resume_from_text(resume_input)
            else:
                result = resume_from_text(resume_input)
        else:
            raise TypeError(f"Unsupported resume input type: {type(resume_input)}")

        await self._emit(Event(
            type=EventType.RESUME_PARSED,
            source=self.name,
            data={"name": result.name},
        ))

        logger.info(f"Resume parsed: {result.name}")
        return result

    async def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Agent 接口"""
        resume_input = context.get("resume_input", "")
        input_type = context.get("input_type", "auto")
        result = await self.parse(resume_input, input_type)
        return {"resume": result.model_dump()}
