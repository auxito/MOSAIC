"""
文件解析器 — PDF/DOCX/TXT → ResumeData。
支持文件路径和字节流两种输入方式（后者用于 Web 上传）。

两步解析：
1. 基础解析：提取原始文本 → ResumeData(summary=原文)
2. LLM 结构化解析（可选）：用 LLM 把原始文本拆为结构化字段，同时保留原文到 summary
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

from mosaic.resume.schema import Education, Project, ResumeData, WorkExperience

logger = logging.getLogger(__name__)


# LLM 结构化解析的 prompt
_LLM_PARSE_PROMPT = """\
你是一个简历解析助手。请将以下简历原始文本解析为结构化 JSON 格式。

要求：
1. **完整提取**所有信息，不要遗漏任何内容，不要精简或概括
2. 如果某个字段在简历中找不到，留空字符串或空数组
3. achievements 和 tech_stack 是数组，每项一个要点，保持原文表述
4. 日期格式保持原文格式即可
5. skills 是技能关键词数组
6. description 字段保留原文的完整描述，不要缩减

请严格按照以下 JSON Schema 输出（不要输出任何其他内容）：
{
  "name": "姓名",
  "phone": "手机号",
  "email": "邮箱",
  "target_position": "求职意向",
  "summary": "个人简介/自我评价（如有，保留原文）",
  "education": [
    {
      "school": "学校名",
      "degree": "学位（本科/硕士/博士等）",
      "major": "专业",
      "start_year": "入学年份",
      "end_year": "毕业年份",
      "gpa": "GPA（如有）",
      "highlights": ["荣誉/奖项，保留原文"]
    }
  ],
  "work_experience": [
    {
      "company": "公司名",
      "title": "职位",
      "start_date": "开始时间",
      "end_date": "结束时间",
      "description": "工作描述（保留原文完整内容）",
      "achievements": ["工作成果1（原文）", "工作成果2（原文）"],
      "tech_stack": ["技术1", "技术2"]
    }
  ],
  "projects": [
    {
      "name": "项目名",
      "role": "角色",
      "description": "项目描述（保留原文完整内容）",
      "achievements": ["项目成果1（原文）"],
      "tech_stack": ["技术1"]
    }
  ],
  "skills": ["技能1", "技能2"],
  "certifications": ["证书1"],
  "languages": ["语言能力1"]
}

--- 简历原文 ---
"""


class FileParser:
    """简历文件解析器"""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    def parse(self, file_path: str | Path) -> ResumeData:
        """解析简历文件（基础解析，仅提取原始文本）"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if ext in (".txt", ".md"):
            return self._parse_text(path)
        elif ext == ".pdf":
            return self._parse_pdf(path)
        elif ext == ".docx":
            return self._parse_docx(path)

        raise ValueError(f"Unhandled format: {ext}")

    @classmethod
    def parse_bytes(cls, filename: str, data: bytes) -> ResumeData:
        """
        从字节流解析简历（Gradio 上传场景）。

        Args:
            filename: 原始文件名（用于判断格式）
            data: 文件二进制内容

        Returns:
            ResumeData
        """
        ext = Path(filename).suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {ext}. Supported: {cls.SUPPORTED_EXTENSIONS}"
            )

        parser = cls()

        if ext in (".txt", ".md"):
            return parser._parse_text_bytes(data)
        elif ext == ".pdf":
            return parser._parse_pdf_bytes(data)
        elif ext == ".docx":
            return parser._parse_docx_bytes(data)

        raise ValueError(f"Unhandled format: {ext}")

    # ---- LLM 结构化解析 ----

    @staticmethod
    async def llm_structured_parse(raw_text: str, llm_client) -> ResumeData:
        """用 LLM 将简历原始文本解析为结构化 ResumeData。

        Args:
            raw_text: 简历原始文本（从 PDF/DOCX/TXT 提取的完整内容）
            llm_client: LLMClient 实例

        Returns:
            ResumeData — 结构化字段已填充，summary 保留完整原文
        """
        try:
            response = await llm_client.chat_json(
                messages=[
                    {"role": "system", "content": "你是一个专业的简历解析助手，只输出 JSON。注意保留原文内容，不要精简。"},
                    {"role": "user", "content": _LLM_PARSE_PROMPT + raw_text},
                ],
                temperature=0.1,
                max_tokens=4000,
            )

            json_text = response.strip()
            if json_text.startswith("```"):
                json_text = re.sub(r"^```(?:json)?\s*", "", json_text)
                json_text = re.sub(r"\s*```$", "", json_text)

            data = json.loads(json_text)

            education = []
            for e in data.get("education", []):
                education.append(Education(
                    school=e.get("school", ""),
                    degree=e.get("degree", ""),
                    major=e.get("major", ""),
                    start_year=str(e.get("start_year", "")),
                    end_year=str(e.get("end_year", "")),
                    gpa=str(e.get("gpa", "")),
                    highlights=e.get("highlights", []),
                ))

            work_experience = []
            for w in data.get("work_experience", []):
                work_experience.append(WorkExperience(
                    company=w.get("company", ""),
                    title=w.get("title", ""),
                    start_date=str(w.get("start_date", "")),
                    end_date=str(w.get("end_date", "")),
                    description=w.get("description", ""),
                    achievements=w.get("achievements", []),
                    tech_stack=w.get("tech_stack", []),
                ))

            projects = []
            for p in data.get("projects", []):
                projects.append(Project(
                    name=p.get("name", ""),
                    role=p.get("role", ""),
                    description=p.get("description", ""),
                    achievements=p.get("achievements", []),
                    tech_stack=p.get("tech_stack", []),
                ))

            # 关键：summary 保留完整原文，不用 LLM 精简过的
            resume_data = ResumeData(
                name=data.get("name", ""),
                phone=data.get("phone", ""),
                email=data.get("email", ""),
                target_position=data.get("target_position", ""),
                summary=raw_text.strip(),  # 保留完整原文
                education=education,
                work_experience=work_experience,
                projects=projects,
                skills=data.get("skills", []),
                certifications=data.get("certifications", []),
                languages=data.get("languages", []),
            )

            logger.info(
                f"LLM resume parse OK: name={resume_data.name}, "
                f"edu={len(education)}, work={len(work_experience)}, "
                f"proj={len(projects)}, skills={len(resume_data.skills)}"
            )
            return resume_data

        except Exception as e:
            logger.warning(f"LLM resume parse failed, falling back to summary mode: {e}")
            return ResumeData(name="候选人", summary=raw_text.strip())

    # ---- PDF ----

    def _parse_pdf(self, path: Path) -> ResumeData:
        """解析 PDF 文件"""
        with open(path, "rb") as f:
            return self._parse_pdf_bytes(f.read())

    def _parse_pdf_bytes(self, data: bytes) -> ResumeData:
        """从字节流解析 PDF"""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "PDF 解析需要 pdfplumber 库。请运行: pip install pdfplumber"
            )

        pages_text: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text)
        if not full_text.strip():
            raise ValueError("PDF 文件中未提取到文本内容，可能是扫描件或图片 PDF。")

        logger.info(f"PDF parsed: {len(pages_text)} pages, {len(full_text)} chars")
        return ResumeData(name="候选人", summary=full_text.strip())

    # ---- DOCX ----

    def _parse_docx(self, path: Path) -> ResumeData:
        """解析 DOCX 文件"""
        with open(path, "rb") as f:
            return self._parse_docx_bytes(f.read())

    def _parse_docx_bytes(self, data: bytes) -> ResumeData:
        """从字节流解析 DOCX"""
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "DOCX 解析需要 python-docx 库。请运行: pip install python-docx"
            )

        doc = Document(io.BytesIO(data))
        paragraphs: list[str] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # 保留标题层级
            style_name = (para.style.name or "").lower() if para.style else ""
            if "heading 1" in style_name:
                paragraphs.append(f"# {text}")
            elif "heading 2" in style_name:
                paragraphs.append(f"## {text}")
            elif "heading 3" in style_name:
                paragraphs.append(f"### {text}")
            elif "heading" in style_name:
                paragraphs.append(f"#### {text}")
            else:
                paragraphs.append(text)

        full_text = "\n".join(paragraphs)
        if not full_text.strip():
            raise ValueError("DOCX 文件中未提取到文本内容。")

        logger.info(f"DOCX parsed: {len(paragraphs)} paragraphs, {len(full_text)} chars")
        return ResumeData(name="候选人", summary=full_text.strip())

    # ---- TXT / MD ----

    def _parse_text(self, path: Path) -> ResumeData:
        """解析纯文本简历，支持多编码 fallback"""
        # 尝试多种编码
        for encoding in ("utf-8", "gb18030", "gbk", "latin-1"):
            try:
                text = path.read_text(encoding=encoding)
                logger.debug(f"Text file read with encoding: {encoding}")
                return ResumeData(name="候选人", summary=text.strip())
            except (UnicodeDecodeError, LookupError):
                continue

        raise ValueError(f"无法解码文本文件: {path}（已尝试 utf-8/gb18030/gbk/latin-1）")

    def _parse_text_bytes(self, data: bytes) -> ResumeData:
        """从字节流解析文本"""
        for encoding in ("utf-8", "gb18030", "gbk", "latin-1"):
            try:
                text = data.decode(encoding)
                return ResumeData(name="候选人", summary=text.strip())
            except (UnicodeDecodeError, LookupError):
                continue

        raise ValueError("无法解码文本内容（已尝试 utf-8/gb18030/gbk/latin-1）")
