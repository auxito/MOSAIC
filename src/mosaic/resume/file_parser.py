"""
文件解析器 — PDF/DOCX → ResumeData。
v1 为 stub 实现，架构预留扩展。
"""

from __future__ import annotations

from pathlib import Path

from mosaic.resume.schema import ResumeData


class FileParser:
    """简历文件解析器（v1 stub）"""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    def parse(self, file_path: str | Path) -> ResumeData:
        """解析简历文件"""
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

    def _parse_text(self, path: Path) -> ResumeData:
        """解析纯文本简历"""
        text = path.read_text(encoding="utf-8")
        return ResumeData(name="候选人", summary=text.strip())

    def _parse_pdf(self, path: Path) -> ResumeData:
        """PDF 解析 — v1 stub"""
        raise NotImplementedError(
            "PDF 解析将在 v2 实现。请使用文本格式输入简历，"
            "或手动将 PDF 内容复制为文本。"
        )

    def _parse_docx(self, path: Path) -> ResumeData:
        """DOCX 解析 — v1 stub"""
        raise NotImplementedError(
            "DOCX 解析将在 v2 实现。请使用文本格式输入简历，"
            "或手动将 DOCX 内容复制为文本。"
        )
