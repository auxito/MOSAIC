"""
文件解析器 — PDF/DOCX/TXT → ResumeData。
支持文件路径和字节流两种输入方式（后者用于 Web 上传）。
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from mosaic.resume.schema import ResumeData

logger = logging.getLogger(__name__)


class FileParser:
    """简历文件解析器"""

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
