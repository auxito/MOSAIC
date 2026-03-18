"""
简历数据模型 — Pydantic 结构化格式。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Education(BaseModel):
    """教育经历"""
    school: str = ""
    degree: str = ""
    major: str = ""
    start_year: str = ""
    end_year: str = ""
    gpa: str = ""
    highlights: list[str] = Field(default_factory=list)


class WorkExperience(BaseModel):
    """工作经历"""
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""
    achievements: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)


class Project(BaseModel):
    """项目经历"""
    name: str = ""
    role: str = ""
    description: str = ""
    achievements: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    """标准化简历格式"""
    name: str = ""
    target_position: str = ""
    phone: str = ""
    email: str = ""
    summary: str = ""
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    extra: dict = Field(default_factory=dict)

    def to_text(self) -> str:
        """转为可读文本（用于 LLM 上下文）"""
        sections: list[str] = []
        sections.append(f"# {self.name}")
        if self.target_position:
            sections.append(f"**求职意向**: {self.target_position}")
        if self.summary:
            sections.append(f"\n## 个人简介\n{self.summary}")

        if self.education:
            sections.append("\n## 教育经历")
            for e in self.education:
                line = f"- {e.school} | {e.degree} {e.major} ({e.start_year}-{e.end_year})"
                if e.gpa:
                    line += f" GPA: {e.gpa}"
                sections.append(line)
                for h in e.highlights:
                    sections.append(f"  - {h}")

        if self.work_experience:
            sections.append("\n## 工作经历")
            for w in self.work_experience:
                sections.append(
                    f"- **{w.company}** | {w.title} ({w.start_date}-{w.end_date})"
                )
                if w.description:
                    sections.append(f"  {w.description}")
                for a in w.achievements:
                    sections.append(f"  - {a}")
                if w.tech_stack:
                    sections.append(f"  技术栈: {', '.join(w.tech_stack)}")

        if self.projects:
            sections.append("\n## 项目经历")
            for p in self.projects:
                sections.append(f"- **{p.name}** | {p.role}")
                if p.description:
                    sections.append(f"  {p.description}")
                for a in p.achievements:
                    sections.append(f"  - {a}")
                if p.tech_stack:
                    sections.append(f"  技术栈: {', '.join(p.tech_stack)}")

        if self.skills:
            sections.append(f"\n## 技能\n{', '.join(self.skills)}")

        if self.certifications:
            sections.append(f"\n## 证书\n{', '.join(self.certifications)}")

        return "\n".join(sections)
