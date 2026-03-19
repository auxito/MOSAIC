"""
MOSAIC Web UI — Gradio 应用。
提供简历优化、模拟面试、评估报告的 Web 交互界面。

独立入口，不影响 CLI 流程。直接调用 agents 方法编排流程。
"""

from __future__ import annotations

import asyncio
import logging
import re
import traceback
from typing import Any

import gradio as gr

from mosaic.config import Config
from mosaic.core.agent import AgentState
from mosaic.core.events import EventBus
from mosaic.core.memory import (
    ContextWindowManager,
    DialogueTurn,
    EpisodicMemory,
    PrivilegedMemory,
    SemanticMemory,
    WorkingMemory,
)
from mosaic.core.workflow import FULL_INTERVIEW, WorkflowConfig
from mosaic.llm.client import LLMClient
from mosaic.resume.file_parser import FileParser
from mosaic.resume.schema import ResumeData

# Agents
from mosaic.agents.career_coach import CareerCoachAgent
from mosaic.agents.evaluator import EvaluatorAgent
from mosaic.agents.interviewee import IntervieweeAgent
from mosaic.agents.interviewer import InterviewerAgent
from mosaic.agents.memory_manager import MemoryManagerAgent
from mosaic.participants.ai_participant import AIParticipant
from mosaic.output.report import generate_report

logger = logging.getLogger(__name__)

# Gradio 6 把 theme/css 移到 launch()，这里统一维护
LAUNCH_KWARGS = {
    "theme": gr.themes.Soft(primary_hue="indigo", radius_size="lg"),
    "css": """
/* ── Header Banner ── */
.header-banner {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    color: #fff;
    box-shadow: 0 8px 32px rgba(79, 70, 229, 0.25);
    margin-bottom: 1.2rem;
}
.header-banner h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.3rem;
    letter-spacing: 0.02em;
}
.header-banner .subtitle {
    font-size: 0.95rem;
    opacity: 0.9;
    margin-bottom: 1rem;
}
.header-banner .features {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.header-banner .feat {
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-size: 0.88rem;
    backdrop-filter: blur(4px);
}

/* ── Section Title ── */
.section-title {
    border-left: 4px solid #6366f1;
    padding-left: 0.8rem;
    font-weight: 700;
    font-size: 1.1rem;
    margin: 0.8rem 0 0.5rem;
    color: #312e81;
}

/* ── Tip Box ── */
.tip-box {
    background: #f0f0ff;
    border-left: 4px solid #818cf8;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.8rem;
    font-size: 0.9rem;
    color: #3730a3;
    line-height: 1.6;
}

/* ── Button Enhancements ── */
button.primary {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
}
button.primary:active {
    transform: translateY(0);
}

/* ── Chatbot Area ── */
.chatbot-container { min-height: 500px; }

/* ── General Spacing ── */
.gr-panel { margin-bottom: 0.6rem; }
.gr-tab-content { padding-top: 0.5rem; }

/* ── Upload Bar — 去掉灰色背景 ── */
.upload-bar {
    border: none !important;
    background: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin-bottom: 0.5rem;
}
.upload-bar .gr-file,
.upload-bar .gr-group {
    background: transparent !important;
    border: 1.5px dashed #c7d2fe !important;
    border-radius: 12px !important;
}

/* ── Resume Form Section ── */
.resume-form-section {
    border: 1px solid #e0e0ef;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    background: #fafaff;
}
.resume-form-section .section-label {
    font-weight: 700;
    font-size: 1rem;
    color: #4338ca;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── 动态经历卡片 ── */
.exp-card {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    background: #fff;
    transition: box-shadow 0.15s;
}
.exp-card:hover {
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1);
}

/* ── 加减按钮行 ── */
.add-remove-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.3rem;
}
.add-remove-row button {
    min-width: 0 !important;
    padding: 0.3rem 0.8rem !important;
    font-size: 0.85rem !important;
}

/* ── Diff 优化结果 — 整体双栏布局 ── */
.result-split {
    display: flex;
    gap: 0;
    min-height: 400px;
    border: 1px solid #e0e0ef;
    border-radius: 12px;
    overflow: hidden;
    background: #fff;
}
.result-left {
    flex: 3;
    padding: 1rem 1.2rem;
    overflow-y: auto;
    max-height: 650px;
}
.result-divider {
    width: 1px;
    background: linear-gradient(180deg, #c7d2fe 0%, #e0e0ef 50%, #c7d2fe 100%);
    flex-shrink: 0;
}
.result-right {
    flex: 2;
    padding: 1rem 1.2rem;
    overflow-y: auto;
    max-height: 650px;
    background: #fafaff;
}

/* 左栏标题 */
.result-left-title {
    font-weight: 700;
    font-size: 1.05rem;
    color: #4338ca;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e0e7ff;
}
/* 右栏标题 */
.result-right-title {
    font-weight: 700;
    font-size: 1.05rem;
    color: #4338ca;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e0e7ff;
}

/* Diff 卡片 */
.diff-item {
    border: 1px solid #f0f0f5;
    border-radius: 8px;
    padding: 0.7rem 0.8rem;
    margin-bottom: 0.6rem;
    background: #fafafa;
    transition: border-color 0.15s;
}
.diff-item:hover {
    border-color: #c7d2fe;
}
.diff-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6366f1;
    margin-bottom: 0.4rem;
    letter-spacing: 0.02em;
}
.diff-old {
    color: #b91c1c;
    text-decoration: line-through;
    background: #fef2f2;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    display: block;
    line-height: 1.65;
    font-size: 0.88rem;
    word-break: break-word;
}
.diff-arrow {
    color: #a5b4fc;
    font-size: 0.85rem;
    margin: 0.15rem 0 0.15rem 0.5rem;
    display: block;
}
.diff-new {
    color: #15803d;
    background: #f0fdf4;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    display: block;
    font-weight: 500;
    line-height: 1.65;
    font-size: 0.88rem;
    word-break: break-word;
}
.diff-reason {
    margin-top: 0.35rem;
    font-size: 0.78rem;
    color: #6b7280;
    line-height: 1.45;
    padding-top: 0.3rem;
    border-top: 1px dashed #ebebf0;
}

/* 右栏教练笔记内容 */
.coach-content {
    font-size: 0.88rem;
    line-height: 1.6;
    color: #374151;
}
.coach-content h1, .coach-content h2, .coach-content h3, .coach-content h4 {
    color: #312e81;
    margin-top: 0.7rem;
    margin-bottom: 0.3rem;
}
.coach-content h1 { font-size: 1.05rem; font-weight: 700; }
.coach-content h2 { font-size: 0.95rem; font-weight: 700; }
.coach-content h3 { font-size: 0.9rem; font-weight: 600; }
.coach-content h4 { font-size: 0.85rem; font-weight: 600; }
.coach-content p {
    margin: 0.2rem 0 0.4rem;
}
.coach-content ul, .coach-content ol {
    margin: 0.2rem 0 0.4rem 1.1rem;
    padding: 0;
}
.coach-content li {
    margin: 0.1rem 0;
}
.coach-content strong {
    color: #4338ca;
}
.coach-content code {
    background: #eef2ff;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    font-size: 0.82rem;
}
.coach-content table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin: 0.4rem 0;
}
.coach-content th, .coach-content td {
    border: 1px solid #e5e7eb;
    padding: 0.3rem 0.5rem;
    text-align: left;
}
.coach-content th {
    background: #f0f0ff;
    font-weight: 600;
}

/* ── 开始优化按钮加大 ── */
.optimize-btn button {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.5rem !important;
    letter-spacing: 0.05em;
}

/* ── Result Area ── */
.result-area {
    border-top: 3px solid #6366f1;
    padding-top: 1rem;
    margin-top: 1rem;
}

/* ── 评分翻页卡片 ── */
.eval-carousel {
    position: relative;
    min-height: 280px;
    border: 1px solid #e0e0ef;
    border-radius: 14px;
    background: linear-gradient(135deg, #fafaff 0%, #f5f3ff 100%);
    overflow: hidden;
}
.eval-card {
    display: none;
    padding: 1.2rem 1.4rem;
    animation: evalFadeIn 0.3s ease;
}
.eval-card.active {
    display: block;
}
@keyframes evalFadeIn {
    from { opacity: 0; transform: translateX(12px); }
    to   { opacity: 1; transform: translateX(0); }
}
.eval-card-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
}
.eval-card-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
    white-space: nowrap;
}
.eval-card-title {
    font-weight: 600;
    font-size: 1rem;
    color: #312e81;
}
.eval-card table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.eval-card th, .eval-card td {
    padding: 0.35rem 0.6rem;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}
.eval-card th {
    color: #6b7280;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.eval-card td:last-child {
    font-weight: 600;
    color: #4338ca;
}
.eval-card-comment {
    margin-top: 0.6rem;
    font-size: 0.88rem;
    color: #374151;
    line-height: 1.55;
}
.eval-card-comment strong {
    color: #4338ca;
}
.eval-card-extras {
    margin-top: 0.5rem;
    font-size: 0.82rem;
    color: #6b7280;
    line-height: 1.5;
}
/* 翻页导航 */
.eval-nav {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.6rem 0 0.8rem;
    border-top: 1px solid #e5e7eb;
    background: #fff;
}
.eval-nav-btn {
    border: 1px solid #d1d5db;
    background: #fff;
    border-radius: 8px;
    padding: 0.3rem 0.75rem;
    font-size: 0.85rem;
    cursor: pointer;
    color: #4338ca;
    font-weight: 600;
    transition: all 0.15s;
}
.eval-nav-btn:hover {
    background: #eef2ff;
    border-color: #a5b4fc;
}
.eval-nav-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}
.eval-nav-indicator {
    font-size: 0.82rem;
    color: #6b7280;
    min-width: 5rem;
    text-align: center;
}
/* 圆点指示器 */
.eval-dots {
    display: flex;
    gap: 0.35rem;
    justify-content: center;
    padding-bottom: 0.5rem;
}
.eval-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #d1d5db;
    cursor: pointer;
    transition: all 0.15s;
}
.eval-dot.active {
    background: #6366f1;
    transform: scale(1.2);
}

/* ── 面试开始按钮 ── */
.interview-start-btn button {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}
""",
}


# ============================================================
# Helper: 构建 session 运行时组件
# ============================================================

def _build_session_components(config: Config) -> dict[str, Any]:
    """构建一次面试会话所需的全部组件"""
    llm = LLMClient(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        model=config.openai_model,
    )

    # 四层记忆
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    privileged = PrivilegedMemory()
    context_manager = ContextWindowManager()
    working_memory = WorkingMemory(
        episodic=episodic,
        semantic=semantic,
        privileged=privileged,
        context_manager=context_manager,
    )

    # 事件总线
    event_bus = EventBus()

    # Agents
    memory_mgr = MemoryManagerAgent(llm=llm, working_memory=working_memory)
    memory_mgr.bind_event_bus(event_bus)

    coach = CareerCoachAgent(llm=llm)
    coach.bind_event_bus(event_bus)

    interviewer = InterviewerAgent(llm=llm, working_memory=working_memory)
    interviewer.bind_event_bus(event_bus)

    ai_participant = AIParticipant(llm=llm, working_memory=working_memory)
    interviewee = IntervieweeAgent(
        llm=llm, working_memory=working_memory, participant=ai_participant,
    )
    interviewee.bind_event_bus(event_bus)

    evaluator = EvaluatorAgent(llm=llm, working_memory=working_memory)
    evaluator.bind_event_bus(event_bus)

    return {
        "llm": llm,
        "episodic": episodic,
        "semantic": semantic,
        "privileged": privileged,
        "working_memory": working_memory,
        "event_bus": event_bus,
        "memory_mgr": memory_mgr,
        "coach": coach,
        "interviewer": interviewer,
        "interviewee": interviewee,
        "evaluator": evaluator,
        "state": AgentState(),
    }


# ============================================================
# 格式化工具
# ============================================================

def _format_turn_eval(evaluation: dict[str, Any]) -> str:
    """格式化单轮评估为 HTML 卡片内容（不含 .eval-card 外壳）"""
    if "error" in evaluation:
        return f'<div style="color:#b91c1c;padding:1rem;">⚠️ 评估出错: {_escape_html(str(evaluation["error"]))}</div>'

    scores = evaluation.get("scores", {})
    rn = evaluation.get("round_number", "?")

    rows = ""
    for label, key in [
        ("回答质量", "answer_quality"),
        ("技术准确", "technical_accuracy"),
        ("沟通表达", "communication"),
        ("真实可信", "credibility"),
    ]:
        val = scores.get(key, "?")
        rows += f"<tr><td>{label}</td><td>{val}/5</td></tr>\n"

    comment = _escape_html(evaluation.get("comment", ""))
    extras = []

    highlights = evaluation.get("highlights", "")
    if highlights:
        extras.append(f"✨ <strong>亮点</strong>: {_escape_html(str(highlights))}")
    suggestions = evaluation.get("improvement_suggestions", "")
    if suggestions:
        extras.append(f"💡 <strong>改进</strong>: {_escape_html(str(suggestions))}")
    ref_points = evaluation.get("reference_points", [])
    if ref_points:
        extras.append(f"📌 <strong>参考</strong>: {'；'.join(_escape_html(str(r)) for r in ref_points)}")
    gaps = evaluation.get("knowledge_gaps", "")
    if gaps:
        extras.append(f"📚 <strong>盲区</strong>: {_escape_html(str(gaps))}")

    extras_html = '<div class="eval-card-extras">' + "<br>".join(extras) + "</div>" if extras else ""

    return f'''<div class="eval-card-header">
    <span class="eval-card-badge">第 {rn} 轮</span>
    <span class="eval-card-title">面试评估</span>
</div>
<table>
    <tr><th>维度</th><th>评分</th></tr>
    {rows}
</table>
<div class="eval-card-comment">📋 <strong>评语</strong>: {comment}</div>
{extras_html}'''


def _format_final_report(report: dict[str, Any]) -> str:
    """格式化最终评估为 Markdown"""
    if "error" in report:
        return f"⚠️ 评估出错: {report['error']}"

    lines = [
        "# 📊 综合评估报告",
        "",
        f"## 综合评分: {report.get('overall_score', 'N/A')}/10",
        "",
    ]

    dims = report.get("dimension_scores", {})
    if dims:
        lines.append("### 五维评分")
        lines.append("| 维度 | 评分 |")
        lines.append("|------|------|")
        for dim, score in dims.items():
            bar = "🟢" * score + "⚪" * (5 - score) if isinstance(score, int) else ""
            lines.append(f"| {dim} | {score}/5 {bar} |")
        lines.append("")

    strengths = report.get("strengths", [])
    if strengths:
        lines.append("### ✅ 优势")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    improvements = report.get("improvements", [])
    if improvements:
        lines.append("### 📈 改进建议")
        for i in improvements:
            lines.append(f"- {i}")
        lines.append("")

    rec = report.get("recommendation", "")
    if rec:
        lines.append(f"### 🏷️ 录用建议: **{rec}**")
        lines.append("")

    comments = report.get("detailed_comments", "")
    if comments:
        lines.append(f"### 详细评语\n{comments}")
        lines.append("")

    coaching = report.get("coaching_notes", "")
    if coaching:
        lines.append(f"### 🎯 教练建议\n{coaching}")
        lines.append("")

    roadmap = report.get("learning_roadmap", [])
    if roadmap:
        lines.append("### 📚 学习路线图")
        for item in roadmap:
            area = item.get("area", "")
            lines.append(f"\n#### {area}")
            current = item.get("current_level", "")
            if current:
                lines.append(f"- **当前水平**: {current}")
            target = item.get("target_level", "")
            if target:
                lines.append(f"- **目标水平**: {target}")
            timeline = item.get("timeline", "")
            if timeline:
                lines.append(f"- **建议周期**: {timeline}")
            resources = item.get("resources", [])
            if resources:
                lines.append("- **推荐资源**:")
                for r in resources:
                    lines.append(f"  - {r}")

    return "\n".join(lines)


def _ensure_session(session: dict | None) -> dict:
    """确保 session 已初始化"""
    if session is None:
        session = {
            "resume_data": None,
            "resume_text": "",
            "jd": "",
            "modified_resumes": {},
            "selected_style": "",
            "final_report": None,
            "state": None,
        }
    return session


# ============================================================
# LLM 结构化简历解析 — 委托给 FileParser.llm_structured_parse
# ============================================================


# ============================================================
# ResumeData ↔ dict 转换（用于 gr.State）
# ============================================================

def _resume_data_to_dict(rd: ResumeData) -> dict:
    """将 ResumeData 转为可 JSON 序列化的 dict（给 gr.State 用）"""
    return rd.model_dump()


def _dict_to_resume_data(d: dict) -> ResumeData:
    """从 dict 重建 ResumeData"""
    if not d:
        return ResumeData()
    return ResumeData(**d)


# ============================================================
# Diff HTML 构建（已废弃，改为直接渲染 coach 输出的 Markdown）
# ============================================================


def _md_to_html_simple(md_text: str) -> str:
    """简易 Markdown → HTML 转换（用于教练笔记右栏）。

    支持标题、列表、加粗、代码、表格等常见格式。
    """
    lines = md_text.split("\n")
    html_lines: list[str] = []
    in_ul = False
    in_ol = False
    in_table = False

    for line in lines:
        stripped = line.strip()

        # 空行：关闭列表
        if not stripped:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if in_table:
                html_lines.append("</table>")
                in_table = False
            continue

        # 表格（| 开头）
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            # 分隔行跳过
            if all(set(c.strip()) <= set("-: ") for c in cells):
                continue
            if not in_table:
                html_lines.append('<table>')
                # 首行作为表头
                html_lines.append("<tr>" + "".join(f"<th>{_escape_html(c)}</th>" for c in cells) + "</tr>")
                in_table = True
            else:
                html_lines.append("<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in cells) + "</tr>")
            continue

        if in_table:
            html_lines.append("</table>")
            in_table = False

        # 标题
        if stripped.startswith("####"):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            if in_ol: html_lines.append("</ol>"); in_ol = False
            html_lines.append(f'<h4>{_inline_md(stripped[4:].strip())}</h4>')
            continue
        if stripped.startswith("###"):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            if in_ol: html_lines.append("</ol>"); in_ol = False
            html_lines.append(f'<h3>{_inline_md(stripped[3:].strip())}</h3>')
            continue
        if stripped.startswith("##"):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            if in_ol: html_lines.append("</ol>"); in_ol = False
            html_lines.append(f'<h2>{_inline_md(stripped[2:].strip())}</h2>')
            continue
        if stripped.startswith("#"):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            if in_ol: html_lines.append("</ol>"); in_ol = False
            html_lines.append(f'<h1>{_inline_md(stripped[1:].strip())}</h1>')
            continue

        # 无序列表
        if re.match(r"^[-*•]\s", stripped):
            if not in_ul:
                if in_ol: html_lines.append("</ol>"); in_ol = False
                html_lines.append("<ul>")
                in_ul = True
            content = re.sub(r"^[-*•]\s+", "", stripped)
            html_lines.append(f"<li>{_inline_md(content)}</li>")
            continue

        # 有序列表
        if re.match(r"^\d+[.、]\s", stripped):
            if not in_ol:
                if in_ul: html_lines.append("</ul>"); in_ul = False
                html_lines.append("<ol>")
                in_ol = True
            content = re.sub(r"^\d+[.、]\s+", "", stripped)
            html_lines.append(f"<li>{_inline_md(content)}</li>")
            continue

        # 子列表（缩进 + - ）
        if re.match(r"^\s+[-*•]\s", stripped):
            content = re.sub(r"^\s*[-*•]\s+", "", stripped)
            html_lines.append(f"<li style='margin-left:1rem'>{_inline_md(content)}</li>")
            continue

        # 普通段落
        if in_ul: html_lines.append("</ul>"); in_ul = False
        if in_ol: html_lines.append("</ol>"); in_ol = False
        html_lines.append(f"<p>{_inline_md(stripped)}</p>")

    # 关闭未关闭的标签
    if in_ul: html_lines.append("</ul>")
    if in_ol: html_lines.append("</ol>")
    if in_table: html_lines.append("</table>")

    return "\n".join(html_lines)


def _inline_md(text: str) -> str:
    """处理 Markdown 内联格式：加粗、行内代码、箭头"""
    t = _escape_html(text)
    # **bold**
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    # `code`
    t = re.sub(r'`(.+?)`', r'<code>\1</code>', t)
    return t


def _escape_html(text: str) -> str:
    """基本 HTML 转义"""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\n", "<br>"))


def _split_coach_output(content: str) -> tuple[str, str]:
    """将 coach 输出拆分为（优化后简历, 教练笔记/修改说明）。

    模板要求 LLM 用 `---` 分隔两部分，但简历内容本身也可能含 `---`。
    策略：
    1. 找「第二部分」「修改说明」「教练笔记」「学习路线图」等标志性关键词
    2. 从该关键词向上找最近的 `---` 行作为分割点
    3. 如果都找不到，fallback 到最后一个 `---`
    """
    lines = content.split("\n")

    # 第二部分的标志性关键词
    part2_markers = [
        "第二部分", "修改说明", "教练笔记", "学习路线图",
        "修改理由", "扩展方向", "学习建议", "面试准备",
    ]

    # 找所有 --- 行的位置
    separator_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---" or stripped == "---\n":
            separator_indices.append(i)

    if not separator_indices:
        # 没有 --- 分隔符，把整个内容当简历
        return content.strip(), ""

    # 找标志性关键词的行号
    marker_line = -1
    for i, line in enumerate(lines):
        for marker in part2_markers:
            if marker in line:
                marker_line = i
                break
        if marker_line >= 0:
            break

    if marker_line >= 0:
        # 从 marker_line 向上找最近的 ---
        best_sep = -1
        for sep_idx in separator_indices:
            if sep_idx < marker_line:
                best_sep = sep_idx
        if best_sep >= 0:
            resume_part = "\n".join(lines[:best_sep]).strip()
            notes_part = "\n".join(lines[best_sep + 1:]).strip()
            return resume_part, notes_part

    # Fallback: 用最后一个 --- 分割（教练笔记通常在后面）
    last_sep = separator_indices[-1]
    # 但如果最后一个 --- 太靠前（前 20% 内容），可能是简历内的分隔线
    # 此时用第一个 ---
    if len(separator_indices) == 1:
        sep = separator_indices[0]
    else:
        # 选择最能均匀分割的那个
        sep = separator_indices[-1]

    resume_part = "\n".join(lines[:sep]).strip()
    notes_part = "\n".join(lines[sep + 1:]).strip()
    return resume_part, notes_part


def _build_eval_carousel(eval_cards: list[str], finished: bool = False, current_page: int = -1) -> str:
    """将多个评估卡片 HTML 拼装为可翻页的卡片轮播。

    由于 Gradio gr.HTML 不执行 <script>，翻页通过 Gradio State + 按钮事件实现。
    这里只负责渲染当前页的那一张卡片。

    Args:
        eval_cards: 每项为 _format_turn_eval 返回的 HTML 片段
        finished: 面试是否已结束
        current_page: 当前显示的页码（-1 表示最后一页）
    """
    if not eval_cards:
        return '<div class="eval-carousel" style="display:flex;align-items:center;justify-content:center;color:#9ca3af;padding:2rem;">面试开始后显示评分卡片</div>'

    all_pages = list(eval_cards)
    if finished:
        all_pages.append(f'''<div class="eval-card-header">
    <span class="eval-card-badge" style="background:linear-gradient(135deg,#059669,#10b981);">完成</span>
    <span class="eval-card-title">面试结束</span>
</div>
<div style="text-align:center;padding:1.5rem 0;color:#374151;font-size:0.95rem;">
    ✅ 全部轮次已完成！请切换到「📊 评估报告」Tab 查看完整报告。
</div>''')

    total = len(all_pages)
    if current_page < 0 or current_page >= total:
        current_page = total - 1

    # 当前卡片内容
    card_html = all_pages[current_page]

    # 圆点
    dots = ""
    for i in range(total):
        active_cls = " active" if i == current_page else ""
        dots += f'<span class="eval-dot{active_cls}"></span>'

    # 导航按钮的 disabled 状态
    prev_disabled = " disabled" if current_page == 0 else ""
    next_disabled = " disabled" if current_page == total - 1 else ""

    return f'''<div class="eval-carousel">
  <div class="eval-card active">{card_html}</div>
  <div class="eval-dots">{dots}</div>
  <div class="eval-nav">
    <span class="eval-nav-indicator">{current_page + 1} / {total}</span>
  </div>
</div>'''


# ============================================================
# 核心交互函数
# ============================================================

async def _handle_upload_to_state(file_obj) -> tuple[dict, str]:
    """上传文件 → FileParser 基础解析 → LLM 结构化 → 返回 (resume_dict, status_msg)"""
    if file_obj is None:
        return {}, "请上传简历文件"

    try:
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        parser = FileParser()
        raw_resume = parser.parse(file_path)
        raw_text = raw_resume.summary  # 原始完整文本

        # 用 LLM 结构化解析（保留原文到 summary）
        config = Config.from_env()
        llm = LLMClient(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=config.openai_model,
        )
        resume_data = await FileParser.llm_structured_parse(raw_text, llm)
        rd_dict = _resume_data_to_dict(resume_data)

        n_edu = len(resume_data.education)
        n_work = len(resume_data.work_experience)
        n_proj = len(resume_data.projects)
        status = f"✅ 解析成功 — {resume_data.name} | {n_edu} 段教育 · {n_work} 段工作 · {n_proj} 个项目 · {len(resume_data.skills)} 项技能"
        return rd_dict, status

    except Exception as e:
        logger.error(f"Upload parse failed: {e}\n{traceback.format_exc()}")
        return {}, f"❌ 解析失败: {e}"


async def _handle_resume_modify_structured(
    session: dict,
    jd: str,
    style: str,
    resume_dict: dict,
) -> tuple[dict, str, str]:
    """从 State 读取简历 → coach.modify → 返回 (session, result_html, status)"""
    session = _ensure_session(session)

    if not resume_dict:
        return session, "", "❌ 请先上传简历"

    resume_data = _dict_to_resume_data(resume_dict)
    resume_text = resume_data.to_text()

    if not resume_data.name.strip() and not resume_text.strip():
        return session, "", "❌ 简历信息为空，请先上传简历"

    if not jd.strip():
        return session, "", "❌ 请输入目标职位 JD"

    # 映射风格
    style_map = {
        "🎯 专业打磨 (零虚构)": "rigorous",
        "🚀 技术深化 (进阶表达)": "embellished",
        "🌟 成长路线 (前沿扩展)": "wild",
    }
    style_key = style_map.get(style, "rigorous")

    config = Config.from_env()
    llm = LLMClient(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        model=config.openai_model,
    )
    event_bus = EventBus()
    coach = CareerCoachAgent(llm=llm)
    coach.bind_event_bus(event_bus)

    try:
        # 调 coach.modify（返回：优化简历 + --- + 教练笔记）
        results = await coach.modify(resume_data, jd, [style_key])
        content = results.get(style_key, "")

        # 分离简历和教练笔记
        # 模板要求 LLM 输出「第一部分：简历」+ `---` +「第二部分：教练笔记/修改说明」
        # 但 LLM 简历内容本身也可能包含 `---`（Markdown 水平线），所以不能简单 split
        # 策略：找到「第二部分」标志性关键词前最近的 `---` 行
        resume_part, notes_part = _split_coach_output(content)

        # 直接渲染两部分 Markdown 为 HTML 双栏
        resume_html = _md_to_html_simple(resume_part)
        notes_html = _md_to_html_simple(notes_part) if notes_part else '<div style="color:#9ca3af;text-align:center;padding:2rem;">暂无教练笔记</div>'

        result_html = f'''<div class="result-split">
  <div class="result-left">
    <div class="result-left-title">📄 优化后简历</div>
    <div class="coach-content">{resume_html}</div>
  </div>
  <div class="result-divider"></div>
  <div class="result-right">
    <div class="result-right-title">📝 教练笔记 / 修改说明</div>
    <div class="coach-content">{notes_html}</div>
  </div>
</div>'''

        # 更新 session
        session["resume_data"] = resume_data.model_dump()
        session["resume_text"] = resume_text
        session["jd"] = jd
        session["modified_resumes"] = {style_key: content}
        session["selected_style"] = style_key

        return session, result_html, "✅ 优化完成"

    except Exception as e:
        logger.error(f"Resume modification failed: {e}\n{traceback.format_exc()}")
        return session, "", f"❌ 优化失败: {e}"


# ============================================================
# 面试相关函数（不变）
# ============================================================

async def _run_interview_ai(
    session: dict,
    total_rounds: int,
    chatbot: list,
    eval_md: str,
    eval_page: dict,
):
    """AI 模式面试 — generator，逐步 yield (session, chatbot, eval_html, eval_page)"""
    session = _ensure_session(session)
    config = Config.from_env()
    components = _build_session_components(config)

    interviewer = components["interviewer"]
    interviewee = components["interviewee"]
    evaluator = components["evaluator"]
    privileged = components["privileged"]
    episodic = components["episodic"]
    state = components["state"]

    resume_data_dict = session.get("resume_data", {})
    jd = session.get("jd", "")
    modified = session.get("modified_resumes", {})
    selected_style = session.get("selected_style", "rigorous")

    privileged.set("original_resume", resume_data_dict)
    privileged.set("job_description", jd)
    if modified:
        privileged.set("modified_resume", modified.get(selected_style, ""))

    state.original_resume = resume_data_dict
    state.job_description = jd
    state.interview_rounds = int(total_rounds)

    # 面试开始前：生成简历差异摘要（一次性，后续每轮复用）
    await evaluator.generate_resume_diff_summary()

    all_evals: list[str] = []
    last_answer: str | None = None

    for round_num in range(1, int(total_rounds) + 1):
        state.current_round = round_num

        question = await interviewer.ask(round_num, int(total_rounds), last_answer)
        chatbot.append({"role": "assistant", "content": ""})
        for char in question:
            chatbot[-1]["content"] += char
            yield session, chatbot, eval_md, eval_page
            await asyncio.sleep(0.01)

        episodic.append(DialogueTurn(
            round_number=round_num, role="interviewer", content=question,
        ))

        answer = await interviewee.answer(question, round_num)
        last_answer = answer
        chatbot.append({"role": "user", "content": ""})
        for char in answer:
            chatbot[-1]["content"] += char
            yield session, chatbot, eval_md, eval_page
            await asyncio.sleep(0.01)

        episodic.append(DialogueTurn(
            round_number=round_num, role="interviewee", content=answer,
        ))

        state.conversation.append({"role": "interviewer", "content": question})
        state.conversation.append({"role": "interviewee", "content": answer})

        turn_eval = await evaluator.evaluate_turn(
            question=question,
            answer=answer,
            round_number=round_num,
            total_rounds=int(total_rounds),
        )
        state.turn_evaluations.append(turn_eval)

        # 更新面试官覆盖度追踪
        for dim in turn_eval.get("covered_dimensions", []):
            interviewer.mark_coverage(dim)

        all_evals.append(_format_turn_eval(turn_eval))
        eval_page = {"cards": list(all_evals), "page": len(all_evals) - 1, "finished": False}
        eval_md = _build_eval_carousel(all_evals, finished=False, current_page=len(all_evals) - 1)
        yield session, chatbot, eval_md, eval_page

    final_report = await evaluator.generate_final_report()
    state.final_evaluation = final_report

    # 写回 session（关键：这样 Gradio gr.State 才能保存）
    session["final_report"] = final_report
    session["state"] = {
        "original_resume": state.original_resume,
        "job_description": state.job_description,
        "conversation": state.conversation,
        "turn_evaluations": state.turn_evaluations,
        "final_evaluation": state.final_evaluation,
        "current_round": state.current_round,
        "selected_resume_style": state.selected_resume_style,
        "metadata": state.metadata,
    }

    eval_page = {"cards": list(all_evals), "page": len(all_evals), "finished": True}
    eval_md = _build_eval_carousel(all_evals, finished=True)
    yield session, chatbot, eval_md, eval_page


async def _run_interview_human_step(
    session: dict,
    user_message: str,
    chatbot: list,
    eval_md: str,
    total_rounds: int,
):
    """人类模式 — 处理用户发送的一条消息"""
    session = _ensure_session(session)
    if "components" not in session:
        config = Config.from_env()
        components = _build_session_components(config)

        resume_data_dict = session.get("resume_data", {})
        jd = session.get("jd", "")
        modified = session.get("modified_resumes", {})
        selected_style = session.get("selected_style", "rigorous")

        components["privileged"].set("original_resume", resume_data_dict)
        components["privileged"].set("job_description", jd)
        if modified:
            components["privileged"].set("modified_resume", modified.get(selected_style, ""))

        components["state"].original_resume = resume_data_dict
        components["state"].job_description = jd
        components["state"].interview_rounds = int(total_rounds)

        session["components"] = "initialized"
        session["_comp"] = components
        session["human_round"] = 0
        session["human_evals"] = []
        session["needs_question"] = True

        # 面试开始前：生成简历差异摘要
        await components["evaluator"].generate_resume_diff_summary()

    components = session["_comp"]
    interviewer = components["interviewer"]
    evaluator = components["evaluator"]
    episodic = components["episodic"]
    state = components["state"]

    round_num = session["human_round"]

    if session.get("needs_question", True):
        round_num += 1
        session["human_round"] = round_num
        state.current_round = round_num

        last_answer = user_message if round_num > 1 else None

        if round_num > 1 and user_message.strip():
            chatbot.append({"role": "user", "content": user_message})

            episodic.append(DialogueTurn(
                round_number=round_num - 1, role="interviewee", content=user_message,
            ))
            state.conversation.append({"role": "interviewee", "content": user_message})

            prev_q = session.get("last_question", "")
            if prev_q:
                turn_eval = await evaluator.evaluate_turn(
                    question=prev_q,
                    answer=user_message,
                    round_number=round_num - 1,
                    total_rounds=int(total_rounds),
                )
                state.turn_evaluations.append(turn_eval)
                # 更新面试官覆盖度追踪
                for dim in turn_eval.get("covered_dimensions", []):
                    interviewer.mark_coverage(dim)
                session["human_evals"].append(_format_turn_eval(turn_eval))
                eval_md = _build_eval_carousel(session["human_evals"], finished=False)

        if round_num > int(total_rounds):
            final_report = await evaluator.generate_final_report()
            session["final_report"] = final_report
            eval_md = _build_eval_carousel(session["human_evals"], finished=True)
            return session, chatbot, eval_md

        question = await interviewer.ask(round_num, int(total_rounds), last_answer)
        chatbot.append({"role": "assistant", "content": question})
        episodic.append(DialogueTurn(
            round_number=round_num, role="interviewer", content=question,
        ))
        state.conversation.append({"role": "interviewer", "content": question})
        session["last_question"] = question
        session["needs_question"] = False

    return session, chatbot, eval_md


# ============================================================
# Gradio 应用工厂
# ============================================================

def create_app() -> gr.Blocks:
    """创建 Gradio 应用"""

    with gr.Blocks(
        title="MOSAIC — 职业发展教练",
    ) as app:
        # 全局 session state
        session = gr.State(value=None)

        # 简历数据 State（dict 格式，给 @gr.render 用）
        resume_state = gr.State(value={})

        gr.HTML("""
        <div class="header-banner">
            <h1>🎯 MOSAIC — 职业发展教练</h1>
            <div class="subtitle"><b>M</b>ulti-agent <b>O</b>rchestrated <b>S</b>imulation for <b>A</b>daptive <b>I</b>nterview <b>C</b>oaching</div>
            <div class="features">
                <span class="feat">📝 智能简历优化</span>
                <span class="feat">🎤 多轮模拟面试</span>
                <span class="feat">📊 深度评估报告</span>
            </div>
        </div>
        """)

        # ============ Tab 1: 简历优化 ============
        with gr.Tab("📝 简历优化"):

            # ── 区块 1: 上传操作栏 ──
            gr.HTML('<div class="tip-box">💡 上传简历（PDF / DOCX / TXT），系统将自动解析为结构化信息。确认后填写 JD、选择风格，点击「开始优化」。</div>')
            with gr.Row(equal_height=False):
                resume_file = gr.File(
                    label="📎 上传简历",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                    type="filepath",
                    scale=1,
                )
                jd_input = gr.Textbox(
                    label="📋 目标职位 JD",
                    placeholder="粘贴岗位描述...",
                    lines=5,
                    scale=2,
                )
            with gr.Row(equal_height=True):
                style_radio = gr.Radio(
                    label="🎨 优化风格",
                    choices=[
                        "🎯 专业打磨 (零虚构)",
                        "🚀 技术深化 (进阶表达)",
                        "🌟 成长路线 (前沿扩展)",
                    ],
                    value="🎯 专业打磨 (零虚构)",
                    scale=2,
                )
                modify_btn = gr.Button(
                    "🚀 开始优化", variant="primary", size="lg", scale=1,
                    min_width=180, elem_classes="optimize-btn",
                )

            # 状态提示
            parse_status = gr.Markdown(value="*上传简历后，结构化信息将自动填充到下方表单*")

            # ── 区块 2: 动态结构化表单（@gr.render） ──
            @gr.render(inputs=[resume_state])
            def render_resume_form(rd_dict: dict):
                """根据 resume_state 动态渲染表单。"""
                if not rd_dict:
                    gr.HTML('<div style="color:#9ca3af;text-align:center;padding:2rem;border:2px dashed #e5e7eb;border-radius:12px;">📄 上传简历后，解析结果将在此显示为可编辑表单</div>')
                    return

                rd = _dict_to_resume_data(rd_dict)

                with gr.Column(variant="panel"):
                    gr.HTML('<div class="section-title">📋 简历信息（可编辑）</div>')

                    # 个人信息
                    with gr.Group(elem_classes="resume-form-section"):
                        gr.HTML('<div class="section-label">👤 个人信息</div>')
                        with gr.Row():
                            gr.Textbox(value=rd.name, label="姓名", interactive=True, scale=1)
                            gr.Textbox(value=rd.phone, label="手机", interactive=True, scale=1)
                            gr.Textbox(value=rd.email, label="邮箱", interactive=True, scale=1)
                            gr.Textbox(value=rd.target_position, label="求职意向", interactive=True, scale=1)

                    # 教育经历
                    if rd.education:
                        with gr.Group(elem_classes="resume-form-section"):
                            gr.HTML('<div class="section-label">🎓 教育经历</div>')
                            for i, e in enumerate(rd.education):
                                label = f"{e.school}" if e.school else f"教育经历 {i+1}"
                                if e.degree:
                                    label += f" · {e.degree}"
                                if e.major:
                                    label += f" · {e.major}"
                                with gr.Accordion(label=label, open=(i == 0)):
                                    with gr.Row():
                                        gr.Textbox(value=e.school, label="学校", interactive=True, scale=2)
                                        gr.Textbox(value=e.degree, label="学位", interactive=True, scale=1)
                                        gr.Textbox(value=e.major, label="专业", interactive=True, scale=1)
                                    with gr.Row():
                                        gr.Textbox(value=e.start_year, label="入学年份", interactive=True, scale=1)
                                        gr.Textbox(value=e.end_year, label="毕业年份", interactive=True, scale=1)
                                        gr.Textbox(value=e.gpa, label="GPA", interactive=True, scale=1)
                                    if e.highlights:
                                        gr.Textbox(value="\n".join(e.highlights), label="荣誉/奖项", lines=2, interactive=True)

                    # 工作/实习经历
                    if rd.work_experience:
                        with gr.Group(elem_classes="resume-form-section"):
                            gr.HTML('<div class="section-label">🏢 工作/实习经历</div>')
                            for i, w in enumerate(rd.work_experience):
                                label = f"{w.company}" if w.company else f"工作经历 {i+1}"
                                if w.title:
                                    label += f" · {w.title}"
                                if w.start_date:
                                    label += f" ({w.start_date}~{w.end_date})"
                                with gr.Accordion(label=label, open=(i == 0)):
                                    with gr.Row():
                                        gr.Textbox(value=w.company, label="公司", interactive=True, scale=2)
                                        gr.Textbox(value=w.title, label="职位", interactive=True, scale=1)
                                    with gr.Row():
                                        gr.Textbox(value=w.start_date, label="开始时间", interactive=True, scale=1)
                                        gr.Textbox(value=w.end_date, label="结束时间", interactive=True, scale=1)
                                    gr.Textbox(value=w.description, label="工作描述", lines=2, interactive=True)
                                    if w.achievements:
                                        gr.Textbox(value="\n".join(w.achievements), label="工作成果", lines=3, interactive=True)
                                    if w.tech_stack:
                                        gr.Textbox(value=", ".join(w.tech_stack), label="技术栈", interactive=True)

                    # 项目经历
                    if rd.projects:
                        with gr.Group(elem_classes="resume-form-section"):
                            gr.HTML('<div class="section-label">📁 项目经历</div>')
                            for i, p in enumerate(rd.projects):
                                label = f"{p.name}" if p.name else f"项目 {i+1}"
                                if p.role:
                                    label += f" · {p.role}"
                                with gr.Accordion(label=label, open=(i == 0)):
                                    with gr.Row():
                                        gr.Textbox(value=p.name, label="项目名称", interactive=True, scale=2)
                                        gr.Textbox(value=p.role, label="角色", interactive=True, scale=1)
                                    gr.Textbox(value=p.description, label="项目描述", lines=2, interactive=True)
                                    if p.achievements:
                                        gr.Textbox(value="\n".join(p.achievements), label="项目成果", lines=3, interactive=True)
                                    if p.tech_stack:
                                        gr.Textbox(value=", ".join(p.tech_stack), label="技术栈", interactive=True)

                    # 技能
                    if rd.skills:
                        with gr.Group(elem_classes="resume-form-section"):
                            gr.HTML('<div class="section-label">🛠️ 技能</div>')
                            gr.Textbox(value=", ".join(rd.skills), label="技能（逗号分隔）", interactive=True)

            # ── 区块 3: 优化结果（单一 HTML 双栏布局） ──
            with gr.Group(elem_classes="result-area"):
                gr.HTML('<div class="section-title">✨ 优化结果</div>')
                result_display = gr.HTML(
                    value='<div class="result-split"><div class="result-left"><div class="result-left-title">📄 优化后简历</div><div style="color:#9ca3af;text-align:center;padding:2rem;">点击「开始优化」后显示优化后的完整简历</div></div><div class="result-divider"></div><div class="result-right"><div class="result-right-title">📝 教练笔记 / 修改说明</div><div style="color:#9ca3af;text-align:center;padding:2rem;">教练笔记将在此显示</div></div></div>',
                )

            # ── 事件绑定 ──

            # 上传 → 先显示"解析中" → LLM 解析 → 更新 resume_state + 状态
            resume_file.change(
                fn=lambda _: "⏳ 正在解析简历，请稍候…（LLM 结构化解析中）",
                inputs=[resume_file],
                outputs=[parse_status],
            ).then(
                fn=_handle_upload_to_state,
                inputs=[resume_file],
                outputs=[resume_state, parse_status],
            )

            # 开始优化
            modify_event = modify_btn.click(
                fn=_handle_resume_modify_structured,
                inputs=[session, jd_input, style_radio, resume_state],
                outputs=[session, result_display, parse_status],
            )
            # 切换风格时取消正在进行的优化（用户改主意了）
            style_radio.change(
                fn=None, inputs=[], outputs=[],
                cancels=[modify_event],
            )

        # ============ Tab 2: 模拟面试 ============
        with gr.Tab("🎤 模拟面试"):
            gr.HTML('<div class="tip-box">🎯 选择面试轮次和回答方式后点击「开始面试」。AI 托管模式下系统自动问答；亲自作答模式下你回答面试官的提问。</div>')
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.HTML('<div class="section-title">⚙️ 面试设置</div>')
                    rounds_slider = gr.Slider(
                        label="面试轮次",
                        minimum=1,
                        maximum=15,
                        value=5,
                        step=1,
                    )
                    mode_radio = gr.Radio(
                        label="回答方式",
                        choices=["🤖 AI 托管", "✍️ 亲自作答"],
                        value="🤖 AI 托管",
                    )
                    start_btn = gr.Button(
                        "▶️ 开始面试", variant="primary", size="lg",
                        elem_classes="interview-start-btn",
                    )
                    stop_btn = gr.Button(
                        "⏹️ 停止面试", variant="stop", size="lg",
                        visible=True,
                    )

                    # 输入区（默认隐藏，选亲自作答时显示）
                    human_input_group = gr.Group(visible=False)
                    with human_input_group:
                        gr.HTML('<div class="section-title" style="margin-top:0.5rem;">💬 你的回答</div>')
                        human_input = gr.Textbox(
                            label="输入回答",
                            placeholder="面试官提问后，在这里输入你的回答...",
                            lines=5,
                            interactive=True,
                            show_label=False,
                        )
                        send_btn = gr.Button("📤 发送回答", variant="primary", size="sm")

                with gr.Column(scale=2, variant="panel"):
                    chatbot = gr.Chatbot(
                        label="💬 面试对话",
                        height=480,
                    )
                    gr.HTML('<div class="section-title" style="margin-top:0.5rem;">📊 逐轮评分</div>')
                    eval_display = gr.HTML(
                        value='<div class="eval-carousel" style="display:flex;align-items:center;justify-content:center;color:#9ca3af;padding:2rem;">面试开始后显示评分卡片</div>',
                    )
                    with gr.Row():
                        eval_prev_btn = gr.Button("◀ 上一轮", size="sm", scale=1, min_width=80)
                        eval_next_btn = gr.Button("下一轮 ▶", size="sm", scale=1, min_width=80)

            # 评分翻页 State: {"cards": [...], "page": int, "finished": bool}
            eval_page_state = gr.State(value={"cards": [], "page": 0, "finished": False})

            # 切换模式时 显示/隐藏 输入区
            def toggle_human_input(mode):
                return gr.update(visible="亲自" in mode)

            mode_radio.change(
                fn=toggle_human_input,
                inputs=[mode_radio],
                outputs=[human_input_group],
            )

            async def start_ai_interview(sess, rounds, mode, chat, ep):
                sess = _ensure_session(sess)
                if not sess.get("resume_data"):
                    chat = [{"role": "assistant", "content": "⚠️ 请先在「简历优化」Tab 完成简历优化！"}]
                    yield sess, chat, '<div class="eval-carousel" style="display:flex;align-items:center;justify-content:center;color:#b91c1c;padding:2rem;">请先完成简历优化</div>', ep
                    return

                chat = []
                chat.append({"role": "assistant", "content": f"🎤 面试开始！共 {int(rounds)} 轮，让我们开始吧。\n\n---"})
                eval_text = ""
                ep = {"cards": [], "page": 0, "finished": False}

                if "AI" in mode:
                    async for updated_sess, updated_chat, updated_eval, updated_ep in _run_interview_ai(
                        sess, int(rounds), chat, eval_text, ep,
                    ):
                        yield updated_sess, updated_chat, updated_eval, updated_ep
                else:
                    sess["needs_question"] = True
                    sess, chat, eval_text = await _run_interview_human_step(
                        sess, "", chat, eval_text, int(rounds),
                    )
                    yield sess, chat, eval_text, ep

            # 开始面试
            interview_event = start_btn.click(
                fn=start_ai_interview,
                inputs=[session, rounds_slider, mode_radio, chatbot, eval_page_state],
                outputs=[session, chatbot, eval_display, eval_page_state],
            )

            # ⏹️ 停止按钮 → 取消正在进行的面试流程
            stop_btn.click(
                fn=None, inputs=[], outputs=[],
                cancels=[interview_event],
            )

            async def send_human_answer(sess, msg, chat, eval_text, rounds, ep):
                sess = _ensure_session(sess)
                if not msg.strip():
                    return sess, chat, eval_text, "", ep
                sess["needs_question"] = True
                sess, chat, eval_text = await _run_interview_human_step(
                    sess, msg, chat, eval_text, int(rounds),
                )
                # 更新翻页 state
                ep = {"cards": sess.get("human_evals", []), "page": len(sess.get("human_evals", [])) - 1, "finished": False}
                return sess, chat, eval_text, "", ep

            send_btn.click(
                fn=send_human_answer,
                inputs=[session, human_input, chatbot, eval_display, rounds_slider, eval_page_state],
                outputs=[session, chatbot, eval_display, human_input, eval_page_state],
            )

            # 翻页按钮
            def eval_prev(ep):
                cards = ep.get("cards", [])
                page = ep.get("page", 0)
                finished = ep.get("finished", False)
                page = max(0, page - 1)
                ep["page"] = page
                html = _build_eval_carousel(cards, finished=finished, current_page=page)
                return html, ep

            def eval_next(ep):
                cards = ep.get("cards", [])
                page = ep.get("page", 0)
                finished = ep.get("finished", False)
                total = len(cards) + (1 if finished else 0)
                page = min(total - 1, page + 1)
                ep["page"] = page
                html = _build_eval_carousel(cards, finished=finished, current_page=page)
                return html, ep

            eval_prev_btn.click(
                fn=eval_prev,
                inputs=[eval_page_state],
                outputs=[eval_display, eval_page_state],
            )
            eval_next_btn.click(
                fn=eval_next,
                inputs=[eval_page_state],
                outputs=[eval_display, eval_page_state],
            )

        # ============ Tab 3: 评估报告 ============
        with gr.Tab("📊 评估报告"):
            gr.HTML("""
            <div class="tip-box">
                📊 面试结束后点击下方按钮加载完整评估报告。报告包含五维评分、优势分析、改进建议和学习路线图，支持下载留存。
            </div>
            """)
            with gr.Column(variant="panel"):
                refresh_btn = gr.Button("🔄 加载报告", variant="primary")
                report_display = gr.Markdown(
                    value="*面试结束后点击「加载报告」查看完整评估*",
                )
            with gr.Column(variant="panel"):
                gr.HTML('<div class="section-title">📥 报告下载</div>')
                report_file = gr.File(label="📥 下载报告", visible=False)

            async def load_report(sess):
                sess = _ensure_session(sess)
                final_report = sess.get("final_report")
                if not final_report:
                    return "⚠️ 尚未完成面试，无法生成报告。请先在「模拟面试」Tab 完成面试。", gr.update(visible=False)

                report_md = _format_final_report(final_report)

                try:
                    state_data = sess.get("state", {})
                    if not state_data:
                        logger.warning("No state data in session, showing report without file download")
                        return report_md, gr.update(visible=False)

                    state = AgentState(
                        original_resume=state_data.get("original_resume", {}),
                        job_description=state_data.get("job_description", ""),
                        conversation=state_data.get("conversation", []),
                        turn_evaluations=state_data.get("turn_evaluations", []),
                        final_evaluation=state_data.get("final_evaluation", {}),
                        current_round=state_data.get("current_round", 0),
                        selected_resume_style=state_data.get("selected_resume_style", ""),
                        metadata=state_data.get("metadata", {}),
                    )

                    modified = sess.get("modified_resumes", {})
                    selected_style = sess.get("selected_style", "")
                    if selected_style and selected_style in modified:
                        state.selected_resume = {"content": modified[selected_style]}

                    filepath = generate_report(state, final_report, "reports")
                    logger.info(f"Report generated: {filepath}")
                    return report_md, gr.update(value=filepath, visible=True)
                except Exception as e:
                    logger.error(f"Report save failed: {e}\n{traceback.format_exc()}")
                    return report_md, gr.update(visible=False)

            refresh_btn.click(
                fn=load_report,
                inputs=[session],
                outputs=[report_display, report_file],
            )

    return app


# ============================================================
# 直接运行入口
# ============================================================

if __name__ == "__main__":
    app = create_app()
    app.queue(default_concurrency_limit=2).launch(server_name="0.0.0.0", server_port=7860, **LAUNCH_KWARGS)
