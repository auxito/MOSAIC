"""
Markdown 报告生成器。
将面试结果转换为结构化的 Markdown 报告。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mosaic.core.agent import AgentState


def generate_report(
    state: AgentState,
    final_evaluation: dict[str, Any],
    output_dir: str = "reports",
) -> str:
    """
    生成面试评估报告。

    Returns:
        报告文件路径
    """
    sections: list[str] = []

    # 标题
    name = state.original_resume.get("name", "候选人")
    sections.append(f"# 面试模拟报告 — {name}")
    sections.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 基本信息
    sections.append("\n## 基本信息")
    sections.append(f"- **候选人**: {name}")
    if state.job_description:
        sections.append(f"- **目标职位**: {state.job_description[:100]}...")
    sections.append(f"- **简历风格**: {state.selected_resume_style or '未指定'}")
    sections.append(f"- **面试轮次**: {state.current_round}")

    # 综合评分
    if final_evaluation and "error" not in final_evaluation:
        sections.append("\n## 综合评分")

        overall = final_evaluation.get("overall_score", "N/A")
        sections.append(f"\n**综合评分: {overall}/10**")

        dimensions = final_evaluation.get("dimension_scores", {})
        if dimensions:
            sections.append("\n| 维度 | 评分 |")
            sections.append("|------|------|")
            for dim, score in dimensions.items():
                sections.append(f"| {dim} | {score}/5 |")

        # 优势
        strengths = final_evaluation.get("strengths", [])
        if strengths:
            sections.append("\n### 优势")
            for s in strengths:
                sections.append(f"- {s}")

        # 改进建议
        improvements = final_evaluation.get("improvements", [])
        if improvements:
            sections.append("\n### 改进建议")
            for i in improvements:
                sections.append(f"- {i}")

        # 一致性评估
        consistency = final_evaluation.get("consistency_assessment", "")
        if consistency:
            sections.append(f"\n### 一致性评估\n{consistency}")

        # 录用建议
        recommendation = final_evaluation.get("recommendation", "")
        if recommendation:
            sections.append(f"\n### 录用建议\n**{recommendation}**")

        # 详细评语
        comments = final_evaluation.get("detailed_comments", "")
        if comments:
            sections.append(f"\n### 详细评语\n{comments}")

    # 各轮评分详情
    if state.turn_evaluations:
        sections.append("\n## 各轮评分详情")
        for te in state.turn_evaluations:
            rn = te.get("round_number", "?")
            scores = te.get("scores", {})
            comment = te.get("comment", "")
            sections.append(
                f"\n**第 {rn} 轮**: "
                f"质量={scores.get('answer_quality', '?')}, "
                f"技术={scores.get('technical_accuracy', '?')}, "
                f"沟通={scores.get('communication', '?')}, "
                f"可信={scores.get('credibility', '?')}"
            )
            if comment:
                sections.append(f"> {comment}")
            if te.get("exaggeration_detected"):
                sections.append(f"> ⚠️ 夸大检测: {te.get('exaggeration_details', '')}")

    # 矛盾记录
    if state.contradictions:
        sections.append("\n## 矛盾记录")
        for c in state.contradictions:
            sections.append(
                f"- [第{c.get('detected_at_round', '?')}轮] {c.get('description', '')}"
            )

    # 语义事实
    if state.semantic_facts:
        sections.append("\n## 抽取的事实")
        for f in state.semantic_facts:
            sections.append(
                f"- [{f.get('category', 'other')}] "
                f"(第{f.get('round_number', '?')}轮) {f.get('content', '')}"
            )

    # 元数据
    elapsed = state.metadata.get("elapsed_time", 0)
    if elapsed:
        sections.append(f"\n---\n*面试总耗时: {elapsed:.1f}秒*")

    # 写入文件
    report_content = "\n".join(sections)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"interview_report_{name}_{timestamp}.md"
    filepath = output_path / filename

    filepath.write_text(report_content, encoding="utf-8")
    return str(filepath)
