"""
完整面试工作流 — 阶段处理器。
FULL_INTERVIEW: INIT → RESUME_INPUT → RESUME_MODIFY → INTERVIEW_LOOP → EVALUATION → REPORT → COMPLETE
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, IntPrompt

from mosaic.core.memory.episodic import DialogueTurn
from mosaic.core.workflow import Phase
from mosaic.resume.structured_input import create_sample_resume

if TYPE_CHECKING:
    from mosaic.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)
console = Console()


async def handle_init(orch: Orchestrator) -> None:
    """初始化阶段"""
    console.print(Panel(
        "[bold]MOSAIC[/bold] — Multi-agent Orchestrated Simulation for Adaptive Interview Coaching\n"
        f"工作流: {orch.workflow.name}\n"
        f"面试轮次: {orch.workflow.interview_rounds}",
        title="欢迎",
        border_style="green",
    ))


async def handle_resume_input(orch: Orchestrator) -> None:
    """简历输入阶段"""
    console.print("\n[bold]📋 简历输入[/bold]")
    choice = Prompt.ask(
        "选择输入方式",
        choices=["sample", "text"],
        default="sample",
    )

    if choice == "sample":
        resume = create_sample_resume()
        console.print("[green]已加载示例简历[/green]")
    else:
        console.print("请输入简历内容（输入两次空行结束）:")
        lines: list[str] = []
        empty = 0
        while True:
            try:
                line = input()
                if not line:
                    empty += 1
                    if empty >= 2:
                        break
                    lines.append("")
                else:
                    empty = 0
                    lines.append(line)
            except EOFError:
                break
        from mosaic.resume.structured_input import resume_from_text
        resume = resume_from_text("\n".join(lines))

    orch.state.original_resume = resume.model_dump()

    # JD 输入
    console.print("\n[bold]📝 职位描述 (JD)[/bold]")
    jd = Prompt.ask(
        "输入目标职位描述（或按回车使用默认）",
        default="高级后端工程师，要求3年以上Python经验，熟悉微服务架构、Redis、Kafka，有推荐系统经验优先。",
    )
    orch.state.job_description = jd

    # 设置特权记忆
    orch.privileged.set("original_resume", resume.to_text())
    orch.privileged.set("job_description", jd)

    # 简历解析
    parser = orch.get_agent("resume_parser")
    await parser.parse(resume)


async def handle_resume_modify(orch: Orchestrator) -> None:
    """简历优化阶段（职业发展教练）"""
    console.print("\n[bold]✏️ 简历优化（职业发展教练）[/bold]")

    coach = orch.get_agent("career_coach")
    from mosaic.resume.schema import ResumeData
    resume = ResumeData(**orch.state.original_resume)

    styles = orch.workflow.resume_styles
    console.print(f"正在生成 {len(styles)} 种风格的简历...")

    results = await coach.modify(
        resume=resume,
        job_description=orch.state.job_description,
        styles=styles,
    )
    orch.state.modified_resumes = results

    # 展示并选择
    style_labels = {
        "rigorous": "专业打磨版",
        "embellished": "技术深化版",
        "wild": "成长路线版",
    }

    for style, content in results.items():
        label = style_labels.get(style, style)

        # 分离简历和教练笔记
        parts = content.split("---", 1)
        resume_part = parts[0].strip()
        coach_notes = parts[1].strip() if len(parts) > 1 else ""

        console.print(Panel(
            Markdown(resume_part[:500] + "..." if len(resume_part) > 500 else resume_part),
            title=f"[bold]{label}[/bold]",
            border_style="cyan",
        ))

        if coach_notes:
            console.print(Panel(
                Markdown(coach_notes[:300] + "..." if len(coach_notes) > 300 else coach_notes),
                title=f"[bold]📝 教练笔记 — {label}[/bold]",
                border_style="yellow",
            ))

    selected = Prompt.ask(
        "选择面试用简历风格",
        choices=list(results.keys()),
        default="embellished",
    )

    orch.state.selected_resume_style = selected
    orch.state.selected_resume = {"content": results[selected]}

    # 更新特权记忆
    orch.privileged.set("modified_resume", results[selected])


async def handle_interview_loop(orch: Orchestrator) -> None:
    """面试循环阶段"""
    console.print("\n[bold]🎤 面试开始[/bold]")

    interviewer = orch.get_agent("interviewer")
    interviewee = orch.get_agent("interviewee")
    evaluator = orch.get_agent("evaluator")
    memory_mgr = orch.get_agent("memory_manager")

    total = orch.state.interview_rounds
    last_answer = None

    for round_num in range(1, total + 1):
        orch.state.current_round = round_num
        console.rule(f"[bold]第 {round_num}/{total} 轮[/bold]")

        # 1. 面试官提问
        question = await interviewer.ask(
            current_round=round_num,
            total_rounds=total,
            last_answer=last_answer,
        )

        console.print(Panel(
            Markdown(question),
            title="[bold blue]面试官[/bold blue]",
            border_style="blue",
        ))

        # 记录到 episodic memory
        orch.episodic.append(DialogueTurn(
            round_number=round_num,
            role="interviewer",
            content=question,
        ))

        # 2. 面试者回答
        answer = await interviewee.answer(
            question=question,
            current_round=round_num,
        )

        console.print(Panel(
            Markdown(answer),
            title="[bold green]候选人[/bold green]",
            border_style="green",
        ))

        orch.episodic.append(DialogueTurn(
            round_number=round_num,
            role="interviewee",
            content=answer,
        ))

        # 3. 评估（后台）
        turn_eval = await evaluator.evaluate_turn(
            question=question,
            answer=answer,
            round_number=round_num,
            total_rounds=total,
        )
        orch.state.turn_evaluations.append(turn_eval)

        # 显示评分和改进建议
        scores = turn_eval.get("scores", {})
        console.print(
            f"  [dim]评分: 质量={scores.get('answer_quality', '?')} "
            f"技术={scores.get('technical_accuracy', '?')} "
            f"沟通={scores.get('communication', '?')} "
            f"可信={scores.get('credibility', '?')}[/dim]"
        )

        # 显示改进建议
        highlights = turn_eval.get("highlights", "")
        if highlights:
            console.print(f"  [green]✨ 亮点: {highlights}[/green]")

        suggestions = turn_eval.get("improvement_suggestions", "")
        if suggestions:
            console.print(f"  [yellow]💡 改进建议: {suggestions}[/yellow]")

        knowledge_gaps = turn_eval.get("knowledge_gaps", "")
        if knowledge_gaps:
            console.print(f"  [red]📚 知识盲区: {knowledge_gaps}[/red]")

        last_answer = answer

    console.print("\n[bold]面试结束！[/bold]")


async def handle_evaluation(orch: Orchestrator) -> None:
    """评估阶段"""
    console.print("\n[bold]📊 生成评估报告...[/bold]")

    # 确保有最终摘要
    memory_mgr = orch.get_agent("memory_manager")
    await memory_mgr.handle({"action": "summarize"})

    evaluator = orch.get_agent("evaluator")
    final_eval = await evaluator.generate_final_report()
    orch.state.final_evaluation = final_eval

    # 同步语义记忆到 state
    orch.state.semantic_facts = [
        {
            "content": f.content,
            "category": f.category,
            "round_number": f.round_number,
            "confidence": f.confidence.name,
        }
        for f in orch.semantic.facts
    ]
    orch.state.contradictions = [
        {
            "description": c.description,
            "severity": c.severity,
            "detected_at_round": c.detected_at_round,
        }
        for c in orch.semantic.contradictions
    ]


async def handle_report(orch: Orchestrator) -> None:
    """报告生成阶段"""
    from mosaic.output.report import generate_report

    report_path = generate_report(
        state=orch.state,
        final_evaluation=orch.state.final_evaluation,
        output_dir=orch.state.metadata.get("report_dir", "reports"),
    )

    console.print(Panel(
        f"报告已保存至: {report_path}",
        title="[bold green]完成[/bold green]",
        border_style="green",
    ))

    # 显示摘要
    eval_data = orch.state.final_evaluation
    if eval_data and "error" not in eval_data:
        overall = eval_data.get("overall_score", "N/A")
        rec = eval_data.get("recommendation", "N/A")
        console.print(f"\n  综合评分: [bold]{overall}/10[/bold]")
        console.print(f"  录用建议: [bold]{rec}[/bold]")

        # 维度评分
        dims = eval_data.get("dimension_scores", {})
        if dims:
            console.print("\n  维度评分:")
            for dim, score in dims.items():
                bar = "█" * int(score) + "░" * (5 - int(score))
                console.print(f"    {dim}: {bar} {score}/5")

        # 一致性
        consistency = eval_data.get("consistency_assessment", "")
        if consistency:
            console.print(f"\n  一致性评估: {consistency}")

        # 教练建议
        coaching_notes = eval_data.get("coaching_notes", "")
        if coaching_notes:
            console.print(Panel(
                Markdown(coaching_notes),
                title="[bold yellow]🎯 教练建议[/bold yellow]",
                border_style="yellow",
            ))

        # 学习路线图
        learning_roadmap = eval_data.get("learning_roadmap", [])
        if learning_roadmap:
            roadmap_lines = []
            for item in learning_roadmap:
                area = item.get("area", "未知")
                current = item.get("current_level", "")
                target = item.get("target_level", "")
                timeline = item.get("timeline", "")
                resources = item.get("resources", [])
                roadmap_lines.append(f"### {area}")
                if current:
                    roadmap_lines.append(f"- 当前水平: {current}")
                if target:
                    roadmap_lines.append(f"- 目标水平: {target}")
                if timeline:
                    roadmap_lines.append(f"- 建议周期: {timeline}")
                if resources:
                    roadmap_lines.append("- 推荐资源: " + "、".join(resources))
                roadmap_lines.append("")
            console.print(Panel(
                Markdown("\n".join(roadmap_lines)),
                title="[bold green]📚 学习路线图[/bold green]",
                border_style="green",
            ))

        # 矛盾数
        n_contradictions = len(orch.state.contradictions)
        if n_contradictions:
            console.print(
                f"\n  [bold red]⚠️ 检测到 {n_contradictions} 处矛盾[/bold red]"
            )
            for c in orch.state.contradictions:
                console.print(f"    - {c.get('description', '')}")


# 注册阶段处理器
PHASE_HANDLERS = {
    Phase.INIT: handle_init,
    Phase.RESUME_INPUT: handle_resume_input,
    Phase.RESUME_MODIFY: handle_resume_modify,
    Phase.INTERVIEW_LOOP: handle_interview_loop,
    Phase.EVALUATION: handle_evaluation,
    Phase.REPORT: handle_report,
}
