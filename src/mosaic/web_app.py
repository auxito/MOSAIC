"""
MOSAIC Web UI — Gradio 应用。
提供简历优化、模拟面试、评估报告的 Web 交互界面。

独立入口，不影响 CLI 流程。直接调用 agents 方法编排流程。
"""

from __future__ import annotations

import asyncio
import logging
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
    """格式化单轮评估为 Markdown"""
    if "error" in evaluation:
        return f"⚠️ 评估出错: {evaluation['error']}"

    scores = evaluation.get("scores", {})
    rn = evaluation.get("round_number", "?")

    lines = [
        f"### 第 {rn} 轮评分",
        f"| 维度 | 分数 |",
        f"|------|------|",
        f"| 回答质量 | {scores.get('answer_quality', '?')}/5 |",
        f"| 技术准确 | {scores.get('technical_accuracy', '?')}/5 |",
        f"| 沟通表达 | {scores.get('communication', '?')}/5 |",
        f"| 真实可信 | {scores.get('credibility', '?')}/5 |",
        "",
        f"**评语**: {evaluation.get('comment', '')}",
    ]

    highlights = evaluation.get("highlights", "")
    if highlights:
        lines.append(f"\n✨ **亮点**: {highlights}")

    suggestions = evaluation.get("improvement_suggestions", "")
    if suggestions:
        lines.append(f"\n💡 **改进建议**: {suggestions}")

    ref_points = evaluation.get("reference_points", [])
    if ref_points:
        lines.append(f"\n📌 **参考要点**: {'；'.join(ref_points)}")

    gaps = evaluation.get("knowledge_gaps", "")
    if gaps:
        lines.append(f"\n📚 **知识盲区**: {gaps}")

    return "\n".join(lines)


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

    # 维度评分
    dims = report.get("dimension_scores", {})
    if dims:
        lines.append("### 五维评分")
        lines.append("| 维度 | 评分 |")
        lines.append("|------|------|")
        for dim, score in dims.items():
            bar = "🟢" * score + "⚪" * (5 - score) if isinstance(score, int) else ""
            lines.append(f"| {dim} | {score}/5 {bar} |")
        lines.append("")

    # 优势
    strengths = report.get("strengths", [])
    if strengths:
        lines.append("### ✅ 优势")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    # 改进
    improvements = report.get("improvements", [])
    if improvements:
        lines.append("### 📈 改进建议")
        for i in improvements:
            lines.append(f"- {i}")
        lines.append("")

    # 建议
    rec = report.get("recommendation", "")
    if rec:
        lines.append(f"### 🏷️ 录用建议: **{rec}**")
        lines.append("")

    # 详细评语
    comments = report.get("detailed_comments", "")
    if comments:
        lines.append(f"### 详细评语\n{comments}")
        lines.append("")

    # 教练建议
    coaching = report.get("coaching_notes", "")
    if coaching:
        lines.append(f"### 🎯 教练建议\n{coaching}")
        lines.append("")

    # 学习路线图
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


# ============================================================
# 核心交互函数
# ============================================================

async def _handle_upload(file_obj) -> tuple[str, ResumeData | None]:
    """处理文件上传，返回 (预览文本, ResumeData)"""
    if file_obj is None:
        return "请上传简历文件", None

    try:
        # Gradio File 组件给出文件路径
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        parser = FileParser()
        resume_data = parser.parse(file_path)
        preview = resume_data.to_text() if resume_data.summary != resume_data.to_text().strip() else resume_data.summary
        return f"✅ 简历解析成功（{len(resume_data.summary)} 字符）\n\n---\n\n{preview}", resume_data
    except Exception as e:
        return f"❌ 解析失败: {e}", None


async def _handle_resume_modify(
    session: dict,
    file_obj,
    jd: str,
    style: str,
) -> tuple[dict, str, str, str]:
    """简历优化流程，返回 (session, 简历预览, 优化结果, 教练笔记)"""
    # 解析简历
    if file_obj is None:
        return session, "", "❌ 请先上传简历文件", ""

    preview_text, resume_data = await _handle_upload(file_obj)
    if resume_data is None:
        return session, preview_text, "❌ 简历解析失败", ""

    if not jd.strip():
        return session, preview_text, "❌ 请输入目标职位 JD", ""

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
        results = await coach.modify(resume_data, jd, [style_key])
        content = results.get(style_key, "")

        # 分离简历和教练笔记
        if "---" in content:
            parts = content.split("---", 1)
            resume_part = parts[0].strip()
            notes_part = parts[1].strip()
        else:
            resume_part = content
            notes_part = ""

        # 更新 session
        session["resume_data"] = resume_data.model_dump()
        session["resume_text"] = resume_data.summary
        session["jd"] = jd
        session["modified_resumes"] = {style_key: content}
        session["selected_style"] = style_key

        return session, preview_text, resume_part, notes_part

    except Exception as e:
        logger.error(f"Resume modification failed: {e}\n{traceback.format_exc()}")
        return session, preview_text, f"❌ 优化失败: {e}", ""


async def _run_interview_ai(
    session: dict,
    total_rounds: int,
    chatbot: list,
    eval_md: str,
):
    """AI 模式面试 — generator，逐步 yield 更新"""
    config = Config.from_env()
    components = _build_session_components(config)

    interviewer = components["interviewer"]
    interviewee = components["interviewee"]
    evaluator = components["evaluator"]
    privileged = components["privileged"]
    episodic = components["episodic"]
    state = components["state"]

    # 设置特权记忆
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

    all_evals: list[str] = []
    last_answer: str | None = None

    for round_num in range(1, int(total_rounds) + 1):
        state.current_round = round_num

        # 面试官提问
        question = await interviewer.ask(round_num, int(total_rounds), last_answer)
        chatbot.append({"role": "assistant", "content": ""})
        for char in question:
            chatbot[-1]["content"] += char
            yield chatbot, eval_md
            await asyncio.sleep(0.01)

        # 记录到 episodic
        episodic.append(DialogueTurn(
            round_number=round_num, role="interviewer", content=question,
        ))

        # AI 回答
        answer = await interviewee.answer(question, round_num)
        last_answer = answer
        chatbot.append({"role": "user", "content": ""})
        for char in answer:
            chatbot[-1]["content"] += char
            yield chatbot, eval_md
            await asyncio.sleep(0.01)

        # 记录到 episodic
        episodic.append(DialogueTurn(
            round_number=round_num, role="interviewee", content=answer,
        ))

        # 对话存入 state
        state.conversation.append({"role": "interviewer", "content": question})
        state.conversation.append({"role": "interviewee", "content": answer})

        # 评估
        turn_eval = await evaluator.evaluate_turn(
            question=question,
            answer=answer,
            round_number=round_num,
            total_rounds=int(total_rounds),
        )
        state.turn_evaluations.append(turn_eval)

        all_evals.append(_format_turn_eval(turn_eval))
        eval_md = "\n\n---\n\n".join(all_evals)
        yield chatbot, eval_md

    # 面试结束，生成最终报告
    final_report = await evaluator.generate_final_report()
    state.final_evaluation = final_report

    # 保存到 session（用于 Tab 3）
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

    # 最后一次 yield 带上完成提示
    all_evals.append("\n\n---\n\n## ✅ 面试结束！请切换到「评估报告」Tab 查看完整报告。")
    eval_md = "\n\n---\n\n".join(all_evals)
    yield chatbot, eval_md


async def _run_interview_human_step(
    session: dict,
    user_message: str,
    chatbot: list,
    eval_md: str,
    total_rounds: int,
):
    """人类模式 — 处理用户发送的一条消息"""
    # 从 session 中获取或初始化组件
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

    components = session["_comp"]
    interviewer = components["interviewer"]
    evaluator = components["evaluator"]
    episodic = components["episodic"]
    state = components["state"]

    round_num = session["human_round"]

    if session.get("needs_question", True):
        # 需要先出题
        round_num += 1
        session["human_round"] = round_num
        state.current_round = round_num

        last_answer = user_message if round_num > 1 else None

        # 如果有上一轮的回答，先处理评估
        if round_num > 1 and user_message.strip():
            # 把上一轮答案记录
            chatbot.append({"role": "user", "content": user_message})

            episodic.append(DialogueTurn(
                round_number=round_num - 1, role="interviewee", content=user_message,
            ))
            state.conversation.append({"role": "interviewee", "content": user_message})

            # 评估上一轮
            prev_q = session.get("last_question", "")
            if prev_q:
                turn_eval = await evaluator.evaluate_turn(
                    question=prev_q,
                    answer=user_message,
                    round_number=round_num - 1,
                    total_rounds=int(total_rounds),
                )
                state.turn_evaluations.append(turn_eval)
                session["human_evals"].append(_format_turn_eval(turn_eval))
                eval_md = "\n\n---\n\n".join(session["human_evals"])

        # 检查是否已结束
        if round_num > int(total_rounds):
            final_report = await evaluator.generate_final_report()
            session["final_report"] = final_report
            session["human_evals"].append(
                "\n\n## ✅ 面试结束！请切换到「评估报告」Tab 查看完整报告。"
            )
            eval_md = "\n\n---\n\n".join(session["human_evals"])
            return session, chatbot, eval_md

        # 面试官提问
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
        theme=gr.themes.Soft(),
        css="""
        .chatbot-container { min-height: 500px; }
        .header { text-align: center; margin-bottom: 1em; }
        """,
    ) as app:
        # 全局 session state
        session = gr.State({
            "resume_data": None,
            "resume_text": "",
            "jd": "",
            "modified_resumes": {},
            "selected_style": "",
            "final_report": None,
            "state": None,
        })

        gr.Markdown(
            "# 🎯 MOSAIC — 职业发展教练\n"
            "**M**ulti-agent **O**rchestrated **S**imulation for **A**daptive **I**nterview **C**oaching",
            elem_classes=["header"],
        )

        # ============ Tab 1: 简历优化 ============
        with gr.Tab("📝 简历优化"):
            with gr.Row():
                with gr.Column(scale=1):
                    resume_file = gr.File(
                        label="📎 上传简历（PDF / DOCX / TXT）",
                        file_types=[".pdf", ".docx", ".txt", ".md"],
                        type="filepath",
                    )
                    jd_input = gr.Textbox(
                        label="📋 目标职位 JD",
                        placeholder="粘贴岗位描述...",
                        lines=6,
                    )
                    style_radio = gr.Radio(
                        label="🎨 优化风格",
                        choices=[
                            "🎯 专业打磨 (零虚构)",
                            "🚀 技术深化 (进阶表达)",
                            "🌟 成长路线 (前沿扩展)",
                        ],
                        value="🎯 专业打磨 (零虚构)",
                    )
                    modify_btn = gr.Button("🚀 开始优化", variant="primary", size="lg")

                with gr.Column(scale=1):
                    resume_preview = gr.Markdown(
                        label="📄 简历预览",
                        value="*上传简历后显示解析结果*",
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    modified_resume = gr.Markdown(
                        label="📄 优化后简历",
                        value="*点击「开始优化」后显示*",
                    )
                with gr.Column(scale=1):
                    coach_notes = gr.Markdown(
                        label="📝 教练笔记",
                        value="*教练笔记将在此显示*",
                    )

            # 上传后预览
            async def on_upload(file_obj):
                preview, _ = await _handle_upload(file_obj)
                return preview

            resume_file.change(
                fn=on_upload,
                inputs=[resume_file],
                outputs=[resume_preview],
            )

            # 开始优化
            modify_btn.click(
                fn=_handle_resume_modify,
                inputs=[session, resume_file, jd_input, style_radio],
                outputs=[session, resume_preview, modified_resume, coach_notes],
            )

        # ============ Tab 2: 模拟面试 ============
        with gr.Tab("🎤 模拟面试"):
            with gr.Row():
                with gr.Column(scale=1):
                    rounds_slider = gr.Slider(
                        label="面试轮次",
                        minimum=1,
                        maximum=15,
                        value=5,
                        step=1,
                    )
                    mode_radio = gr.Radio(
                        label="面试模式",
                        choices=["🤖 AI 自动模式", "👤 人类回答模式"],
                        value="🤖 AI 自动模式",
                    )
                    start_btn = gr.Button("▶️ 开始面试", variant="primary", size="lg")

                    gr.Markdown("---")
                    gr.Markdown("**👤 人类模式输入区**")
                    human_input = gr.Textbox(
                        label="你的回答",
                        placeholder="输入你的回答...",
                        lines=4,
                        interactive=True,
                    )
                    send_btn = gr.Button("📤 发送回答", size="sm")

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="💬 面试对话",
                        height=500,
                        type="messages",
                        elem_classes=["chatbot-container"],
                    )
                    eval_display = gr.Markdown(
                        label="📊 逐轮评分",
                        value="*面试开始后显示评分*",
                    )

            # AI 面试
            async def start_ai_interview(sess, rounds, mode, chat):
                if not sess.get("resume_data"):
                    chat = [{"role": "assistant", "content": "⚠️ 请先在「简历优化」Tab 完成简历优化！"}]
                    yield chat, "*请先完成简历优化*"
                    return

                chat = []
                chat.append({"role": "assistant", "content": f"🎤 面试开始！共 {int(rounds)} 轮，让我们开始吧。\n\n---"})
                eval_text = ""

                if "AI" in mode:
                    async for updated_chat, updated_eval in _run_interview_ai(
                        sess, int(rounds), chat, eval_text,
                    ):
                        yield updated_chat, updated_eval
                else:
                    # 人类模式：先出第一个问题
                    sess["needs_question"] = True
                    sess, chat, eval_text = await _run_interview_human_step(
                        sess, "", chat, eval_text, int(rounds),
                    )
                    yield chat, eval_text

            start_btn.click(
                fn=start_ai_interview,
                inputs=[session, rounds_slider, mode_radio, chatbot],
                outputs=[chatbot, eval_display],
            )

            # 人类模式发送
            async def send_human_answer(sess, msg, chat, eval_text, rounds):
                if not msg.strip():
                    return sess, chat, eval_text, ""
                sess["needs_question"] = True
                sess, chat, eval_text = await _run_interview_human_step(
                    sess, msg, chat, eval_text, int(rounds),
                )
                return sess, chat, eval_text, ""

            send_btn.click(
                fn=send_human_answer,
                inputs=[session, human_input, chatbot, eval_display, rounds_slider],
                outputs=[session, chatbot, eval_display, human_input],
            )

        # ============ Tab 3: 评估报告 ============
        with gr.Tab("📊 评估报告"):
            refresh_btn = gr.Button("🔄 加载报告", variant="primary")
            report_display = gr.Markdown(
                value="*面试结束后点击「加载报告」查看完整评估*",
            )
            report_file = gr.File(label="📥 下载报告", visible=False)

            async def load_report(sess):
                final_report = sess.get("final_report")
                if not final_report:
                    return "⚠️ 尚未完成面试，无法生成报告。请先在「模拟面试」Tab 完成面试。", gr.update(visible=False)

                report_md = _format_final_report(final_report)

                # 尝试保存报告文件
                try:
                    state_data = sess.get("state", {})
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

                    # 设置 modified resumes info
                    modified = sess.get("modified_resumes", {})
                    selected_style = sess.get("selected_style", "")
                    if selected_style and selected_style in modified:
                        state.selected_resume = {"content": modified[selected_style]}

                    filepath = generate_report(state, final_report, "reports")
                    return report_md, gr.update(value=filepath, visible=True)
                except Exception as e:
                    logger.error(f"Report save failed: {e}")
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
    app.queue().launch(server_name="0.0.0.0", server_port=7860)
