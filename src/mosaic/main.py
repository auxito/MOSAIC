"""
MOSAIC — Multi-agent Orchestrated Simulation for Adaptive Interview Coaching
CLI 入口。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from mosaic.config import Config
from mosaic.core.orchestrator import Orchestrator
from mosaic.core.workflow import WORKFLOWS, WorkflowConfig
from mosaic.llm.client import LLMClient

# Agent imports
from mosaic.agents.evaluator import EvaluatorAgent
from mosaic.agents.interviewee import IntervieweeAgent
from mosaic.agents.interviewer import InterviewerAgent
from mosaic.agents.memory_manager import MemoryManagerAgent
from mosaic.agents.career_coach import CareerCoachAgent
from mosaic.agents.resume_parser import ResumeParserAgent

# Participant imports
from mosaic.participants.ai_participant import AIParticipant
from mosaic.participants.human_participant import HumanParticipant

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(
            console=console,
            show_time=True,
            show_path=verbose,
            rich_tracebacks=True,
        )],
    )


def build_orchestrator(
    workflow: WorkflowConfig,
    config: Config,
    human_mode: bool = False,
) -> Orchestrator:
    """构建编排器 + 注册所有 Agent"""

    orch = Orchestrator(workflow)

    # LLM 客户端
    llm = LLMClient(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        model=config.openai_model,
    )

    # 记忆管理器
    memory_mgr = MemoryManagerAgent(llm=llm, working_memory=orch.working_memory)
    orch.register_agent(memory_mgr)

    # 简历解析器
    parser = ResumeParserAgent()
    orch.register_agent(parser)

    # 职业发展教练（原简历修改器）
    coach = CareerCoachAgent(llm=llm)
    orch.register_agent(coach)

    # 面试官
    interviewer = InterviewerAgent(llm=llm, working_memory=orch.working_memory)
    orch.register_agent(interviewer)

    # 面试者
    if human_mode:
        participant = HumanParticipant(show_hints=True)
    else:
        participant = AIParticipant(llm=llm, working_memory=orch.working_memory)

    interviewee = IntervieweeAgent(
        llm=llm,
        working_memory=orch.working_memory,
        participant=participant,
    )
    orch.register_agent(interviewee)

    # 评估官
    evaluator = EvaluatorAgent(llm=llm, working_memory=orch.working_memory)
    orch.register_agent(evaluator)

    # 注册阶段处理器
    workflow_module = _load_workflow_handlers(workflow.name)
    for phase, handler in workflow_module.items():
        orch.register_phase_handler(phase, handler)

    # 设置报告目录
    orch.state.metadata["report_dir"] = config.report_dir

    return orch


def _load_workflow_handlers(workflow_name: str) -> dict:
    """加载工作流的阶段处理器"""
    if workflow_name == "full_interview":
        from mosaic.workflows.full_interview import PHASE_HANDLERS
    elif workflow_name == "resume_only":
        from mosaic.workflows.resume_only import PHASE_HANDLERS
    elif workflow_name == "human_practice":
        from mosaic.workflows.human_practice import PHASE_HANDLERS
    else:
        # 默认使用完整面试
        from mosaic.workflows.full_interview import PHASE_HANDLERS

    return PHASE_HANDLERS


async def main_async(args: argparse.Namespace) -> None:
    """异步主函数"""
    config = Config.from_env()

    # 选择工作流
    workflow_name = args.workflow
    if workflow_name not in WORKFLOWS:
        console.print(f"[red]未知工作流: {workflow_name}[/red]")
        console.print(f"可用: {', '.join(WORKFLOWS.keys())}")
        sys.exit(1)

    workflow = WORKFLOWS[workflow_name]

    # 覆盖轮次
    if args.rounds:
        workflow.interview_rounds = args.rounds

    # 构建编排器
    human_mode = args.human or workflow_name == "human_practice"
    orch = build_orchestrator(workflow, config, human_mode=human_mode)

    # 运行
    try:
        state = await orch.run()
        console.print("\n[bold green]✅ 流程完成！[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]错误: {e}[/red]")
        if args.verbose:
            console.print_exception()


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="MOSAIC — Multi-agent Orchestrated Simulation for Adaptive Interview Coaching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
工作流模式:
  full_interview   完整流程（简历修改→面试→评估）
  resume_only      仅简历修改
  human_practice   真人练习模式
  eval_only        仅面试评估

示例:
  python -m mosaic.main --workflow full_interview
  python -m mosaic.main --workflow human_practice --rounds 5
  python -m mosaic.main --human --rounds 3
        """,
    )

    parser.add_argument(
        "--workflow", "-w",
        default="full_interview",
        choices=list(WORKFLOWS.keys()),
        help="工作流模式 (default: full_interview)",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=None,
        help="面试轮次 (default: 10)",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="真人模式（你来回答面试问题）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="启动 Gradio Web UI (http://localhost:7860)",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.web:
        from mosaic.web_app import create_app
        app = create_app()
        app.queue().launch(server_name="0.0.0.0", server_port=7860)
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
