"""
CareerCoach Agent — 职业发展教练（原 ResumeModifier）。
三种风格改写简历：
- rigorous (专业打磨): 零虚构，优化表达 + 修改说明
- embellished (技术深化): 技术深度扩展 + 学习建议
- wild (成长路线): 前沿技术扩展 + 完整学习路线图
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from mosaic.core.agent import CAREER_COACH_POLICY, BaseAgent
from mosaic.core.events import Event, EventType
from mosaic.llm.client import LLMClient
from mosaic.resume.schema import ResumeData

logger = logging.getLogger(__name__)

# 模板目录
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class CareerCoachAgent(BaseAgent):
    """
    职业发展教练 Agent。

    根据 JD 和原始简历，生成三种风格的优化版本：
    - rigorous: 温度 0.3，专业打磨版（零虚构）
    - embellished: 温度 0.5，技术深化版（技术深度扩展）
    - wild: 温度 0.7，成长路线版（前沿技术 + 学习路线图）

    输出格式：单个 Markdown 字符串，`---` 分隔简历和教练笔记。

    事件发布：RESUME_MODIFIED
    """

    STYLE_CONFIG = {
        "rigorous": {
            "temperature": 0.3,
            "template": "resume_rigorous.j2",
            "label": "专业打磨版",
        },
        "embellished": {
            "temperature": 0.5,
            "template": "resume_embellished.j2",
            "label": "技术深化版",
        },
        "wild": {
            "temperature": 0.7,
            "template": "resume_wild.j2",
            "label": "成长路线版",
        },
    }

    def __init__(self, llm: LLMClient) -> None:
        super().__init__(name="career_coach", memory_policy=CAREER_COACH_POLICY)
        self.llm = llm
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def modify(
        self,
        resume: ResumeData,
        job_description: str,
        styles: list[str] | None = None,
    ) -> dict[str, str]:
        """
        生成多种风格的简历优化版本。

        Returns:
            {"rigorous": "...", "embellished": "...", "wild": "..."}
            每个值为 Markdown 字符串，包含简历 + `---` + 教练笔记。
        """
        styles = styles or list(self.STYLE_CONFIG.keys())
        results: dict[str, str] = {}
        resume_text = resume.to_text()

        for style in styles:
            if style not in self.STYLE_CONFIG:
                logger.warning(f"Unknown style: {style}, skipping")
                continue

            config = self.STYLE_CONFIG[style]
            try:
                template = self._jinja_env.get_template(config["template"])
                prompt = template.render(
                    resume=resume_text,
                    job_description=job_description,
                )
            except Exception:
                # fallback: 直接用文本提示
                prompt = self._fallback_prompt(resume_text, job_description, style)

            logger.info(f"Generating {style} resume (temp={config['temperature']})")

            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=config["temperature"],
                max_tokens=3000,
            )
            results[style] = response

        await self._emit(Event(
            type=EventType.RESUME_MODIFIED,
            source=self.name,
            data={"styles": list(results.keys())},
        ))

        return results

    def _fallback_prompt(
        self, resume_text: str, jd: str, style: str
    ) -> str:
        """模板加载失败时的后备提示词"""
        style_instructions = {
            "rigorous": (
                "你是一位职业发展教练，请对简历进行专业打磨。\n"
                "- **零虚构**：绝不添加任何简历中没有的内容\n"
                "- 优化表达方式，使之更专业、更有说服力\n"
                "- 根据 JD 重新组织重点，突出匹配的经验\n"
                "- 使用 STAR 格式重构项目描述\n"
                "- 输出两部分（用 `---` 分隔）：修改后简历 + 修改说明（每处修改的理由和学习建议）"
            ),
            "embellished": (
                "你是一位职业发展教练，请对简历进行技术深化。\n"
                "- 基于已有技术方向扩展到进阶表达\n"
                "- 不改基本事实（教育、公司、头衔、时间、数字）\n"
                "- 根据 JD 在相关技术领域做更深入的表达\n"
                "- 输出两部分（用 `---` 分隔）：修改后简历 + 教练笔记（每处扩展的学习建议和面试准备要点）"
            ),
            "wild": (
                "你是一位职业发展教练，请规划成长路线。\n"
                "- 大胆扩展到行业前沿技术\n"
                "- 不改基本事实（教育、公司、头衔、时间、数字）\n"
                "- 输出两部分（用 `---` 分隔）：修改后简历 + 完整学习路线图（按周规划 + 推荐资源 + 面试必备知识点）"
            ),
        }

        return (
            f"你是一位职业发展教练。请根据以下JD优化简历。\n\n"
            f"## 优化风格要求\n{style_instructions.get(style, '')}\n\n"
            f"## 不可变字段\n"
            f"教育背景、公司名、头衔、任职时间、个人信息绝不可修改。\n\n"
            f"## 目标职位 JD\n{jd}\n\n"
            f"## 原始简历\n{resume_text}\n\n"
            f"请输出优化后的完整简历 + 教练笔记（用 `---` 分隔）："
        )

    async def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Agent 接口"""
        resume = context.get("resume")
        jd = context.get("job_description", "")
        styles = context.get("styles")

        if not resume:
            return {"error": "No resume provided"}

        if isinstance(resume, dict):
            resume = ResumeData(**resume)

        results = await self.modify(resume, jd, styles)
        return {"modified_resumes": results}
