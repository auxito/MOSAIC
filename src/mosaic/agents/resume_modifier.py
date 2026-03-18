"""
ResumeModifier Agent — 三种风格改写简历。
- rigorous (严谨): 低温度，忠实原文
- embellished (修饰): 中温度，合理美化
- wild (夸张): 高温度，大幅夸大
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from mosaic.core.agent import RESUME_MODIFIER_POLICY, BaseAgent
from mosaic.core.events import Event, EventType
from mosaic.llm.client import LLMClient
from mosaic.resume.schema import ResumeData

logger = logging.getLogger(__name__)

# 模板目录
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class ResumeModifierAgent(BaseAgent):
    """
    简历改写 Agent。

    根据 JD 和原始简历，生成三种风格的修改版本：
    - rigorous: 温度 0.3，严谨风格
    - embellished: 温度 0.6，适度修饰
    - wild: 温度 0.9，大幅夸张

    事件发布：RESUME_MODIFIED
    """

    STYLE_CONFIG = {
        "rigorous": {
            "temperature": 0.3,
            "template": "resume_rigorous.j2",
            "label": "严谨版",
        },
        "embellished": {
            "temperature": 0.6,
            "template": "resume_embellished.j2",
            "label": "修饰版",
        },
        "wild": {
            "temperature": 0.9,
            "template": "resume_wild.j2",
            "label": "夸张版",
        },
    }

    def __init__(self, llm: LLMClient) -> None:
        super().__init__(name="resume_modifier", memory_policy=RESUME_MODIFIER_POLICY)
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
        生成多种风格的简历修改版本。

        Returns:
            {"rigorous": "...", "embellished": "...", "wild": "..."}
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
                "请严格基于原始简历内容进行修改。\n"
                "- 不添加任何简历中没有的经验或技能\n"
                "- 优化表达方式，使之更专业、更有说服力\n"
                "- 根据 JD 重新组织重点，突出匹配的经验\n"
                "- 量化成就时，只使用简历中已有的数据"
            ),
            "embellished": (
                "在原始简历基础上进行合理修饰和美化。\n"
                "- 可以适度提升措辞的影响力\n"
                "- 可以合理推断并补充一些细节\n"
                "- 将普通描述改为成就导向的STAR格式\n"
                "- 根据 JD 强调最相关的经验"
            ),
            "wild": (
                "大幅夸大简历内容，制造一份'注水'简历。\n"
                "- 夸大项目规模和影响力（如团队5人→20人）\n"
                "- 虚构额外的技能和经验\n"
                "- 将'参与'改为'主导'，将'协助'改为'独立完成'\n"
                "- 添加不存在的量化成果\n"
                "- 注意：这是用于面试模拟训练，帮助面试官识别夸大"
            ),
        }

        return (
            f"你是一位简历优化专家。请根据以下JD修改简历。\n\n"
            f"## 修改风格要求\n{style_instructions.get(style, '')}\n\n"
            f"## 目标职位 JD\n{jd}\n\n"
            f"## 原始简历\n{resume_text}\n\n"
            f"请输出修改后的完整简历（Markdown 格式）："
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
