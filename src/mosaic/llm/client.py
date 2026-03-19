"""
LLM 客户端 — OpenAI SDK 封装。
支持 OpenAI 兼容的任何 API（OpenRouter、DeepSeek、本地部署等）。

chat_json() 增强：
- 自动剥离 markdown 代码围栏（```json ... ```）
- 最多重试 2 次（第一次失败后追加纠错提示再试）
- JSON 解析失败时返回结构化错误对象而非抛异常
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# 用于剥离 LLM 常见的 markdown 围栏
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL
)


class LLMClient:
    """
    异步 LLM 客户端。

    通过环境变量配置：
    - OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    或通过构造参数覆盖。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> str:
        """
        发送聊天请求，返回文本响应。
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if response_format:
            params["response_format"] = response_format

        logger.debug(
            f"LLM call: model={self.model}, msgs={len(messages)}, temp={temperature}"
        )

        response = await self._client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""

        logger.debug(f"LLM response: {len(content)} chars")
        return content

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ):
        """
        流式聊天 — 异步生成器，逐 token yield。

        用于 Web UI 场景下的实时输出。
        """
        logger.debug(
            f"LLM stream call: model={self.model}, msgs={len(messages)}, temp={temperature}"
        )

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """清理 LLM 返回的 JSON 文本 — 剥离围栏、修复常见问题。"""
        text = text.strip()

        # 1. 剥离 markdown 代码围栏
        match = _CODE_FENCE_RE.search(text)
        if match:
            text = match.group(1).strip()

        # 2. 如果以 ``` 开头但没闭合，直接去掉第一行
        if text.startswith("```"):
            lines = text.split("\n", 1)
            text = lines[1].strip() if len(lines) > 1 else text

        # 3. 尾部多余的 ``` 清理
        if text.endswith("```"):
            text = text[:-3].strip()

        return text

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_retries: int = 1,
        **kwargs: Any,
    ) -> str:
        """
        请求 JSON 格式响应，带自动清理和重试。

        流程：
        1. 尝试用 response_format=json_object 请求
        2. 对返回内容剥离 markdown 围栏等杂质
        3. 尝试 json.loads 验证
        4. 失败则追加纠错消息重试（最多 max_retries 次）
        5. 仍然失败则返回 {"error": "..."}
        """
        attempt = 0
        last_error = ""
        current_messages = list(messages)

        while attempt <= max_retries:
            try:
                raw = await self.chat(
                    messages=current_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    **kwargs,
                )
                cleaned = self._clean_json_response(raw)
                # 验证是否为合法 JSON
                json.loads(cleaned)
                return cleaned

            except json.JSONDecodeError as e:
                attempt += 1
                last_error = str(e)
                logger.warning(
                    f"JSON parse failed (attempt {attempt}/{max_retries + 1}): {e}"
                )
                if attempt <= max_retries:
                    # 追加纠错提示重试
                    current_messages = list(messages) + [
                        {
                            "role": "assistant",
                            "content": raw if 'raw' in dir() else "",
                        },
                        {
                            "role": "user",
                            "content": (
                                "你的上一次回复不是合法的 JSON 格式。"
                                "请严格按要求只输出 JSON，不要包含任何其他文字、"
                                "markdown 围栏或注释。"
                            ),
                        },
                    ]
            except Exception as e:
                # 网络错误等直接返回错误
                logger.error(f"chat_json failed: {e}")
                return json.dumps({"error": str(e)}, ensure_ascii=False)

        logger.error(f"chat_json exhausted retries, last error: {last_error}")
        return json.dumps(
            {"error": f"JSON parse failed after retries: {last_error}"},
            ensure_ascii=False,
        )
