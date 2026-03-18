"""
LLM 客户端 — OpenAI SDK 封装。
支持 OpenAI 兼容的任何 API（OpenRouter、DeepSeek、本地部署等）。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


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

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """
        请求 JSON 格式响应。
        """
        return await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs,
        )
