import os
import time
from typing import Optional

from openai import OpenAI


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeLLM:
    """Callable LLM wrapper compatible with Holodeck's existing self.llm(prompt) calls."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen-max-latest",
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        retries: int = 3,
        retry_delay: float = 2.0,
    ):
        if not api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY or pass --openai_api_key."
            )
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url or DEFAULT_DASHSCOPE_BASE_URL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.retry_delay = retry_delay
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @classmethod
    def from_env(
        cls,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
    ):
        return cls(
            api_key=api_key
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            model_name=model_name or os.environ.get("HOLODECK_LLM_MODEL") or "qwen-max-latest",
            base_url=base_url
            or os.environ.get("DASHSCOPE_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_DASHSCOPE_BASE_URL,
            max_tokens=max_tokens,
        )

    def __call__(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as error:
                last_error = error
                if attempt < self.retries - 1:
                    time.sleep(self.retry_delay)
        raise RuntimeError(f"DashScope LLM call failed: {last_error}") from last_error
