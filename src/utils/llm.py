"""LLM client helpers for Mind-of-Director."""
from __future__ import annotations

import os
import base64
import mimetypes
from pathlib import Path
from typing import Any, Sequence

from src.utils.io import extract_json_payload, write_json, write_text

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from src.utils.mysecrets import API_KEY as DEFAULT_API_KEY, BASE_URL as DEFAULT_BASE_URL
except ImportError:  # pragma: no cover
    DEFAULT_API_KEY = ""
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails after all retries."""


class GPTClient:
    """Thin wrapper around an OpenAI-compatible chat completion client."""

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is required for GPT calls. Install it with `pip install openai`."
            )
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def request_json(self, prompt: str, *, debug: bool = False) -> Any:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        if debug:
            print(completion.model_dump_json())
        content = completion.choices[0].message.content or ""
        return extract_json_payload(content)

    def request_text(self, prompt: str, *, debug: bool = False) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        if debug:
            print(completion.model_dump_json())
        return completion.choices[0].message.content or ""


def gpt_text_call(
    prompt: str,
    *,
    client: GPTClient,
    output_path: str | Path | None = None,
    retries: int = 3,
    debug: bool = False,
    expect_json: bool = True,
) -> Any:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            if expect_json:
                payload = client.request_json(prompt, debug=debug)
                if output_path is not None:
                    write_json(payload, output_path)
                return payload

            text = client.request_text(prompt, debug=debug)
            if output_path is not None:
                write_text(text, output_path)
            return text
        except Exception as err:
            print(err)
            last_error = err

    raise LLMCallError("LLM call failed after retries.") from last_error


def encode_image_as_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def gpt_text_image_call(
    prompt: str,
    image_paths: Sequence[str | Path],
    *,
    model: str,
    output_path: str | Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    retries: int = 3,
    debug: bool = False,
) -> str:
    if OpenAI is None:
        raise ImportError(
            "openai package is required for image-text GPT calls. Install it with `pip install openai`."
        )

    key = resolve_api_key(api_key)
    url = resolve_base_url(base_url)
    client = OpenAI(api_key=key, base_url=url)

    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        path_text = str(image_path)
        image_url = path_text if path_text.startswith(("http://", "https://")) else encode_image_as_data_url(path_text)
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": prompt})

    last_error: Exception | None = None
    for _ in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
            if debug:
                print(completion.model_dump_json())
            text = completion.choices[0].message.content or ""
            if output_path is not None:
                write_text(text, output_path)
            return text
        except Exception as err:
            print(err)
            last_error = err

    raise LLMCallError("Image-text LLM call failed after retries.") from last_error


def resolve_api_key(explicit_key: str | None = None) -> str:
    return (
        explicit_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or DEFAULT_API_KEY
    )


def resolve_base_url(explicit_url: str | None = None) -> str:
    return (
        explicit_url
        or os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or DEFAULT_BASE_URL
    )
