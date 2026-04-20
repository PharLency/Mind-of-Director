"""Local secret hooks.

Do not hard-code credentials in this file. Set environment variables instead:

    export DASHSCOPE_API_KEY="..."
    export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
"""
from __future__ import annotations

import os


API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
BASE_URL = (
    os.getenv("DASHSCOPE_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
