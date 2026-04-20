"""File and JSON helpers for Mind-of-Director."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(project_root: Path, value: str | None, *, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(text: str, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, separators=(",", ": "))


def write_json(payload: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_json_text(payload), encoding="utf-8")
    return output_path


def extract_json_payload(raw_text: str) -> Any:
    cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("LLM response is empty.")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```json\s*(.*?)\s*```", raw_text, flags=re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    array_match = re.search(r"(\[[\s\S]*\])", raw_text)
    if array_match:
        return json.loads(array_match.group(1))

    object_match = re.search(r"(\{[\s\S]*\})", raw_text)
    if object_match:
        return json.loads(object_match.group(1))

    raise ValueError("Could not extract JSON payload from model response.")


def slugify(value: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return (slug or "stage1-run")[:max_length].rstrip("-") or "stage1-run"
