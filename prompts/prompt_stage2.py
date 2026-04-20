"""Prompt builders for Stage 2: screenplay scene to Holodeck scene query."""
from __future__ import annotations

import json
from typing import Any, Mapping


def build_holodeck_query_prompt(scene: Mapping[str, Any]) -> str:
    return (
        "You convert a screenplay scene into one concise Holodeck scene-generation query.\n\n"
        "Holodeck is better at concrete spaces and visible objects than abstract plot. "
        "Write a single English sentence that names the compact scene type, main furniture, important props, and mood. "
        "Do not mention camera shots, people, character names, dialogue, walls, ceiling, background image, or Unity. "
        "Prefer real physical objects that can be found in indoor 3D asset libraries.\n\n"
        "### Scene JSON\n"
        f"{json.dumps(scene, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "query": "a compact ... with ..."\n'
        "}"
    )


def build_deterministic_holodeck_query(scene: Mapping[str, Any]) -> str:
    """Local non-LLM query builder used when --refine-query is not enabled."""
    selected_location = scene.get("selected-location")
    if not isinstance(selected_location, Mapping) or not str(selected_location.get("short_prompt", "")).strip():
        raise ValueError("Scene JSON must include selected-location.short_prompt for deterministic Stage2 query.")
    location = str(selected_location["short_prompt"]).strip()
    location = location.strip()
    lowered_location = location.lower()
    for article in ("a ", "an ", "the "):
        if lowered_location.startswith(article):
            location = location[len(article):].strip()
            break

    plot = str(scene.get("story-plot", "")).strip()
    if not plot:
        raise ValueError("Scene JSON must include story-plot for deterministic Stage2 query.")

    lowered = plot.lower()
    candidates = [
        "table",
        "chair",
        "sofa",
        "desk",
        "bookshelf",
        "lamp",
        "photo album",
        "wine glass",
        "candle",
        "plant",
        "cabinet",
        "counter",
        "stool",
        "bowl",
        "book",
    ]
    props = ", ".join(item for item in candidates if item in lowered)
    if not props:
        raise ValueError("Could not derive concrete props from story-plot. Pass --query or use --refine-query.")

    return f"a compact {location} with {props}, arranged for an intimate dramatic conversation"
