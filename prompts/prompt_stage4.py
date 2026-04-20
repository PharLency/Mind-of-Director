"""Prompt builders for Stage 4: camera segment planning."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


FLOOR_SPEC = (
    "The scene uses a five by five meter x-z floor plane. "
    "The top-left corner is (0, 0), the bottom-right corner is (5, 5). "
    "Positive x means forward in the top-down image, and positive z means right. "
    "All coordinates are in meters."
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_state_matrix(blocking: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Map clip_id -> character -> static/walking."""
    result: dict[str, dict[str, str]] = {}
    for clip in blocking.get("clips", []):
        states: dict[str, str] = {}
        for character in clip.get("characters", []):
            name = character.get("name")
            if name:
                states[str(name)] = "walking" if character.get("position_change") else "static"
        result[str(clip.get("clip_id", ""))] = states
    return result


def build_common_camera_context(
    scene_info: Mapping[str, Any],
    blocking: Mapping[str, Any],
    motion: Sequence[Mapping[str, Any]],
    clip_time: Mapping[str, Any],
    camera_library: Mapping[str, Any],
) -> str:
    state_matrix = build_state_matrix(blocking)
    return (
        "### Selected scene\n"
        f"{scene_info.get('scene_id', '')}\n\n"
        "### Script context\n"
        f"{scene_info.get('scene_outline', '')}\n\n"
        "### Location\n"
        f"{scene_info.get('location', '')}\n\n"
        "### Plot focus\n"
        f"{scene_info.get('scene_plot', '')}\n\n"
        "### Dialogue goal\n"
        f"{scene_info.get('dialogue_goal', '')}\n\n"
        "### Characters\n"
        f"{_json(scene_info.get('characters', []))}\n\n"
        "### Floor coordinate system\n"
        f"{FLOOR_SPEC}\n\n"
        "### Reference images\n"
        "- Image A: topdown_detect.png shows detected objects and furniture names.\n"
        "- Image B: topdown_annotated.png shows valid standing zones T* and seat zones S*.\n\n"
        "### Final blocking\n"
        f"{_json(blocking)}\n\n"
        "### Motion selection\n"
        f"{_json(motion)}\n\n"
        "### Clip timing\n"
        f"{_json(clip_time)}\n\n"
        "### State matrix derived from blocking\n"
        f"{_json(state_matrix)}\n\n"
        "### Structured camera library\n"
        f"{_json(camera_library)}\n\n"
        "### Hard camera rules\n"
        "1. Output exactly one camera segment per clip.\n"
        "2. Use only template types from camera_library.templates, and prefer templates whose status is active.\n"
        "3. Every shot must include all required_fields for its selected template.\n"
        "4. Respect camera_library.selection_rules using the state matrix.\n"
        "5. For three dramatically active characters, prefer ensemble_three_shot or master_wide unless a single reaction is clearly dominant and geography was already established.\n"
        "6. Prefer visual continuity: avoid gratuitous angle/size jumps between adjacent clips.\n"
        "7. Use top-down geometry to preserve screen direction and avoid impossible occlusion.\n"
        "8. start/end must match clip timing when timing is available.\n"
        "9. Return JSON only. No markdown, comments, or extra text.\n"
    )


def camera_plan_schema(scene_id: str) -> str:
    return (
        "{\n"
        f'  "scene_id": "{scene_id}",\n'
        '  "camera_segments": [\n'
        "    {\n"
        '      "clip_id": "clip_01",\n'
        '      "start": 0.0,\n'
        '      "end": 3.5,\n'
        '      "duration_seconds": 3.5,\n'
        '      "speaker": "Character Name",\n'
        '      "primary_subjects": ["Character Name"],\n'
        '      "dramatic_function": "reaction|confession|reveal|spatial_reset|tension|transition",\n'
        '      "shot": {\n'
        '        "type": "template_type_from_camera_library",\n'
        '        "...": "template-specific required fields"\n'
        "      },\n"
        '      "rationale": "Why this camera segment fits the beat, blocking, movement state, and continuity."\n'
        "    }\n"
        "  ],\n"
        '  "overall_rationale": "How the camera progression supports the scene."\n'
        "}\n"
    )


def build_cinematographer_plan_prompt(
    name: str,
    scene_info: Mapping[str, Any],
    blocking: Mapping[str, Any],
    motion: Sequence[Mapping[str, Any]],
    clip_time: Mapping[str, Any],
    camera_library: Mapping[str, Any],
) -> str:
    scene_id = str(scene_info.get("scene_id", "scene_01"))
    return (
        f"You are {name}, a senior cinematographer designing camera coverage for previsualization. "
        "Create a complete camera plan with exactly one camera segment for every clip. "
        "Your taste: clear emotional storytelling, disciplined continuity, and physically plausible top-down geometry.\n\n"
        f"{build_common_camera_context(scene_info, blocking, motion, clip_time, camera_library)}\n"
        "Your response should only contain this JSON shape:\n"
        f"{camera_plan_schema(scene_id)}"
    )


def build_cinematographer_review_prompt(
    reviewer: str,
    target_author: str,
    target_plan: Mapping[str, Any],
    scene_info: Mapping[str, Any],
    blocking: Mapping[str, Any],
    motion: Sequence[Mapping[str, Any]],
    clip_time: Mapping[str, Any],
    camera_library: Mapping[str, Any],
) -> str:
    return (
        f"You are {reviewer}. Review {target_author}'s camera plan. "
        "Check template validity, state-rule compliance, subject choice, visual continuity, and timing coverage. "
        "Give concise actionable notes only.\n\n"
        f"{build_common_camera_context(scene_info, blocking, motion, clip_time, camera_library)}\n"
        "### Target camera plan\n"
        f"{_json(target_plan)}\n\n"
        "Your response should only contain this JSON:\n"
        "{\n"
        f'  "reviewer": "{reviewer}",\n'
        f'  "target_plan_author": "{target_author}",\n'
        '  "agreements": ["clip_01: what works"],\n'
        '  "disagreements": ["clip_02: what violates rules or weakens storytelling"],\n'
        '  "suggestions": [\n'
        '    {"clip_id": "clip_02", "change": "specific replacement or field edit", "reason": "state rule, continuity, or dramatic reason"}\n'
        "  ],\n"
        '  "summary": "overall assessment"\n'
        "}\n"
    )


def build_director_synthesis_prompt(
    scene_info: Mapping[str, Any],
    blocking: Mapping[str, Any],
    motion: Sequence[Mapping[str, Any]],
    clip_time: Mapping[str, Any],
    camera_library: Mapping[str, Any],
    plan_a: Mapping[str, Any],
    plan_b: Mapping[str, Any],
    review_a_on_b: Mapping[str, Any],
    review_b_on_a: Mapping[str, Any],
) -> str:
    scene_id = str(scene_info.get("scene_id", "scene_01"))
    return (
        "You are the Director finalizing the camera plan. "
        "Merge the two cinematographer plans and reviews into one complete plan. "
        "Prioritize emotional clarity, timing correctness, state-rule compliance, and stable continuity. "
        "Return exactly one camera segment per clip.\n\n"
        f"{build_common_camera_context(scene_info, blocking, motion, clip_time, camera_library)}\n"
        "### Cinematographer A plan\n"
        f"{_json(plan_a)}\n\n"
        "### Cinematographer B plan\n"
        f"{_json(plan_b)}\n\n"
        "### A review of B\n"
        f"{_json(review_a_on_b)}\n\n"
        "### B review of A\n"
        f"{_json(review_b_on_a)}\n\n"
        "Your response should only contain this JSON shape:\n"
        f"{camera_plan_schema(scene_id)}"
    )
