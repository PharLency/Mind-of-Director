"""Prompt builders for Stage 3: blocking, motion, and performance audio."""
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


def collect_valid_face_targets(characters: Sequence[str], objects: Sequence[Mapping[str, Any]]) -> list[str]:
    valid: list[str] = []
    seen: set[str] = set()
    for name in list(characters) + [str(obj.get("name", "")) for obj in objects]:
        if name and name not in seen:
            seen.add(name)
            valid.append(name)
    return valid


def build_blocking_context(scene_info: Mapping[str, Any]) -> str:
    """Shared context used by director, actors, cinematographer, and scene designer."""
    valid_face_to = collect_valid_face_targets(
        scene_info.get("characters", []),
        scene_info.get("object_list", []),
    )
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
        "### Dialogue turns, used as clips in this exact order\n"
        f"{_json(scene_info.get('dialogues', []))}\n\n"
        "### Reference images\n"
        "- Image A: topdown_detect.png shows detected objects and names.\n"
        "- Image B: topdown_annotated.png shows valid standing zones T* and seat zones S*.\n\n"
        "### Object list\n"
        "Use object names exactly when a character faces an object.\n"
        f"{_json(scene_info.get('object_list', []))}\n\n"
        "### Available points\n"
        "- Standing points T*: valid state is standing.\n"
        "- Seat points S*: valid states are standing or sitting.\n"
        f"{_json(scene_info.get('points_json', {}))}\n\n"
        "### Hard constraints\n"
        "1. Use only provided T*/S* labels; never invent points.\n"
        "2. Each point is exclusive in each clip; two characters cannot occupy the same label.\n"
        "3. Seat points are exclusive whether the character is standing at the seat or sitting.\n"
        "4. Every scene character must appear in every clip, even when silent.\n"
        "5. Preserve continuity between clips: a character's next start point/state/face_to must match the previous end point/state/face_to.\n"
        "6. T* face_to must be an exact character name or exact object name. S* face_to must be null.\n"
        "7. Avoid visible collision with object boxes and avoid placing characters at corners unless strongly motivated.\n"
        "8. Do not add unnecessary movement. Every point/state/facing change must be motivated by dialogue, emotion, or composition.\n\n"
        "### Valid face_to targets for T* points\n"
        f"{_json(valid_face_to)}\n\n"
        "### face_to rules\n"
        "- Never use T*/S* labels as face_to.\n"
        "- To face another person, write the character name.\n"
        "- To face furniture/prop, write the exact object name.\n"
        "- At S* points, face_to must be null regardless of sitting or standing.\n\n"
        "Return JSON only. No markdown, comments, or extra prose.\n"
    )


def blocking_json_template(scene_id: str) -> str:
    return (
        "{\n"
        f'  "scene_id": "{scene_id}",\n'
        '  "clips": [\n'
        '    {\n'
        '      "clip_id": "clip_01",\n'
        '      "speaker": "Character Name",\n'
        '      "content": "dialogue text",\n'
        '      "characters": [\n'
        '        {\n'
        '          "name": "Character Name",\n'
        '          "start_point": "T1 or S1",\n'
        '          "start_state": "standing or sitting",\n'
        '          "start_face_to": "Exact Object Name or Exact Character Name, or null when point is S*",\n'
        '          "end_point": "T1 or S1",\n'
        '          "end_state": "standing or sitting",\n'
        '          "end_face_to": "Exact Object Name or Exact Character Name, or null when point is S*",\n'
        '          "position_change": false\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "design_rationale": "Explain continuity, composition, motivated movement, seat usage, and collision awareness."\n'
        "}\n"
    )


def build_director_blocking_prompt(scene_info: Mapping[str, Any]) -> str:
    scene_id = str(scene_info.get("scene_id", "scene_01"))
    return (
        "You are the Director creating the initial blocking plan for a previsualization scene. "
        "Create one clip per dialogue turn, keep dialogue order, and assign every character's start/end point, state, and facing. "
        "Prefer simple, motivated blocking over busy movement.\n\n"
        f"{build_blocking_context(scene_info)}\n"
        "Your response should only contain the following JSON content:\n"
        f"{blocking_json_template(scene_id)}"
    )


def build_actor_blocking_review_prompt(
    actor_name: str,
    actor_profile: Mapping[str, Any],
    proposal: Mapping[str, Any],
    scene_info: Mapping[str, Any],
    round_number: int,
) -> str:
    return (
        f"You are the actor playing {actor_name}. This is review round {round_number}. "
        "Review only your own point/state/facing choices for motivation, continuity, and physical feasibility. "
        "If you disagree, propose minimal exact T*/S* changes and valid face_to values.\n\n"
        f"{build_blocking_context(scene_info)}\n"
        "### Character profile\n"
        f"{_json(actor_profile)}\n\n"
        "### Current blocking proposal\n"
        f"{_json(proposal)}\n\n"
        "Your response should only contain the following JSON content:\n"
        "{\n"
        '  "role": "actor",\n'
        f'  "name": "{actor_name}",\n'
        '  "approval_level": 4,\n'
        '  "agreements": ["clip_01: what works"],\n'
        '  "disagreements": ["clip_02: what does not work"],\n'
        '  "suggestions": ["clip_02: concrete minimal fix with exact labels"],\n'
        '  "summary": "short assessment"\n'
        "}\n"
    )


def build_cinematographer_review_prompt(
    proposal: Mapping[str, Any],
    scene_info: Mapping[str, Any],
    round_number: int,
) -> str:
    return (
        f"You are the Cinematographer. This is review round {round_number}. "
        "Evaluate sightlines, occlusion risk, screen-direction continuity, and coverage practicality. "
        "Recommend only minimal changes that improve camera coverage.\n\n"
        f"{build_blocking_context(scene_info)}\n"
        "### Current blocking proposal\n"
        f"{_json(proposal)}\n\n"
        "Your response should only contain the following JSON content:\n"
        "{\n"
        '  "role": "cinematographer",\n'
        '  "name": "Cinematographer",\n'
        '  "approval_level": 4,\n'
        '  "agreements": ["clip_01: what works for coverage"],\n'
        '  "disagreements": ["clip_02: coverage problem"],\n'
        '  "suggestions": ["clip_02: minimal fix"],\n'
        '  "summary": "short assessment"\n'
        "}\n"
    )


def build_scene_designer_review_prompt(
    proposal: Mapping[str, Any],
    scene_info: Mapping[str, Any],
    round_number: int,
) -> str:
    return (
        f"You are the Scene Designer. This is review round {round_number}. "
        "Validate furniture collision, reachable seats, plausible standing zones, and use of the top-down layout. "
        "Recommend only minimal fixes.\n\n"
        f"{build_blocking_context(scene_info)}\n"
        "### Current blocking proposal\n"
        f"{_json(proposal)}\n\n"
        "Your response should only contain the following JSON content:\n"
        "{\n"
        '  "role": "scene_designer",\n'
        '  "name": "Scene Designer",\n'
        '  "approval_level": 4,\n'
        '  "agreements": ["clip_01: spatial logic that works"],\n'
        '  "disagreements": ["clip_02: spatial problem"],\n'
        '  "suggestions": ["clip_02: minimal fix"],\n'
        '  "summary": "short assessment"\n'
        "}\n"
    )


def build_director_revision_prompt(
    scene_info: Mapping[str, Any],
    current_proposal: Mapping[str, Any],
    feedback: Sequence[Mapping[str, Any]],
    round_number: int,
) -> str:
    scene_id = str(scene_info.get("scene_id", "scene_01"))
    return (
        f"You are the Director revising the blocking after feedback round {round_number}. "
        "Synthesize feedback, fix violations, and preserve strong choices. "
        "Make the smallest useful change set and avoid extra movement.\n\n"
        f"{build_blocking_context(scene_info)}\n"
        "### Current proposal\n"
        f"{_json(current_proposal)}\n\n"
        "### Feedback\n"
        f"{_json(feedback)}\n\n"
        "Your response should only contain the following JSON content:\n"
        f"{blocking_json_template(scene_id)}"
    )


GLOBAL_MOTION_GOAL = (
    "GLOBAL GOAL:\n"
    "- Assign exactly one action to every character in every clip.\n"
    "- Fixed actions must not change.\n"
    "- Non-fixed actions must be selected from the provided allowed_actions list.\n"
    "- Keep action choices emotionally aligned, simple, and not over-busy.\n"
)


MOTION_RULES = (
    "HARD RULES:\n"
    "1. position_change == true means action id must be 37, Walking Middle Speed.\n"
    "2. standing -> sitting means action id must be 38, Stand To Sit.\n"
    "3. sitting -> standing means action id must be 39, Stand Up.\n"
    "4. standing -> standing must choose from standing_actions only.\n"
    "5. sitting -> sitting must choose from sitting_actions only.\n"
    "6. Include both id and name for every action.\n"
    "7. Every clip must cover all characters exactly once.\n"
)


def build_motion_screenwriter_prompt(
    scene_outline: str,
    blocking_summary: Mapping[str, Any],
    motion_context: Mapping[str, Any],
) -> str:
    return (
        "You are the Screenwriter designing initial per-clip body actions.\n\n"
        "### Scene outline\n"
        f"{scene_outline}\n\n"
        "### Blocking summary\n"
        f"{_json(blocking_summary)}\n\n"
        "### Motion context\n"
        f"{_json(motion_context)}\n\n"
        f"{GLOBAL_MOTION_GOAL}\n{MOTION_RULES}\n"
        "Return JSON only:\n"
        "[\n"
        "  {\n"
        '    "clip_id": "clip_01",\n'
        '    "actions": [\n'
        '      {"character": "Name", "state": "standing|sitting|transition|walking", "action": {"id": 15, "name": "Standing Idle"}, "reasoning": "short reason"}\n'
        "    ]\n"
        "  }\n"
        "]"
    )


def build_motion_actor_review_prompt(
    actor_name: str,
    initial_plan: Sequence[Mapping[str, Any]],
    motion_context: Mapping[str, Any],
    blocking_summary: Mapping[str, Any],
) -> str:
    return (
        f"You are actor {actor_name}. Review only your own action choices. "
        "Do not change fixed actions. If a non-fixed action feels wrong, suggest one valid replacement from allowed_actions.\n\n"
        "### Blocking summary\n"
        f"{_json(blocking_summary)}\n\n"
        "### Initial plan\n"
        f"{_json(initial_plan)}\n\n"
        "### Motion context\n"
        f"{_json(motion_context)}\n\n"
        f"{GLOBAL_MOTION_GOAL}\n{MOTION_RULES}\n"
        "Return JSON only:\n"
        "{\n"
        f'  "actor": "{actor_name}",\n'
        '  "approval_level": 4,\n'
        '  "agreements": ["clip_01 ok"],\n'
        '  "disagreements": [],\n'
        '  "suggestions": [\n'
        f'    {{"clip_id": "clip_01", "character": "{actor_name}", "replace_with": {{"id": 15, "name": "Standing Idle"}}, "reason": "short reason"}}\n'
        "  ]\n"
        "}"
    )


def build_motion_director_prompt(
    plan_after_actors: Sequence[Mapping[str, Any]],
    motion_context: Mapping[str, Any],
    blocking_summary: Mapping[str, Any],
) -> str:
    return (
        "You are the Director giving final approval for motion selection. "
        "Check coverage, fixed action correctness, action validity, and emotional clarity.\n\n"
        "### Blocking summary\n"
        f"{_json(blocking_summary)}\n\n"
        "### Plan after actor reviews\n"
        f"{_json(plan_after_actors)}\n\n"
        "### Motion context\n"
        f"{_json(motion_context)}\n\n"
        f"{GLOBAL_MOTION_GOAL}\n{MOTION_RULES}\n"
        "Return JSON only:\n"
        "{\n"
        '  "decision": "APPROVED",\n'
        '  "reasoning": "short reason",\n'
        '  "revised_motions": null\n'
        "}\n"
    )
