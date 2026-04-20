"""Stage 3 pipeline: blocking, motion selection, and dialogue audio.

Run from Mind-of-Director:
    python -m src.pipeline.pipeline_stage_3 --scene-index 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from prompts import prompt_stage3
from src.utils.io import ensure_directory, extract_json_payload, load_json, read_text, resolve_path, write_json, write_text
from src.utils.llm import GPTClient, gpt_text_call, gpt_text_image_call, resolve_api_key, resolve_base_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Stage3Inputs:
    script_root: Path
    scene_dir: Path
    scene_index: int
    motion_library: Path
    output_dir: Path

    @property
    def scene_folder(self) -> str:
        return f"scene_{self.scene_index:02d}"

    @property
    def s1_dir(self) -> Path:
        return self.script_root / "s1"

    @property
    def outline_path(self) -> Path:
        return self.s1_dir / "outline.txt"

    @property
    def dialogues_path(self) -> Path:
        return self.s1_dir / "dialogues.json"

    @property
    def characters_path(self) -> Path:
        return self.s1_dir / "characters.json"

    @property
    def detect_image(self) -> Path:
        return self.scene_dir / "topdown_images" / "topdown_detect.png"

    @property
    def annotated_image(self) -> Path:
        return self.scene_dir / "topdown_annotated.png"

    @property
    def objects_path(self) -> Path:
        return self.scene_dir / "topdown_images" / "objects.json"

    @property
    def points_path(self) -> Path:
        return self.scene_dir / "topdown_labels.json"


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def extract_scene_block(outline_text: str, scene_index: int) -> str:
    lines = outline_text.splitlines()
    start_token = f"**Scene {scene_index}:"
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(start_token):
            start = idx
            break
    if start is None:
        raise ValueError(f"Scene {scene_index} not found in outline.")
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("**Scene ") and ":" in lines[idx]:
            end = idx
            break
    return "\n".join(lines[start:end]).strip()


def parse_outline_field(scene_block: str, key: str) -> str:
    for line in scene_block.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(f"{key.lower()}:"):
            return stripped.split(":", 1)[1].strip()
    raise ValueError(f"Scene outline is missing required field: {key}")


def collect_scene_characters(scene_block: str, scene_dialogue: Sequence[Mapping[str, Any]]) -> list[str]:
    characters: list[str] = []
    for line in scene_block.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("involved characters:"):
            for item in stripped.split(":", 1)[1].split(","):
                name = item.strip()
                if name and name not in characters:
                    characters.append(name)
            break
    for turn in scene_dialogue:
        speaker = turn.get("speaker")
        if speaker and speaker not in characters:
            characters.append(str(speaker))
    if not characters:
        raise ValueError("Could not collect any scene characters from outline/dialogue.")
    return characters


def gather_character_profiles(characters_path: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    data = load_json(characters_path)
    profiles: dict[str, dict[str, Any]] = {}
    for item in data:
        name = item.get("name")
        if name in names:
            profiles[name] = {key: value for key, value in item.items() if key != "name"}
    missing = sorted(set(names).difference(profiles))
    if missing:
        raise ValueError(f"Missing character profiles for: {missing}")
    return profiles


def load_scene_info(inputs: Stage3Inputs) -> dict[str, Any]:
    for path, label in [
        (inputs.outline_path, "outline.txt"),
        (inputs.dialogues_path, "dialogues.json"),
        (inputs.characters_path, "characters.json"),
        (inputs.detect_image, "topdown_detect.png"),
        (inputs.annotated_image, "topdown_annotated.png"),
        (inputs.objects_path, "objects.json"),
        (inputs.points_path, "topdown_labels.json"),
    ]:
        require_file(path, label)

    outline_text = read_text(inputs.outline_path)
    scene_block = extract_scene_block(outline_text, inputs.scene_index)
    dialogues = load_json(inputs.dialogues_path)
    if not (1 <= inputs.scene_index <= len(dialogues)):
        raise IndexError(f"scene_index {inputs.scene_index} out of range for {inputs.dialogues_path}")
    scene_dialogue = dialogues[inputs.scene_index - 1].get("scene-dialogue", [])
    characters = collect_scene_characters(scene_block, scene_dialogue)

    return {
        "scene_id": inputs.scene_folder,
        "scene_outline": scene_block,
        "scene_plot": parse_outline_field(scene_block, "Plot"),
        "location": parse_outline_field(scene_block, "Location"),
        "dialogue_goal": parse_outline_field(scene_block, "Dialogue goal"),
        "dialogues": scene_dialogue,
        "points_json": load_json(inputs.points_path),
        "object_list": load_json(inputs.objects_path),
        "characters": characters,
        "character_profiles": gather_character_profiles(inputs.characters_path, characters),
    }


def enforce_seat_face_to_null(blocking: dict[str, Any]) -> dict[str, Any]:
    for clip in blocking.get("clips", []):
        for character in clip.get("characters", []):
            if str(character.get("start_point", "")).upper().startswith("S"):
                character["start_face_to"] = None
            if str(character.get("end_point", "")).upper().startswith("S"):
                character["end_face_to"] = None
    return blocking


def validate_blocking(blocking: Mapping[str, Any], scene_info: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    points = scene_info.get("points_json", {})
    valid_points = {
        item.get("label")
        for group in (points.get("standing", []), points.get("seats", []))
        for item in group
    }
    valid_characters = set(scene_info.get("characters", []))

    previous_by_character: dict[str, Mapping[str, Any]] = {}
    for clip in blocking.get("clips", []):
        clip_id = clip.get("clip_id", "unknown")
        occupied: dict[str, str] = {}
        present = set()
        for character in clip.get("characters", []):
            name = character.get("name")
            present.add(name)
            for key in ("start_point", "end_point"):
                point = character.get(key)
                if point not in valid_points:
                    warnings.append(f"{clip_id}/{name}: invalid {key}={point}")
            point = character.get("start_point")
            if point in occupied and occupied[point] != name:
                warnings.append(f"{clip_id}: point {point} occupied by both {occupied[point]} and {name}")
            occupied[point] = str(name)
            for prefix in ("start", "end"):
                point_value = str(character.get(f"{prefix}_point", ""))
                face_to = character.get(f"{prefix}_face_to")
                if point_value.upper().startswith("S") and face_to is not None:
                    warnings.append(f"{clip_id}/{name}: {prefix}_face_to must be null at seat {point_value}")
            previous = previous_by_character.get(str(name))
            if previous:
                if character.get("start_point") != previous.get("end_point"):
                    warnings.append(f"{clip_id}/{name}: start_point breaks continuity")
                if character.get("start_state") != previous.get("end_state"):
                    warnings.append(f"{clip_id}/{name}: start_state breaks continuity")
                if character.get("start_face_to") != previous.get("end_face_to"):
                    warnings.append(f"{clip_id}/{name}: start_face_to breaks continuity")
            previous_by_character[str(name)] = character
        missing = valid_characters - present
        if missing:
            warnings.append(f"{clip_id}: missing characters {sorted(missing)}")
    return warnings


def ask_vlm_json(prompt: str, image_paths: Sequence[Path], output_base: Path, model: str, retries: int) -> Any:
    write_text(prompt, output_base.with_name(output_base.name + "_prompt.txt"))
    raw_path = output_base.with_name(output_base.name + "_raw.txt")
    raw = gpt_text_image_call(
        prompt,
        image_paths,
        model=model,
        output_path=raw_path,
        retries=retries,
    )
    payload = extract_json_payload(raw)
    write_json(payload, output_base.with_name(output_base.name + "_output.json"))
    return payload


def run_blocking_generation(
    scene_info: dict[str, Any],
    inputs: Stage3Inputs,
    blocking_dir: Path,
    *,
    model: str,
    review_rounds: int,
    retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_paths = [inputs.detect_image, inputs.annotated_image]
    history: list[dict[str, Any]] = []

    proposal = ask_vlm_json(
        prompt_stage3.build_director_blocking_prompt(scene_info),
        image_paths,
        blocking_dir / "round_0_director_initial",
        model,
        retries,
    )
    proposal = enforce_seat_face_to_null(proposal)
    history.append({"round": 0, "stage": "director_initial", "proposal": proposal})

    for round_number in range(1, review_rounds + 1):
        feedback: list[dict[str, Any]] = []
        cine_feedback = ask_vlm_json(
            prompt_stage3.build_cinematographer_review_prompt(proposal, scene_info, round_number),
            image_paths,
            blocking_dir / f"round_{round_number}_cinematographer",
            model,
            retries,
        )
        feedback.append(cine_feedback)

        scene_feedback = ask_vlm_json(
            prompt_stage3.build_scene_designer_review_prompt(proposal, scene_info, round_number),
            image_paths,
            blocking_dir / f"round_{round_number}_scene_designer",
            model,
            retries,
        )
        feedback.append(scene_feedback)

        for actor in scene_info["characters"]:
            actor_feedback = ask_vlm_json(
                prompt_stage3.build_actor_blocking_review_prompt(
                    actor,
                    scene_info.get("character_profiles", {}).get(actor, {}),
                    proposal,
                    scene_info,
                    round_number,
                ),
                image_paths,
                blocking_dir / f"round_{round_number}_actor_{actor}",
                model,
                retries,
            )
            feedback.append(actor_feedback)

        history.append({"round": round_number, "stage": "feedback", "feedback": feedback})
        proposal = ask_vlm_json(
            prompt_stage3.build_director_revision_prompt(scene_info, proposal, feedback, round_number),
            image_paths,
            blocking_dir / f"round_{round_number}_director_revision",
            model,
            retries,
        )
        proposal = enforce_seat_face_to_null(proposal)
        history.append({"round": round_number, "stage": "director_revision", "proposal": proposal})

    write_json(proposal, blocking_dir / "final_blocking.json")
    write_json(history, blocking_dir / "discussion_history.json")
    return proposal, history


def copy_existing_file(source: Path, destination: Path, label: str) -> Any:
    require_file(source, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return load_json(source)
    shutil.copy2(source, destination)
    return load_json(destination)


def load_motion_library(path: Path) -> dict[str, Any]:
    require_file(path, "motion library")
    return load_json(path)


def list_action_category(library: Mapping[str, Any], category: str) -> list[dict[str, Any]]:
    key = {
        "standing": "standing_actions",
        "sitting": "sitting_actions",
        "walking": "walking_actions",
        "transition": "transition_actions",
    }[category]
    return list(library.get(key, []))


def find_action(library: Mapping[str, Any], action_id: int) -> dict[str, Any]:
    for key in ("standing_actions", "sitting_actions", "walking_actions", "transition_actions"):
        for action in library.get(key, []):
            if str(action.get("id")) == str(action_id):
                return dict(action)
    raise ValueError(f"Action id {action_id} not found in motion library.")


def determine_motion_category(start_state: str, end_state: str, position_change: bool) -> str:
    if position_change:
        return "walking"
    if start_state == end_state:
        return start_state
    return "transition"


def build_motion_context(blocking: Mapping[str, Any], library: Mapping[str, Any]) -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    for clip in blocking.get("clips", []):
        clip_context = {
            "clip_id": clip.get("clip_id"),
            "speaker": clip.get("speaker"),
            "content": clip.get("content"),
            "characters": [],
        }
        for character in clip.get("characters", []):
            start_state = character.get("start_state")
            end_state = character.get("end_state")
            position_change = bool(character.get("position_change"))
            category = determine_motion_category(start_state, end_state, position_change)
            fixed_action = None
            is_fixed = False
            if position_change:
                fixed_action = find_action(library, 37)
                is_fixed = True
            elif start_state == "standing" and end_state == "sitting":
                fixed_action = find_action(library, 38)
                is_fixed = True
            elif start_state == "sitting" and end_state == "standing":
                fixed_action = find_action(library, 39)
                is_fixed = True
            clip_context["characters"].append(
                {
                    "name": character.get("name"),
                    "start_state": start_state,
                    "end_state": end_state,
                    "position_change": position_change,
                    "start_face_to": character.get("start_face_to"),
                    "end_face_to": character.get("end_face_to"),
                    "motion_tag": category,
                    "is_fixed": is_fixed,
                    "fixed_action": fixed_action,
                    "allowed_category": category,
                    "allowed_actions": [] if is_fixed else list_action_category(library, category),
                }
            )
        clips.append(clip_context)
    return {"scene_id": blocking.get("scene_id"), "clips": clips}


def strip_points_from_blocking(blocking: Mapping[str, Any]) -> dict[str, Any]:
    output = {"scene_id": blocking.get("scene_id"), "clips": []}
    for clip in blocking.get("clips", []):
        item = {
            "clip_id": clip.get("clip_id"),
            "speaker": clip.get("speaker"),
            "content": clip.get("content"),
            "characters": [],
        }
        for character in clip.get("characters", []):
            item["characters"].append(
                {
                    "name": character.get("name"),
                    "start_state": character.get("start_state"),
                    "end_state": character.get("end_state"),
                    "position_change": character.get("position_change", False),
                    "start_face_to": character.get("start_face_to"),
                    "end_face_to": character.get("end_face_to"),
                }
            )
        output["clips"].append(item)
    return output


def collect_actors_from_blocking(blocking: Mapping[str, Any]) -> list[str]:
    actors: list[str] = []
    for clip in blocking.get("clips", []):
        for character in clip.get("characters", []):
            name = character.get("name")
            if name and name not in actors:
                actors.append(str(name))
    return actors


def validate_motion_selection(motions: Sequence[Mapping[str, Any]], blocking: Mapping[str, Any], library: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    action_ids = {
        int(action["id"])
        for key in ("standing_actions", "sitting_actions", "walking_actions", "transition_actions")
        for action in library.get(key, [])
    }
    expected_by_clip = {
        clip.get("clip_id"): {character.get("name") for character in clip.get("characters", [])}
        for clip in blocking.get("clips", [])
    }
    for clip in motions:
        clip_id = clip.get("clip_id")
        actions = clip.get("actions", [])
        actual = {action.get("character") for action in actions}
        expected = expected_by_clip.get(clip_id, set())
        if actual != expected:
            warnings.append(f"{clip_id}: motion coverage mismatch expected={sorted(expected)} actual={sorted(actual)}")
        for action_record in actions:
            action = action_record.get("action", {})
            if int(action.get("id", -1)) not in action_ids:
                warnings.append(f"{clip_id}/{action_record.get('character')}: invalid action id {action.get('id')}")
    return warnings


def run_motion_generation(
    scene_outline: str,
    blocking: dict[str, Any],
    library: dict[str, Any],
    motion_dir: Path,
    *,
    model: str,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    client = GPTClient(api_key=resolve_api_key(), model=model, base_url=resolve_base_url())
    prompts_dir = ensure_directory(motion_dir / "prompts")
    versions_dir = ensure_directory(motion_dir / "motion_versions")

    blocking_summary = strip_points_from_blocking(blocking)
    motion_context = build_motion_context(blocking, library)
    history: list[dict[str, Any]] = []

    prompt = prompt_stage3.build_motion_screenwriter_prompt(scene_outline, blocking_summary, motion_context)
    initial = gpt_text_call(prompt, client=client, output_path=prompts_dir / "screenwriter_initial.json", retries=retries)
    write_json(initial, versions_dir / "motions_v1.json")
    history.append({"stage": "screenwriter_initial", "motion_data": initial})

    feedback: list[dict[str, Any]] = []
    for actor in collect_actors_from_blocking(blocking):
        actor_prompt = prompt_stage3.build_motion_actor_review_prompt(actor, initial, motion_context, blocking_summary)
        actor_feedback = gpt_text_call(
            actor_prompt,
            client=client,
            output_path=prompts_dir / f"actor_{actor}_review.json",
            retries=retries,
        )
        feedback.append(actor_feedback)
    history.append({"stage": "actor_reviews", "feedback": feedback})

    director_prompt = prompt_stage3.build_motion_director_prompt(initial, motion_context, blocking_summary)
    decision = gpt_text_call(
        director_prompt,
        client=client,
        output_path=prompts_dir / "director_final.json",
        retries=retries,
    )
    final_motions = decision.get("revised_motions") if decision.get("decision") == "REVISED" else initial
    history.append({"stage": "director_final", "decision": decision})

    write_json(final_motions, motion_dir / "motion_selection.json")
    write_json(history, motion_dir / "motion_discussion_history.json")
    return final_motions, history


def load_characters(characters_path: Path) -> dict[str, dict[str, Any]]:
    characters = {}
    for item in load_json(characters_path):
        name = item.get("name")
        if not name:
            raise ValueError(f"Character entry missing name: {item}")
        gender = item.get("gender")
        if gender not in {"male", "female"}:
            raise ValueError(f"Character {name} has invalid or missing gender: {gender}")
        characters[name] = {
            "gender": gender,
            "speaking_style": item["speaking style"],
            "personality": item["personality traits"],
        }
    return characters


def detect_language(text: str) -> str:
    return "Chinese" if any("\u4e00" <= char <= "\u9fff" for char in text) else "English"


def safe_audio_name(clip_id: str, speaker: str) -> str:
    speaker_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", speaker).strip("_")
    if not speaker_slug:
        raise ValueError(f"Invalid speaker name for audio filename: {speaker!r}")
    return f"{clip_id}_{speaker_slug}.mp3"


def run_audio_generation(
    characters_path: Path,
    blocking: Mapping[str, Any],
    audio_dir: Path,
    voices_path: Path,
    *,
    model: str,
    retries: int,
    skip_existing: bool,
) -> dict[str, Any]:
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer

    api_key = os.getenv("DASHSCOPE_API_KEY") or resolve_api_key()
    if not api_key:
        raise RuntimeError("Missing DashScope API key for audio generation.")
    dashscope.api_key = api_key

    voice_pool = {
        "male": ["longshuo_v2", "longye_v2", "longjing_v2"],
        "female": ["longwan_v2", "loongabby_v2", "longxiaochun_v2"],
    }
    characters = load_characters(characters_path)
    voice_assignments = load_json(voices_path) if voices_path.exists() else {"voices": {}, "model": model}
    if "voices" not in voice_assignments:
        raise ValueError(f"Voice assignment file is missing voices field: {voices_path}")

    def choose_voice(character: str) -> tuple[str, str]:
        if character not in characters:
            raise ValueError(f"Speaker {character} is missing from characters.json")
        gender = characters[character]["gender"]
        existing = voice_assignments["voices"].get(character, {}).get("voice")
        if existing:
            return gender, existing
        if gender not in voice_pool:
            raise ValueError(f"Unsupported gender for voice selection: {gender}")
        pool = voice_pool[gender]
        used = {item.get("voice") for item in voice_assignments["voices"].values() if item.get("gender") == gender}
        voice = next((candidate for candidate in pool if candidate not in used), pool[len(used) % len(pool)])
        voice_assignments["voices"][character] = {"voice": voice, "gender": gender}
        return gender, voice

    manifest = {
        "scene_id": blocking.get("scene_id"),
        "total_clips": len(blocking.get("clips", [])),
        "clips": [],
        "voice_assignments": voice_assignments["voices"],
        "errors": [],
    }
    audio_dir.mkdir(parents=True, exist_ok=True)

    for clip in blocking.get("clips", []):
        clip_id = clip.get("clip_id", "clip_xx")
        speaker = clip.get("speaker", "")
        text = clip.get("content", "")
        if not speaker or not text:
            manifest["errors"].append({"clip_id": clip_id, "error": "missing speaker/content"})
            continue
        gender, voice = choose_voice(speaker)
        output_name = safe_audio_name(clip_id, speaker)
        output_path = audio_dir / output_name
        status = "success"
        if skip_existing and output_path.exists():
            status = "skipped"
        else:
            ok = False
            last_error = None
            for _ in range(retries):
                try:
                    synthesizer = SpeechSynthesizer(model=model, voice=voice)
                    output_path.write_bytes(synthesizer.call(text))
                    ok = True
                    break
                except Exception as error:
                    last_error = error
                    time.sleep(2)
            if not ok:
                status = "failed"
                manifest["errors"].append({"clip_id": clip_id, "speaker": speaker, "error": str(last_error)})
        manifest["clips"].append(
            {
                "clip_id": clip_id,
                "speaker": speaker,
                "text": text,
                "gender": gender,
                "voice": voice,
                "language": detect_language(text),
                "output": output_name if status in {"success", "skipped"} else None,
                "status": status,
            }
        )

    voice_assignments["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(voice_assignments, voices_path)
    write_json(manifest, audio_dir / "audio_manifest.json")
    return manifest


def copy_example_audio(source_audio_dir: Path, destination_audio_dir: Path) -> dict[str, Any]:
    require_file(source_audio_dir / "audio_manifest.json", "audio manifest")
    if source_audio_dir.resolve() == destination_audio_dir.resolve():
        return load_json(source_audio_dir / "audio_manifest.json")
    if destination_audio_dir.exists():
        shutil.rmtree(destination_audio_dir)
    shutil.copytree(source_audio_dir, destination_audio_dir)
    return load_json(destination_audio_dir / "audio_manifest.json")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    script_root = resolve_path(PROJECT_ROOT, args.script_root)
    if script_root is None:
        raise RuntimeError("script_root resolved to None")
    scene_index = int(args.scene_index)
    default_scene_dir = script_root / "s3" / f"scene_{scene_index:02d}"
    scene_dir = resolve_path(PROJECT_ROOT, args.scene_dir, default=default_scene_dir)
    motion_library = resolve_path(PROJECT_ROOT, args.motion_library, default=PROJECT_ROOT / "examples" / "assets" / "motion_library.json")
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir, default=PROJECT_ROOT / "log" / "stage3" / f"scene_{scene_index:02d}")
    assert scene_dir is not None and motion_library is not None and output_dir is not None

    inputs = Stage3Inputs(
        script_root=script_root,
        scene_dir=scene_dir,
        scene_index=scene_index,
        motion_library=motion_library,
        output_dir=output_dir,
    )
    ensure_directory(output_dir)
    scene_info = load_scene_info(inputs)
    scene_outline = scene_info["scene_outline"]

    blocking_dir = ensure_directory(output_dir / "blocking")
    motion_dir = ensure_directory(output_dir / "motion")
    audio_dir = output_dir / "audio"

    if args.blocking_source == "generate":
        blocking, blocking_history = run_blocking_generation(
            scene_info,
            inputs,
            blocking_dir,
            model=args.vlm_model,
            review_rounds=args.rounds,
            retries=args.retries,
        )
    else:
        blocking = copy_existing_file(
            scene_dir / "blocking" / "final_blocking.json",
            blocking_dir / "final_blocking.json",
            "existing final_blocking.json",
        )
        blocking_history = []

    blocking = enforce_seat_face_to_null(blocking)
    write_json(blocking, blocking_dir / "final_blocking.json")
    blocking_warnings = validate_blocking(blocking, scene_info)
    write_json(blocking_warnings, blocking_dir / "validation_warnings.json")

    library = load_motion_library(inputs.motion_library)
    if args.motion_source == "generate":
        motions, motion_history = run_motion_generation(
            scene_outline,
            blocking,
            library,
            motion_dir,
            model=args.llm_model,
            retries=args.retries,
        )
    else:
        motions = copy_existing_file(
            scene_dir / "motion" / "motion_selection.json",
            motion_dir / "motion_selection.json",
            "existing motion_selection.json",
        )
        motion_history = []

    motion_warnings = validate_motion_selection(motions, blocking, library)
    write_json(motion_warnings, motion_dir / "validation_warnings.json")

    if args.audio_source == "generate":
        manifest = run_audio_generation(
            inputs.characters_path,
            blocking,
            audio_dir,
            output_dir / "voices" / "voice_assignments.json",
            model=args.tts_model,
            retries=args.retries,
            skip_existing=args.skip_existing_audio,
        )
    elif args.audio_source == "existing":
        manifest = copy_example_audio(scene_dir / "audio", audio_dir)
    else:
        manifest = {"status": "skipped"}

    stage3_manifest = {
        "stage": "stage3",
        "scene_id": inputs.scene_folder,
        "script_root": str(inputs.script_root.relative_to(PROJECT_ROOT) if inputs.script_root.is_relative_to(PROJECT_ROOT) else inputs.script_root),
        "scene_dir": str(inputs.scene_dir.relative_to(PROJECT_ROOT) if inputs.scene_dir.is_relative_to(PROJECT_ROOT) else inputs.scene_dir),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
        "blocking_source": args.blocking_source,
        "motion_source": args.motion_source,
        "audio_source": args.audio_source,
        "blocking_warnings": blocking_warnings,
        "motion_warnings": motion_warnings,
        "audio_errors": manifest.get("errors", []) if isinstance(manifest, dict) else [],
        "blocking_history_count": len(blocking_history),
        "motion_history_count": len(motion_history),
    }
    write_json(stage3_manifest, output_dir / "stage3_manifest.json")
    return stage3_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mind-of-Director Stage3 pipeline")
    parser.add_argument("--script-root", required=True, help="Relative to Mind-of-Director unless absolute.")
    parser.add_argument("--scene-dir", default=None, help="Scene asset dir. Default: <script-root>/s3/scene_NN.")
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--motion-library", default="examples/assets/motion_library.json")
    parser.add_argument("--output-dir", default=None, help="Default: log/stage3/scene_NN.")
    parser.add_argument("--blocking-source", choices=["existing", "generate"], default="existing")
    parser.add_argument("--motion-source", choices=["existing", "generate"], default="existing")
    parser.add_argument("--audio-source", choices=["existing", "generate", "skip"], default="existing")
    parser.add_argument("--vlm-model", default="qwen-vl-plus")
    parser.add_argument("--llm-model", default="qwen-max-latest")
    parser.add_argument("--tts-model", default="cosyvoice-v2")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-existing-audio", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_pipeline(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
