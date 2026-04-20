"""Stage 4 pipeline: assign camera segments to dialogue clips.

Run from Mind-of-Director:
    python -m src.pipeline.pipeline_stage_4 \
      --script-root examples/script_101 \
      --scene-index 1 \
      --blocking-json log/stage3/scene_01/blocking/final_blocking.json \
      --motion-json log/stage3/scene_01/motion/motion_selection.json \
      --clip-time-json examples/script_101/s3/scene_01/clip_time/clip_time.json \
      --detect-image examples/script_101/s3/scene_01/topdown_images/topdown_detect.png \
      --annotated-image examples/script_101/s3/scene_01/topdown_annotated.png
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from prompts import prompt_stage4
from src.utils.io import ensure_directory, extract_json_payload, load_json, read_text, resolve_path, write_json, write_text
from src.utils.llm import gpt_text_image_call


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def collect_characters(blocking: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for clip in blocking.get("clips", []):
        for character in clip.get("characters", []):
            name = character.get("name")
            if name and name not in names:
                names.append(str(name))
    return names


def load_scene_info(script_root: Path, scene_index: int, blocking: Mapping[str, Any]) -> dict[str, Any]:
    outline_path = script_root / "s1" / "outline.txt"
    require_file(outline_path, "outline.txt")
    scene_block = extract_scene_block(read_text(outline_path), scene_index)
    return {
        "scene_id": f"scene_{scene_index:02d}",
        "scene_outline": scene_block,
        "location": parse_outline_field(scene_block, "Location"),
        "scene_plot": parse_outline_field(scene_block, "Plot"),
        "dialogue_goal": parse_outline_field(scene_block, "Dialogue goal"),
        "characters": collect_characters(blocking),
    }


def clip_time_index(clip_time: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(clip.get("clip_id")): clip for clip in clip_time.get("clips", [])}


def motion_index(motions: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(clip.get("clip_id")): clip for clip in motions}


def active_subjects_for_clip(clip: Mapping[str, Any]) -> list[str]:
    speaker = clip.get("speaker")
    characters = [ch.get("name") for ch in clip.get("characters", []) if ch.get("name")]
    subjects = []
    if speaker:
        subjects.append(str(speaker))
    for character in characters:
        if character not in subjects:
            subjects.append(str(character))
    return subjects


def states_for_subjects(clip: Mapping[str, Any], subjects: Sequence[str]) -> dict[str, str]:
    subject_set = set(subjects)
    states = {}
    for character in clip.get("characters", []):
        name = character.get("name")
        if name in subject_set:
            states[str(name)] = "walking" if character.get("position_change") else "static"
    return states


def choose_dramatic_function(content: str) -> str:
    text = content.lower()
    if any(word in text for word in ("tell you", "something", "believe", "confess")):
        return "confession"
    if any(word in text for word in ("why", "feel", "storm", "older")):
        return "reaction"
    if any(word in text for word in ("house", "after dinner", "talk")):
        return "reveal"
    return "tension"


def deterministic_shot_for_clip(clip: Mapping[str, Any], clip_number: int, total_clips: int) -> dict[str, Any]:
    speaker = str(clip.get("speaker") or "")
    content = str(clip.get("content") or "")
    subjects = active_subjects_for_clip(clip)
    states = states_for_subjects(clip, subjects)
    all_static = all(state == "static" for state in states.values())
    group_active = len(subjects) >= 3 and clip_number in {1, total_clips}

    if group_active:
        return {
            "type": "ensemble_three_shot",
            "coverage_role": "ensemble",
            "subjects": subjects,
            "shot_size": "WS",
            "framing": "triangular",
            "angle": "Eye",
            "lens_band": "normal_35_50",
            "camera_height": "standing_eye",
            "movement": "locked_off",
            "focus": "deep_focus",
            "screen_direction": "preserve_axis",
            "rationale": "Re-establishes ensemble geography and relational tension at a key scene boundary.",
        }

    if speaker:
        if all_static and any(word in content.lower() for word in ("something", "tell you", "storm", "older")):
            return {
                "type": "reaction_single" if "storm" in content.lower() or "older" in content.lower() else "clean_single",
                "coverage_role": "reaction" if "storm" in content.lower() or "older" in content.lower() else "single",
                "subject": speaker,
                "position": "Head",
                "direction": "front",
                "shot_size": "CU",
                "angle": "Eye",
                "lens_band": "portrait_65_85",
                "camera_height": "standing_eye",
                "movement": "locked_off",
                "focus": "single_subject",
                "screen_direction": "preserve_axis",
                "rationale": "Static close framing isolates the speaker's emotional subtext without adding unmotivated camera motion.",
            }
        return {
            "type": "clean_single",
            "coverage_role": "single",
            "subject": speaker,
            "position": "Spine",
            "direction": "front-left",
            "shot_size": "MCU",
            "angle": "Eye",
            "lens_band": "portrait_65_85",
            "camera_height": "standing_eye",
            "movement": "locked_off",
            "focus": "single_subject",
            "screen_direction": "preserve_axis",
            "rationale": "A controlled single keeps the dialogue readable while preserving lookroom toward the listener.",
        }

    pair = subjects[:2]
    return {
        "type": "two_shot",
        "coverage_role": "relationship",
        "subjects": pair,
        "relation": "distant",
        "framing": "two_shot",
        "shot_size": "MS",
        "angle": "Eye",
        "lens_band": "normal_35_50",
        "camera_height": "standing_eye",
        "movement": "locked_off",
        "focus": "shared_subjects",
        "screen_direction": "preserve_axis",
        "rationale": "Two-shot preserves relationship geometry for a static dialogue beat.",
    }


def repair_shot_fields(shot: Mapping[str, Any], clip: Mapping[str, Any]) -> dict[str, Any]:
    repaired = dict(shot)
    shot_type = repaired.get("type")
    subjects = active_subjects_for_clip(clip)

    if shot_type == "full_shot":
        repaired["type"] = "ensemble_three_shot"
        repaired.setdefault("coverage_role", "ensemble")
        repaired.setdefault("subjects", subjects)
        repaired.setdefault("shot_size", "WS")
        repaired.setdefault("framing", "triangular")
        repaired.setdefault("angle", "Eye")
        repaired.setdefault("lens_band", "normal_35_50")
        repaired.setdefault("camera_height", "standing_eye")
        repaired.setdefault("movement", "locked_off")
        repaired.setdefault("focus", "deep_focus")
        repaired.setdefault("screen_direction", "preserve_axis")
    elif shot_type == "singlestatic":
        repaired["type"] = "clean_single"
        repaired.setdefault("coverage_role", "single")
        repaired.setdefault("subject", clip.get("speaker") or (subjects[0] if subjects else "Unknown"))
        repaired.setdefault("position", "Head")
        repaired.setdefault("direction", "front")
        repaired.setdefault("shot_size", "MCU")
        repaired.setdefault("angle", "Eye")
        repaired.setdefault("lens_band", "portrait_65_85")
        repaired.setdefault("camera_height", "standing_eye")
        repaired.setdefault("movement", "locked_off")
        repaired.setdefault("focus", "single_subject")
        repaired.setdefault("screen_direction", "preserve_axis")
    elif shot_type == "two_static":
        legacy_subjects = list(repaired.get("subjects") or subjects[:2])
        if repaired.get("framing") == "OTS_pair" and len(legacy_subjects) >= 2:
            repaired["type"] = "over_the_shoulder"
            repaired.setdefault("coverage_role", "relationship")
            repaired.setdefault("subject", legacy_subjects[0])
            repaired.setdefault("foreground_subject", legacy_subjects[1])
            repaired.pop("subjects", None)
            repaired["framing"] = "OTS"
            repaired.setdefault("lens_band", "portrait_65_85")
            repaired.setdefault("focus", "single_subject")
        else:
            repaired["type"] = "two_shot"
            repaired.setdefault("coverage_role", "relationship")
            repaired.setdefault("subjects", legacy_subjects)
            repaired.setdefault("framing", "two_shot")
            repaired.setdefault("lens_band", "normal_35_50")
            repaired.setdefault("focus", "shared_subjects")
        repaired.setdefault("relation", "distant")
        repaired.setdefault("shot_size", "MS")
        repaired.setdefault("angle", "Eye")
        repaired.setdefault("camera_height", "standing_eye")
        repaired.setdefault("movement", "locked_off")
        repaired.setdefault("screen_direction", "preserve_axis")
    return repaired


def normalize_camera_plan(
    camera_plan: Mapping[str, Any],
    blocking: Mapping[str, Any],
    motion: Sequence[Mapping[str, Any]],
    clip_time: Mapping[str, Any],
) -> dict[str, Any]:
    time_by_clip = clip_time_index(clip_time)
    motion_by_clip = motion_index(motion)
    plan_by_clip = {str(clip.get("clip_id")): clip for clip in camera_plan.get("clips", camera_plan.get("camera_segments", []))}
    segments = []

    for idx, clip in enumerate(blocking.get("clips", []), start=1):
        clip_id = str(clip.get("clip_id"))
        timing = time_by_clip.get(clip_id, {})
        planned = plan_by_clip.get(clip_id, {})
        shots = planned.get("shots")
        if shots:
            shot = shots[0]
        else:
            shot = planned.get("shot") or deterministic_shot_for_clip(clip, idx, len(blocking.get("clips", [])))
        shot = repair_shot_fields(shot, clip)
        subjects = shot.get("subjects") or [shot.get("subject")] if isinstance(shot, dict) else []
        subjects = [subject for subject in subjects if subject]
        if not subjects:
            subjects = active_subjects_for_clip(clip)[:1]
        start = float(timing.get("start", 0.0))
        end = float(timing.get("end", start + timing.get("duration_seconds", 4.0)))
        segments.append(
            {
                "clip_id": clip_id,
                "start": start,
                "end": end,
                "duration_seconds": float(timing.get("duration_seconds", end - start)),
                "speaker": clip.get("speaker"),
                "primary_subjects": subjects,
                "dramatic_function": choose_dramatic_function(str(clip.get("content", ""))),
                "shot": shot,
                "rationale": planned.get("rationale") or shot.get("rationale", ""),
                "motion_actions": motion_by_clip.get(clip_id, {}).get("actions", []),
            }
        )

    return {
        "scene_id": blocking.get("scene_id", camera_plan.get("scene_id")),
        "camera_segments": segments,
        "overall_rationale": camera_plan.get("overall_rationale") or "One camera segment is assigned per dialogue clip, with timing aligned to clip_time.json.",
    }


def validate_camera_plan(plan: Mapping[str, Any], blocking: Mapping[str, Any], camera_library: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    expected_clip_ids = [clip.get("clip_id") for clip in blocking.get("clips", [])]
    segments = plan.get("camera_segments", [])
    actual_clip_ids = [segment.get("clip_id") for segment in segments]
    if actual_clip_ids != expected_clip_ids:
        warnings.append(f"clip order mismatch expected={expected_clip_ids} actual={actual_clip_ids}")

    templates = {template["type"]: template for template in camera_library.get("templates", [])}
    for segment in segments:
        clip_id = segment.get("clip_id")
        shot = segment.get("shot", {})
        shot_type = shot.get("type")
        template = templates.get(shot_type)
        if template is None:
            warnings.append(f"{clip_id}: unknown shot type {shot_type}")
            continue
        for field in template.get("required_fields", []):
            if field not in shot:
                warnings.append(f"{clip_id}: shot type {shot_type} missing required field {field}")
        if segment.get("end", 0) <= segment.get("start", 0):
            warnings.append(f"{clip_id}: non-positive segment duration")
    return warnings


def ask_vlm_json(prompt: str, image_paths: Sequence[Path], output_base: Path, model: str, retries: int) -> Any:
    write_text(prompt, output_base.with_name(output_base.name + "_prompt.txt"))
    raw_path = output_base.with_name(output_base.name + "_raw.txt")
    raw = gpt_text_image_call(prompt, image_paths, model=model, output_path=raw_path, retries=retries)
    payload = extract_json_payload(raw)
    write_json(payload, output_base.with_name(output_base.name + "_output.json"))
    return payload


def run_camera_generation(
    scene_info: Mapping[str, Any],
    blocking: Mapping[str, Any],
    motion: Sequence[Mapping[str, Any]],
    clip_time: Mapping[str, Any],
    camera_library: Mapping[str, Any],
    detect_image: Path,
    annotated_image: Path,
    camera_dir: Path,
    *,
    model: str,
    retries: int,
) -> dict[str, Any]:
    images = [detect_image, annotated_image]
    plan_a = ask_vlm_json(
        prompt_stage4.build_cinematographer_plan_prompt("Cinematographer_A", scene_info, blocking, motion, clip_time, camera_library),
        images,
        camera_dir / "cinematographer_A_plan",
        model,
        retries,
    )
    plan_b = ask_vlm_json(
        prompt_stage4.build_cinematographer_plan_prompt("Cinematographer_B", scene_info, blocking, motion, clip_time, camera_library),
        images,
        camera_dir / "cinematographer_B_plan",
        model,
        retries,
    )
    review_a_on_b = ask_vlm_json(
        prompt_stage4.build_cinematographer_review_prompt("Cinematographer_A", "Cinematographer_B", plan_b, scene_info, blocking, motion, clip_time, camera_library),
        images,
        camera_dir / "cinematographer_A_review_of_B",
        model,
        retries,
    )
    review_b_on_a = ask_vlm_json(
        prompt_stage4.build_cinematographer_review_prompt("Cinematographer_B", "Cinematographer_A", plan_a, scene_info, blocking, motion, clip_time, camera_library),
        images,
        camera_dir / "cinematographer_B_review_of_A",
        model,
        retries,
    )
    final_plan = ask_vlm_json(
        prompt_stage4.build_director_synthesis_prompt(scene_info, blocking, motion, clip_time, camera_library, plan_a, plan_b, review_a_on_b, review_b_on_a),
        images,
        camera_dir / "director_final_plan",
        model,
        retries,
    )
    return final_plan


def copy_existing_camera(source_dir: Path, destination_dir: Path) -> Mapping[str, Any]:
    require_file(source_dir / "camera_final_plan.json", "existing camera_final_plan.json")
    if source_dir.resolve() != destination_dir.resolve():
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        shutil.copytree(source_dir, destination_dir)
    return load_json(destination_dir / "camera_final_plan.json")


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    scene_folder = f"scene_{args.scene_index:02d}"
    script_root = resolve_path(PROJECT_ROOT, args.script_root)
    assert script_root is not None

    paths = {
        "script_root": script_root,
        "blocking": resolve_path(PROJECT_ROOT, args.blocking_json),
        "motion": resolve_path(PROJECT_ROOT, args.motion_json),
        "clip_time": resolve_path(PROJECT_ROOT, args.clip_time_json),
        "detect_image": resolve_path(PROJECT_ROOT, args.detect_image),
        "annotated_image": resolve_path(PROJECT_ROOT, args.annotated_image),
        "camera_library": resolve_path(PROJECT_ROOT, args.camera_library, default=PROJECT_ROOT / "examples" / "assets" / "camera_library.json"),
        "existing_camera_dir": resolve_path(PROJECT_ROOT, args.existing_camera_dir),
        "output_dir": resolve_path(PROJECT_ROOT, args.output_dir, default=PROJECT_ROOT / "log" / "stage4" / scene_folder),
    }
    required_explicit = [key for key in ("blocking", "motion", "clip_time", "detect_image", "annotated_image") if paths[key] is None]
    if required_explicit:
        joined = ", ".join(f"--{key.replace('_', '-')}" for key in required_explicit)
        raise RuntimeError(f"Stage4 requires explicit input paths: {joined}")
    if args.camera_source == "existing" and paths["existing_camera_dir"] is None:
        raise RuntimeError("Stage4 camera-source existing requires --existing-camera-dir.")
    return {key: value for key, value in paths.items() if value is not None}


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_inputs(args)
    output_dir = paths["output_dir"]
    camera_dir = ensure_directory(output_dir / "camera")

    for key, label in [
        ("blocking", "final_blocking.json"),
        ("motion", "motion_selection.json"),
        ("clip_time", "clip_time.json"),
        ("camera_library", "camera_library.json"),
        ("detect_image", "topdown_detect.png"),
        ("annotated_image", "topdown_annotated.png"),
    ]:
        require_file(paths[key], label)

    blocking = load_json(paths["blocking"])
    motion = load_json(paths["motion"])
    clip_time = load_json(paths["clip_time"])
    camera_library = load_json(paths["camera_library"])
    scene_info = load_scene_info(paths["script_root"], args.scene_index, blocking)

    if args.camera_source == "existing":
        raw_plan = copy_existing_camera(paths["existing_camera_dir"], camera_dir)
    elif args.camera_source == "deterministic":
        raw_plan = {
            "scene_id": blocking.get("scene_id"),
            "clips": [
                {
                    "clip_id": clip.get("clip_id"),
                    "shots": [deterministic_shot_for_clip(clip, idx, len(blocking.get("clips", [])))],
                    "rationale": "Deterministic local camera choice based on speaker, dialogue beat, and group reset rules.",
                }
                for idx, clip in enumerate(blocking.get("clips", []), start=1)
            ],
        }
        write_json(raw_plan, camera_dir / "camera_raw_plan.json")
    else:
        raw_plan = run_camera_generation(
            scene_info,
            blocking,
            motion,
            clip_time,
            camera_library,
            paths["detect_image"],
            paths["annotated_image"],
            camera_dir,
            model=args.vlm_model,
            retries=args.retries,
        )

    normalized_plan = normalize_camera_plan(raw_plan, blocking, motion, clip_time)
    warnings = validate_camera_plan(normalized_plan, blocking, camera_library)
    write_json(normalized_plan, camera_dir / "camera_segments.json")
    write_json(warnings, camera_dir / "validation_warnings.json")

    manifest = {
        "stage": "stage4",
        "scene_id": f"scene_{args.scene_index:02d}",
        "camera_source": args.camera_source,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
        "camera_segments": str((camera_dir / "camera_segments.json").relative_to(PROJECT_ROOT)),
        "validation_warnings": warnings,
        "inputs": {
            "blocking": str(paths["blocking"].relative_to(PROJECT_ROOT) if paths["blocking"].is_relative_to(PROJECT_ROOT) else paths["blocking"]),
            "motion": str(paths["motion"].relative_to(PROJECT_ROOT) if paths["motion"].is_relative_to(PROJECT_ROOT) else paths["motion"]),
            "clip_time": str(paths["clip_time"].relative_to(PROJECT_ROOT) if paths["clip_time"].is_relative_to(PROJECT_ROOT) else paths["clip_time"]),
            "camera_library": str(paths["camera_library"].relative_to(PROJECT_ROOT)),
        },
    }
    write_json(manifest, output_dir / "stage4_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mind-of-Director Stage4 camera planning pipeline")
    parser.add_argument("--script-root", required=True)
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--blocking-json", required=True)
    parser.add_argument("--motion-json", required=True)
    parser.add_argument("--clip-time-json", required=True)
    parser.add_argument("--detect-image", required=True)
    parser.add_argument("--annotated-image", required=True)
    parser.add_argument("--camera-library", default="examples/assets/camera_library.json")
    parser.add_argument("--existing-camera-dir", default=None)
    parser.add_argument("--camera-source", choices=["existing", "deterministic", "generate"], default="deterministic")
    parser.add_argument("--vlm-model", default="qwen-vl-plus")
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
