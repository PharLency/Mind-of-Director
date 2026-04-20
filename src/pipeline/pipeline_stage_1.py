"""Stage 1 pipeline: from idea to structured screenplay."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts.prompt_stage1 import (  # noqa: E402
    build_actor_feedback_prompt,
    build_actor_revision_prompt,
    build_character_prompt,
    build_dialogue_prompt,
    build_director_review_prompt,
    build_director_revision_prompt,
    build_long_prompt_detailer,
    build_scene_prompt,
)
from src.utils.io import (  # noqa: E402
    dump_json_text,
    ensure_directory,
    load_json,
    read_text,
    resolve_path,
    slugify,
    write_json,
    write_text,
)
from src.utils.llm import (  # noqa: E402
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    GPTClient,
    gpt_text_call,
    resolve_api_key,
    resolve_base_url,
)


@dataclass
class CharacterProfile:
    name: str
    age: str
    gender: str
    occupation: str
    personality_traits: str
    speaking_style: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CharacterProfile":
        required_keys = {
            "name",
            "age",
            "gender",
            "occupation",
            "personality traits",
            "speaking style",
        }
        missing = required_keys.difference(payload)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"Character payload is missing fields: {joined}")
        return cls(
            name=str(payload["name"]),
            age=str(payload["age"]),
            gender=str(payload["gender"]),
            occupation=str(payload["occupation"]),
            personality_traits=str(payload["personality traits"]),
            speaking_style=str(payload["speaking style"]),
        )

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "occupation": self.occupation,
            "personality traits": self.personality_traits,
            "speaking style": self.speaking_style,
        }


def parse_character_payload(raw: Sequence[Mapping[str, Any]] | Any) -> list[CharacterProfile]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("Character payload must decode to a list.")
    return [CharacterProfile.from_dict(item) for item in raw]


def scenes_to_outline(scenes_payload: Sequence[Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for index, scene in enumerate(scenes_payload, start=1):
        required_keys = {"sub-topic", "selected-characters", "selected-location", "story-plot", "dialogue-goal"}
        missing = required_keys.difference(scene)
        if missing:
            raise ValueError(f"Scene {index} payload is missing fields: {sorted(missing)}")
        selected_location = scene["selected-location"]
        if not isinstance(selected_location, Mapping) or "short_prompt" not in selected_location:
            raise ValueError(f"Scene {index} selected-location must include short_prompt.")
        section = (
            f"**Scene {index}: {scene['sub-topic']}**\n"
            f"Involved characters: {', '.join(scene['selected-characters'])}\n"
            f"Location: {selected_location['short_prompt']}\n"
            f"Plot: {scene['story-plot']}\n"
            f"Dialogue goal: {scene['dialogue-goal']}\n"
        )
        sections.append(section)
    return "\n".join(sections)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 pipeline: idea -> characters -> scenes -> dialogue -> feedback -> long prompts"
    )
    parser.add_argument("topic", nargs="?", help="Film topic or high-level idea.")
    parser.add_argument("--topic-file", help="Relative or absolute path to a text file containing the idea.")
    parser.add_argument("--run-name", help="Folder name under log/ for this run.")
    parser.add_argument("--out-dir", help="Optional custom output directory. Relative paths are resolved from Mind-of-Director.")
    parser.add_argument("--characters-json", help="Optional existing characters JSON to reuse.")
    parser.add_argument("--scenes-json", help="Optional existing scenes JSON to reuse.")
    parser.add_argument("--dialogues-json", help="Optional existing dialogues JSON to reuse.")
    parser.add_argument("--script-outline", help="Optional existing outline file path or literal outline text.")
    parser.add_argument("--max-characters", type=int, default=6)
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--enable-feedback", action="store_true")
    parser.add_argument("--feedback-rounds", type=int, default=1)
    parser.add_argument("--skip-long-prompts", action="store_true")
    parser.add_argument("--api-key", help="OpenAI-compatible API key.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default="qwen3-max", help="Chat model name.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--debug-llm", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_topic(args: argparse.Namespace) -> str:
    if args.topic_file:
        topic_path = resolve_path(PROJECT_ROOT, args.topic_file)
        if topic_path is None or not topic_path.exists():
            raise FileNotFoundError(f"Topic file not found: {args.topic_file}")
        return read_text(topic_path).strip()
    if args.topic:
        return args.topic.strip()
    raise ValueError("Please provide a topic or --topic-file.")


def resolve_output_dir(args: argparse.Namespace, topic: str) -> Path:
    default_run_name = args.run_name or f"stage1_{slugify(topic)}"
    default_dir = PROJECT_ROOT / "log" / default_run_name
    return resolve_path(PROJECT_ROOT, args.out_dir, default=default_dir) or default_dir


def build_llm_client(args: argparse.Namespace) -> GPTClient:
    api_key = resolve_api_key(args.api_key)
    base_url = resolve_base_url(args.base_url)
    if not api_key:
        raise RuntimeError(
            "No API key available. Set OPENAI_API_KEY / DASHSCOPE_API_KEY or update src/utils/mysecrets.py."
        )
    return GPTClient(api_key=api_key, model=args.model, base_url=base_url)


def maybe_load_outline(raw_outline: str) -> str:
    outline_path = resolve_path(PROJECT_ROOT, raw_outline)
    if outline_path is not None and outline_path.exists():
        return read_text(outline_path).strip()
    return raw_outline.strip()


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    topic = resolve_topic(args)
    out_dir = ensure_directory(resolve_output_dir(args, topic))

    write_text(topic + "\n", out_dir / "topic.txt")

    needs_llm = (
        not args.characters_json
        or not args.scenes_json
        or not args.dialogues_json
        or args.enable_feedback
        or not args.skip_long_prompts
    )
    llm_client = build_llm_client(args) if needs_llm else None

    characters_path = out_dir / "characters.json"
    scenes_path = out_dir / "scenes.json"
    outline_path = out_dir / "outline.txt"
    dialogue_path = out_dir / "dialogues.json"
    final_dialogue_path = out_dir / "dialogues_final.json"
    long_prompts_path = out_dir / "long_prompts.json"

    if args.characters_json:
        source_path = resolve_path(PROJECT_ROOT, args.characters_json)
        if source_path is None or not source_path.exists():
            raise FileNotFoundError(f"Characters JSON not found: {args.characters_json}")
        characters_payload = load_json(source_path)
        write_json(characters_payload, characters_path)
    else:
        character_prompt = build_character_prompt(topic, max_characters=args.max_characters)
        write_text(character_prompt, out_dir / "characters_prompt.txt")
        characters_payload = gpt_text_call(
            character_prompt,
            client=llm_client,
            output_path=characters_path,
            retries=args.retries,
            debug=args.debug_llm,
        )

    characters = parse_character_payload(characters_payload)
    character_dicts = [character.to_prompt_dict() for character in characters]

    if args.scenes_json:
        source_path = resolve_path(PROJECT_ROOT, args.scenes_json)
        if source_path is None or not source_path.exists():
            raise FileNotFoundError(f"Scenes JSON not found: {args.scenes_json}")
        scenes_payload = load_json(source_path)
        write_json(scenes_payload, scenes_path)
    else:
        scene_prompt = build_scene_prompt(
            topic,
            character_dicts,
            max_scenes=args.max_scenes,
        )
        write_text(scene_prompt, out_dir / "scenes_prompt.txt")
        scenes_payload = gpt_text_call(
            scene_prompt,
            client=llm_client,
            output_path=scenes_path,
            retries=args.retries,
            debug=args.debug_llm,
        )

    if not isinstance(scenes_payload, list):
        raise ValueError("Scenes payload must be a JSON list.")

    if args.script_outline:
        outline_text = maybe_load_outline(args.script_outline)
    else:
        outline_text = scenes_to_outline(scenes_payload)
    write_text(outline_text, outline_path)

    if args.dialogues_json:
        source_path = resolve_path(PROJECT_ROOT, args.dialogues_json)
        if source_path is None or not source_path.exists():
            raise FileNotFoundError(f"Dialogues JSON not found: {args.dialogues_json}")
        initial_dialogue = load_json(source_path)
        write_json(initial_dialogue, dialogue_path)
    else:
        dialogue_prompt = build_dialogue_prompt(outline_text)
        write_text(dialogue_prompt, out_dir / "dialogue_prompt.txt")
        initial_dialogue = gpt_text_call(
            dialogue_prompt,
            client=llm_client,
            output_path=dialogue_path,
            retries=args.retries,
            debug=args.debug_llm,
        )

    current_dialogue = initial_dialogue
    current_dialogue_text = dump_json_text(current_dialogue)

    if args.enable_feedback:
        write_json(initial_dialogue, out_dir / "dialogues_v0_initial.json")
        for round_index in range(1, args.feedback_rounds + 1):
            actor_prompt = build_actor_feedback_prompt(current_dialogue_text, character_dicts)
            write_text(actor_prompt, out_dir / f"actor_prompt_round{round_index}.txt")
            actor_feedback = gpt_text_call(
                actor_prompt,
                client=llm_client,
                output_path=out_dir / f"feedback_actor_round{round_index}.txt",
                retries=args.retries,
                debug=args.debug_llm,
                expect_json=False,
            )

            actor_revision_prompt = build_actor_revision_prompt(
                current_dialogue_text,
                str(actor_feedback),
                outline_text,
            )
            write_text(actor_revision_prompt, out_dir / f"actor_revision_prompt_round{round_index}.txt")
            actor_revised_dialogue = gpt_text_call(
                actor_revision_prompt,
                client=llm_client,
                output_path=out_dir / f"dialogues_v{round_index}_actor_revised.json",
                retries=args.retries,
                debug=args.debug_llm,
            )
            actor_revised_text = dump_json_text(actor_revised_dialogue)

            director_prompt = build_director_review_prompt(
                actor_revised_text,
                str(actor_feedback),
                outline_text,
            )
            write_text(director_prompt, out_dir / f"director_prompt_round{round_index}.txt")
            director_feedback = gpt_text_call(
                director_prompt,
                client=llm_client,
                output_path=out_dir / f"feedback_director_round{round_index}.txt",
                retries=args.retries,
                debug=args.debug_llm,
                expect_json=False,
            )
            director_feedback_text = str(director_feedback)

            if "APPROVED" in director_feedback_text.upper():
                current_dialogue = actor_revised_dialogue
                current_dialogue_text = actor_revised_text
            else:
                director_revision_prompt = build_director_revision_prompt(
                    actor_revised_text,
                    director_feedback_text,
                    str(actor_feedback),
                    outline_text,
                )
                write_text(
                    director_revision_prompt,
                    out_dir / f"director_revision_prompt_round{round_index}.txt",
                )
                current_dialogue = gpt_text_call(
                    director_revision_prompt,
                    client=llm_client,
                    output_path=out_dir / f"dialogues_v{round_index}_director_revised.json",
                    retries=args.retries,
                    debug=args.debug_llm,
                )
                current_dialogue_text = dump_json_text(current_dialogue)

            write_json(current_dialogue, out_dir / f"dialogues_v{round_index}_final.json")

    write_json(current_dialogue, final_dialogue_path)

    if not args.skip_long_prompts:
        long_prompt_task = build_long_prompt_detailer(
            dump_json_text(scenes_payload),
            dump_json_text(current_dialogue),
        )
        write_text(long_prompt_task, out_dir / "long_prompt_task.txt")
        gpt_text_call(
            long_prompt_task,
            client=llm_client,
            output_path=long_prompts_path,
            retries=args.retries,
            debug=args.debug_llm,
        )

    print(f"Stage 1 outputs saved to: {out_dir}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Default API key source present: {'yes' if DEFAULT_API_KEY else 'no'}")
    print(f"Default base URL: {DEFAULT_BASE_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
