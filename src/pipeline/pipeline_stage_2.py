"""Stage 2 pipeline: screenplay scene to Holodeck-generated scene JSON.

This wrapper keeps Stage 2 code inside Mind-of-Director while using the local
Holodeck checkout as the generation backend.

Run from Mind-of-Director:
    python -m src.pipeline.pipeline_stage_2 --query "a compact living room with a sofa, coffee table, bookshelf, photo album, candles, and wine glasses"
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from prompts.prompt_stage2 import build_deterministic_holodeck_query, build_holodeck_query_prompt
from src.utils.io import extract_json_payload, load_json, resolve_path, write_json, write_text
from src.utils.llm import GPTClient, gpt_text_call, resolve_api_key, resolve_base_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_scene(scene_json: Path, scene_index: int) -> Mapping[str, Any]:
    data = load_json(scene_json)
    if isinstance(data, list):
        if not (1 <= scene_index <= len(data)):
            raise IndexError(f"scene_index {scene_index} out of range for {scene_json}")
        return data[scene_index - 1]
    if isinstance(data, dict):
        return data
    raise TypeError(f"Unsupported scene json payload: {scene_json}")


def resolve_dashscope_key() -> str:
    key = resolve_api_key()
    if key:
        return key
    secrets_path = PROJECT_ROOT / "src" / "utils" / "mysecrets.py"
    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets file not found: {secrets_path}")
    secrets = runpy.run_path(str(secrets_path))
    dashscope_key = secrets.get("DASHSCOPE_API_KEY")
    if not dashscope_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing in src/utils/mysecrets.py.")
    return str(dashscope_key)


def build_query(args: argparse.Namespace, output_dir: Path) -> str:
    if args.query:
        query = args.query.strip()
        write_text(query, output_dir / "holodeck_query.txt")
        return query

    if not args.scene_json:
        raise RuntimeError("No scene source. Pass --query or --scene-json.")
    scene_path = resolve_path(PROJECT_ROOT, args.scene_json)
    if scene_path is None or not scene_path.exists():
        raise FileNotFoundError(f"Scene JSON not found: {args.scene_json}")
    scene = load_scene(scene_path, args.scene_index)
    write_json(scene, output_dir / "stage1_scene_input.json")

    if not args.refine_query:
        query = build_deterministic_holodeck_query(scene)
        write_text(query, output_dir / "holodeck_query.txt")
        return query

    prompt = build_holodeck_query_prompt(scene)
    write_text(prompt, output_dir / "holodeck_query_prompt.txt")
    client = GPTClient(api_key=resolve_dashscope_key(), model=args.llm_model, base_url=resolve_base_url())
    response = gpt_text_call(prompt, client=client, output_path=output_dir / "holodeck_query_raw.json", retries=args.retries)
    if isinstance(response, dict):
        query = str(response.get("query", "")).strip()
    else:
        query = str(extract_json_payload(str(response)).get("query", "")).strip()
    if not query:
        raise ValueError("Refined Holodeck query is empty.")
    write_text(query, output_dir / "holodeck_query.txt")
    return query


def run_holodeck(args: argparse.Namespace, query: str, output_dir: Path) -> None:
    holodeck_root = resolve_path(PROJECT_ROOT, args.holodeck_root, default=PROJECT_ROOT / "third_party" / "holodeck")
    assets_base = resolve_path(PROJECT_ROOT, args.objathor_assets_base, default=PROJECT_ROOT / "dataset" / ".objathor-assets")
    if holodeck_root is None or assets_base is None:
        raise RuntimeError("Holodeck paths could not be resolved.")
    if not holodeck_root.exists():
        raise FileNotFoundError(f"Holodeck root not found: {holodeck_root}")
    if not assets_base.exists():
        raise FileNotFoundError(f"Objathor assets base not found: {assets_base}")

    key = resolve_dashscope_key()
    if not key:
        raise RuntimeError("Missing DashScope/OpenAI-compatible API key.")

    holodeck_save_dir = output_dir / "holodeck_scene"
    holodeck_save_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(holodeck_root),
            "OBJATHOR_ASSETS_BASE_DIR": str(assets_base),
            "DASHSCOPE_API_KEY": key,
            "DASHSCOPE_BASE_URL": args.base_url or resolve_base_url(),
            "HOLODECK_LLM_MODEL": args.llm_model,
            "HOLODECK_USE_SBERT": "1" if args.use_sbert else "0",
            "MPLCONFIGDIR": str(output_dir / ".matplotlib"),
        }
    )

    command = [
        args.python,
        "-m",
        "ai2holodeck.main",
        "--mode",
        "generate_single_scene",
        "--query",
        query,
        "--save_dir",
        str(holodeck_save_dir),
        "--generate_image",
        str(bool(args.generate_image)),
        "--generate_video",
        str(bool(args.generate_video)),
        "--add_ceiling",
        str(bool(args.add_ceiling)),
        "--single_room",
        str(bool(args.single_room)),
        "--use_constraint",
        str(bool(args.use_constraint)),
        "--use_milp",
        str(bool(args.use_milp)),
        "--random_selection",
        str(bool(args.random_selection)),
        "--llm_model",
        args.llm_model,
        "--base_url",
        args.base_url or resolve_base_url(),
    ]

    write_json({"command": command, "cwd": str(holodeck_root)}, output_dir / "holodeck_command.json")
    result = subprocess.run(
        command,
        cwd=holodeck_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    write_text(result.stdout, output_dir / "holodeck_stdout.txt")
    write_text(result.stderr, output_dir / "holodeck_stderr.txt")
    if result.returncode != 0:
        raise RuntimeError(f"Holodeck failed with code {result.returncode}. See {output_dir / 'holodeck_stderr.txt'}")


def find_generated_scene(output_dir: Path) -> Path | None:
    candidates = sorted((output_dir / "holodeck_scene").glob("**/*.json"))
    scene_candidates = [
        path for path in candidates
        if path.name not in {"holodeck_command.json"} and not path.name.startswith(".")
    ]
    return scene_candidates[-1] if scene_candidates else None


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir, default=PROJECT_ROOT / "log" / "stage2" / f"scene_{args.scene_index:02d}")
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    query = build_query(args, output_dir)

    if args.run_holodeck:
        run_holodeck(args, query, output_dir)
        scene_path = find_generated_scene(output_dir)
        if scene_path is None:
            raise RuntimeError(f"Holodeck finished but no generated scene JSON was found under {output_dir / 'holodeck_scene'}")
    else:
        scene_path = None

    manifest = {
        "stage": "stage2",
        "backend": "holodeck",
        "query": query,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
        "holodeck_ran": bool(args.run_holodeck),
        "scene_json": str(scene_path.relative_to(PROJECT_ROOT) if scene_path and scene_path.is_relative_to(PROJECT_ROOT) else scene_path) if scene_path else None,
        "notes": "No StageDesigner, no background image generation, no Unity connection.",
    }
    write_json(manifest, output_dir / "stage2_manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mind-of-Director Stage2 Holodeck wrapper")
    parser.add_argument("--query", default=None, help="Direct Holodeck query. If omitted, reads --scene-json.")
    parser.add_argument("--scene-json", default=None, help="Stage1 scenes JSON. Relative to Mind-of-Director unless absolute.")
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--refine-query", action="store_true", help="Use Qwen to convert scene JSON into a cleaner Holodeck query.")
    parser.add_argument("--run-holodeck", action="store_true", help="Actually call Holodeck. Without this, only writes query/manifest.")
    parser.add_argument("--holodeck-root", default="third_party/holodeck")
    parser.add_argument("--objathor-assets-base", default="dataset/.objathor-assets")
    parser.add_argument("--python", default="python")
    parser.add_argument("--llm-model", default="qwen-max-latest")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--use-sbert", action="store_true")
    parser.add_argument("--generate-image", action="store_true")
    parser.add_argument("--generate-video", action="store_true")
    parser.add_argument("--add-ceiling", action="store_true")
    parser.add_argument("--single-room", action="store_true", default=True)
    parser.add_argument("--use-constraint", action="store_true", default=True)
    parser.add_argument("--use-milp", action="store_true")
    parser.add_argument("--random-selection", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
