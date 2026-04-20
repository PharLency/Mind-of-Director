# Mind-of-Director

[arXiv](https://arxiv.org/abs/2603.14790) | [Project Page](https://pharlency.github.io/Mind-of-Director/)

We present **Mind-of-Director**, a multi-modal agent-driven framework for film previz that models the collaborative decision-making process of a film production team.

Given a creative idea, **Mind-of-Director** orchestrates multiple specialized agents to produce film previz sequences.

The framework consists of four cooperative modules: *Script Development*, where agents draft and refine the screenplay iteratively; *Virtual Scene Design*, which transforms text into semantically aligned 3D environments; *Character Behaviour Control*, which determines character blocking and motion; and *Camera Planning*, which optimizes framing, movement, and composition for cinematic camera effects.

## Released Modules

- [x] `prompts`
- [x] `python pipeline`
- [ ] `Unity Assets`

## Project Layout

```text
prompts/                 Prompt builders for stages 1-4
src/pipeline/            Main stage pipelines
src/utils/               Shared IO and LLM helpers
examples/                Small script_101 sample inputs
third_party/holodeck/    Bundled Holodeck backend code
unity/                   Reserved for future Unity integration
log/                     Local generated outputs, ignored by Git
```

## Setup

Use Python 3.10. A conda environment is recommended:

```bash
conda create -n previs python=3.10
conda activate previs
pip install -r requirements.txt
```

Configure DashScope with environment variables. Do not commit real keys.

```bash
cp .env.example .env
export DASHSCOPE_API_KEY="your_dashscope_key"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export HOLODECK_LLM_MODEL="qwen-max-latest"
```

## Download Holodeck / Objathor Data

The Holodeck code is bundled in `third_party/holodeck`, but the Objathor assets are not included because they are large external data.

Download the assets into a local `dataset/.objathor-assets` directory:

```bash
mkdir -p dataset/.objathor-assets
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23 --path dataset/.objathor-assets
python -m objathor.dataset.download_assets --version 2023_09_23 --path dataset/.objathor-assets
python -m objathor.dataset.download_annotations --version 2023_09_23 --path dataset/.objathor-assets
python -m objathor.dataset.download_features --version 2023_09_23 --path dataset/.objathor-assets
export OBJATHOR_ASSETS_BASE_DIR="$PWD/dataset/.objathor-assets"
```

If you keep the dataset elsewhere, pass `--objathor-assets-base /path/to/.objathor-assets` when running Stage2.

## Run The Pipeline

Stage1, idea to structured screenplay:

```bash
python -m src.pipeline.pipeline_stage_1 "A tense conversation around a family photo album" \
  --out-dir log/stage1
```

Stage2, screenplay scene to Holodeck query or scene:

```bash
python -m src.pipeline.pipeline_stage_2 \
  --scene-json log/stage1/scenes.json \
  --scene-index 1 \
  --output-dir log/stage2
```

Add `--run-holodeck` when the Objathor dataset is ready and you want to generate the actual Holodeck scene JSON:

```bash
python -m src.pipeline.pipeline_stage_2 \
  --scene-json log/stage1/scenes.json \
  --scene-index 1 \
  --output-dir log/stage2 \
  --run-holodeck \
  --objathor-assets-base dataset/.objathor-assets
```

Stage3, blocking / motion / audio from a scene folder:

```bash
python -m src.pipeline.pipeline_stage_3 \
  --script-root examples/script_101 \
  --scene-index 1 \
  --output-dir log/stage3/scene_01
```

Stage4, camera segments:

```bash
python -m src.pipeline.pipeline_stage_4 \
  --script-root examples/script_101 \
  --scene-index 1 \
  --blocking-json log/stage3/scene_01/blocking/final_blocking.json \
  --motion-json log/stage3/scene_01/motion/motion_selection.json \
  --clip-time-json examples/script_101/s3/scene_01/clip_time/clip_time.json \
  --detect-image examples/script_101/s3/scene_01/topdown_images/topdown_detect.png \
  --annotated-image examples/script_101/s3/scene_01/topdown_annotated.png \
  --output-dir log/stage4/scene_01
```

## Notes

Generated files under `log/` are ignored by Git. Keep only reproducible code, prompts, sample inputs, and lightweight examples in the repository.

Holodeck is included under `third_party/holodeck` with its original license. The dataset must be downloaded separately.
