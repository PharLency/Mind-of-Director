# Stage2 Holodeck Runbook

Stage2 takes a structured screenplay scene and generates a Holodeck 3D scene. The project no longer uses StageDesigner for Stage2.

## Code

```bash
prompts/prompt_stage2.py
src/pipeline/pipeline_stage_2.py
third_party/holodeck/
```

## Required External Data

Holodeck code is bundled in `third_party/holodeck`, but Objathor assets are not committed. Download them locally:

```bash
mkdir -p dataset/.objathor-assets
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23 --path dataset/.objathor-assets
python -m objathor.dataset.download_assets --version 2023_09_23 --path dataset/.objathor-assets
python -m objathor.dataset.download_annotations --version 2023_09_23 --path dataset/.objathor-assets
python -m objathor.dataset.download_features --version 2023_09_23 --path dataset/.objathor-assets
export OBJATHOR_ASSETS_BASE_DIR="$PWD/dataset/.objathor-assets"
```

The two feature files used by retrieval should then exist under:

```bash
dataset/.objathor-assets/2023_09_23/features/clip_features.pkl
dataset/.objathor-assets/2023_09_23/features/sbert_features.pkl
```

## Environment

```bash
conda activate previs
export DASHSCOPE_API_KEY="your_dashscope_key"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export HOLODECK_LLM_MODEL="qwen-max-latest"
```

Do not write real API keys into `src/utils/mysecrets.py`, README files, logs, or Git history.

## Dry Run

This only converts the scene into a Holodeck query and writes a manifest. It does not call Holodeck.

```bash
python -m src.pipeline.pipeline_stage_2 \
  --scene-json examples/script_101/s1/scenes.json \
  --scene-index 1 \
  --output-dir log/stage2
```

## Full Holodeck Run

```bash
python -m src.pipeline.pipeline_stage_2 \
  --scene-json examples/script_101/s1/scenes.json \
  --scene-index 1 \
  --output-dir log/stage2 \
  --run-holodeck \
  --objathor-assets-base dataset/.objathor-assets
```

Important defaults:

```text
--holodeck-root third_party/holodeck
--objathor-assets-base dataset/.objathor-assets
--generate-image False unless explicitly enabled
--generate-video False unless explicitly enabled
--add-ceiling False unless explicitly enabled
```

## Direct Holodeck Invocation

If you need to debug the bundled Holodeck backend directly:

```bash
PYTHONPATH=third_party/holodeck \
OBJATHOR_ASSETS_BASE_DIR="$PWD/dataset/.objathor-assets" \
python -m ai2holodeck.main \
  --mode generate_single_scene \
  --query "a compact living room table arrangement with photo album and wine glass" \
  --save_dir log/holodeck_direct_test \
  --generate_image False \
  --generate_video False \
  --add_ceiling False \
  --single_room True \
  --use_constraint True \
  --use_milp False \
  --random_selection False
```

## Outputs

Stage2 writes generated files under `log/`, which is intentionally ignored by Git.

```bash
log/stage2/holodeck_query.txt
log/stage2/stage2_manifest.json
log/stage2/holodeck_scene/
```
