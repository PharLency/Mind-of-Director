# Stage4 Runbook

Stage4 负责相机规划：给每个 dialogue clip 分配一个 camera segment，并和 clip 时间线对齐。

## 当前代码位置

```bash
prompts/prompt_stage4.py
src/pipeline/pipeline_stage_4.py
examples/assets/camera_library.json
```

示例输入已经放在项目内：

```bash
examples/script_101/s3/scene_01/clip_time/clip_time.json
examples/script_101/s3/scene_01/topdown_images/topdown_detect.png
examples/script_101/s3/scene_01/topdown_annotated.png
```

默认输出：

```bash
log/stage4/scene_01/
```

## Stage4 输入

必须显式指定：

```bash
log/stage3/scene_01/blocking/final_blocking.json
log/stage3/scene_01/motion/motion_selection.json
examples/script_101/s3/scene_01/clip_time/clip_time.json
examples/script_101/s3/scene_01/topdown_images/topdown_detect.png
examples/script_101/s3/scene_01/topdown_annotated.png
examples/assets/camera_library.json
```

现在不再做隐式回退：路径缺失会直接报错，方便及时发现上游输出没有准备好。

## 相机库

新的相机库是结构化 JSON，不再是一大段松散文本。v2.0 的核心是 coverage-first：先决定电影覆盖方式，再决定运动方式。里面包含：

```text
enums: coverage_role / dramatic_function / position / direction / angle / shot_size / framing / lens_band / movement 等
shot_size_guide: ECU/CU/MCU/MS/MLS/WS 的用途
templates: 每种镜头模板的 required_fields、适用条件、禁忌
selection_rules: static/walking/single/two/group 对应的推荐模板
continuity_rules: 轴线、视线、宽镜重置、特写节制、运动动机等规则
```

核心模板包括：

```text
master_wide
ensemble_three_shot
two_shot
over_the_shoulder
clean_single
dirty_single
reaction_single
insert_detail
pov_subjective
push_in_single
tracking_single
pan_reframe
rack_focus_between
crosscut_pair
```

旧模板 `singlestatic`、`two_static`、`full_shot` 仍能读取，但会被规范化成新模板。

## 快速验证

使用本地规则生成基础相机方案，不调用 API：

```bash
conda activate previs
cd Mind-of-Director
python -m src.pipeline.pipeline_stage_4 \
  --script-root examples/script_101 \
  --scene-index 1 \
  --blocking-json log/stage3/scene_01/blocking/final_blocking.json \
  --motion-json log/stage3/scene_01/motion/motion_selection.json \
  --clip-time-json examples/script_101/s3/scene_01/clip_time/clip_time.json \
  --detect-image examples/script_101/s3/scene_01/topdown_images/topdown_detect.png \
  --annotated-image examples/script_101/s3/scene_01/topdown_annotated.png
```

已验证输出：

```bash
log/stage4/scene_01/camera/camera_segments.json
log/stage4/scene_01/camera/validation_warnings.json
log/stage4/scene_01/stage4_manifest.json
```

当前测试结果：

```text
validation_warnings: []
```

## 本地规则生成

如果不想使用已有 camera plan，也不想调用 VLM，可以用 deterministic 模式：

```bash
python -m src.pipeline.pipeline_stage_4 \
  --script-root examples/script_101 \
  --scene-index 1 \
  --camera-source deterministic \
  --blocking-json log/stage3/scene_01/blocking/final_blocking.json \
  --motion-json log/stage3/scene_01/motion/motion_selection.json \
  --clip-time-json examples/script_101/s3/scene_01/clip_time/clip_time.json \
  --detect-image examples/script_101/s3/scene_01/topdown_images/topdown_detect.png \
  --annotated-image examples/script_101/s3/scene_01/topdown_annotated.png \
  --output-dir log/stage4/scene_01_deterministic
```

这个模式会根据 speaker、dialogue beat、blocking 状态和 group reset 规则生成一个基础相机方案。

## 重新调用 Qwen-VL 生成

需要读入两张图：

```bash
examples/script_101/s3/scene_01/topdown_images/topdown_detect.png
examples/script_101/s3/scene_01/topdown_annotated.png
```

运行：

```bash
python -m src.pipeline.pipeline_stage_4 \
  --script-root examples/script_101 \
  --scene-index 1 \
  --camera-source generate \
  --blocking-json log/stage3/scene_01/blocking/final_blocking.json \
  --motion-json log/stage3/scene_01/motion/motion_selection.json \
  --clip-time-json examples/script_101/s3/scene_01/clip_time/clip_time.json \
  --detect-image examples/script_101/s3/scene_01/topdown_images/topdown_detect.png \
  --annotated-image examples/script_101/s3/scene_01/topdown_annotated.png \
  --vlm-model qwen-vl-plus
```

生成流程：

```text
Cinematographer_A plan
Cinematographer_B plan
A reviews B
B reviews A
Director synthesis
normalize to camera_segments
validate
```

## 最终输出格式

最终重点文件是：

```bash
log/stage4/scene_01/camera/camera_segments.json
```

每个 segment 形如：

```json
{
  "clip_id": "clip_01",
  "start": 0.0,
  "end": 3.79,
  "duration_seconds": 3.79,
  "speaker": "Clara",
  "primary_subjects": ["Clara", "Maya"],
  "dramatic_function": "tension",
  "shot": {
    "type": "two_shot",
    "subjects": ["Clara", "Maya"],
    "relation": "distant",
    "framing": "two_shot",
    "shot_size": "MS",
    "angle": "Eye",
    "rationale": "..."
  },
  "rationale": "...",
  "motion_actions": []
}
```

## 注意事项

`camera-source existing` 不调用模型，只把已有 camera plan 复制并规范化。

`camera-source deterministic` 不调用模型，用规则生成基础方案。

`camera-source generate` 会调用 Qwen-VL，耗时和费用更高。

旧 camera plan 里的 `singlestatic`、`two_static`、`full_shot` 比较粗糙；新 pipeline 会自动转换为 `clean_single`、`two_shot` / `over_the_shoulder`、`ensemble_three_shot`，并补齐 lens、focus、movement、screen_direction 等字段。
