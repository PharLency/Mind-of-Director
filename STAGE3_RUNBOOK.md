# Stage3 Runbook

Stage3 负责把静态场景变成“可表演的预演数据”：人物站位/坐位、每句台词对应的动作、对白声音，以及可坐区域检查。

## 当前代码位置

```bash
prompts/prompt_stage3.py
src/pipeline/pipeline_stage_3.py
```

测试输入已经复制到项目内：

```bash
examples/script_101/s1/
examples/script_101/s3/scene_01/
examples/assets/motion_library.json
```

输出统一写到：

```bash
log/stage3/scene_01/
```

## Stage3 逻辑

1. 读取 Stage1 剧本信息：`outline.txt`、`dialogues.json`、`characters.json`。
2. 读取场景俯视图信息：`topdown_detect.png`、`topdown_annotated.png`、`objects.json`、`topdown_labels.json`。
3. Blocking：用 VLM 根据两张俯视图和标注点生成人物走位、坐/站状态、朝向。
4. Blocking 校验：检查点位是否合法、同一 clip 是否抢点、S* 座位朝向是否为 null、clip 间连续性是否断裂。
5. Motion：根据 blocking 的状态变化，从 `motion_library.json` 给每个角色每个 clip 选择一个动作。
6. Motion 校验：检查每个 clip 是否覆盖所有角色、动作 id 是否存在。
7. Audio：根据 `final_blocking.json` 的 speaker/content，用 DashScope CosyVoice 生成每句台词音频。
8. Manifest：汇总产物和 warning 到 `stage3_manifest.json`。

## 快速验证

默认使用已经复制到 `examples/` 的 existing 结果，不调用 API：

```bash
conda activate previs
cd Mind-of-Director
python -m src.pipeline.pipeline_stage_3 \
  --script-root examples/script_101
```

已验证该命令能生成：

```bash
log/stage3/scene_01/blocking/final_blocking.json
log/stage3/scene_01/blocking/validation_warnings.json
log/stage3/scene_01/motion/motion_selection.json
log/stage3/scene_01/motion/validation_warnings.json
log/stage3/scene_01/audio/audio_manifest.json
log/stage3/scene_01/stage3_manifest.json
```

当前测试结果：

```text
blocking_warnings: []
motion_warnings: []
audio_errors: []
```

## 重新生成 Blocking

需要调用 Qwen-VL：

```bash
python -m src.pipeline.pipeline_stage_3 \
  --script-root examples/script_101 \
  --scene-index 1 \
  --blocking-source generate \
  --motion-source existing \
  --audio-source skip \
  --vlm-model qwen-vl-plus
```

Blocking 会使用：

```bash
examples/script_101/s3/scene_01/topdown_images/topdown_detect.png
examples/script_101/s3/scene_01/topdown_annotated.png
```

## 重新生成 Motion

需要先有 `blocking/final_blocking.json`：

```bash
python -m src.pipeline.pipeline_stage_3 \
  --script-root examples/script_101 \
  --scene-index 1 \
  --blocking-source existing \
  --motion-source generate \
  --audio-source skip \
  --llm-model qwen-max-latest
```

动作硬规则：

```text
position_change=true -> action 37 Walking Middle Speed
standing -> sitting -> action 38 Stand To Sit
sitting -> standing -> action 39 Stand Up
standing -> standing -> standing_actions 中选择
sitting -> sitting -> sitting_actions 中选择
```

## 重新生成 Audio

需要 DashScope key：

```bash
python -m src.pipeline.pipeline_stage_3 \
  --script-root examples/script_101 \
  --scene-index 1 \
  --blocking-source existing \
  --motion-source existing \
  --audio-source generate \
  --tts-model cosyvoice-v2
```

角色音色映射会写入：

```bash
log/stage3/scene_01/voices/voice_assignments.json
```

## 注意事项

`blocking-source existing`、`motion-source existing`、`audio-source existing` 是为了复现实验结果，不会调用模型。

如果要完整重跑，改成：

```bash
python -m src.pipeline.pipeline_stage_3 \
  --script-root examples/script_101 \
  --scene-index 1 \
  --blocking-source generate \
  --motion-source generate \
  --audio-source generate
```

这会调用 Qwen-VL、Qwen 文本模型和 DashScope TTS，耗时和费用都会明显增加。
